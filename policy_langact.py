"""LangACT — stock ACT that reads language, at ACT latency.

ACT builds its encoder input as [latent, robot_state, env_state, image_features]
and `env_state` is an unused slot behind a plain nn.Linear (modeling_act.py:344,
:465). Writing a sentence embedding there makes ACT language-conditioned without
one line of new architecture. See make_lang_dataset.py for the dataset side.

Two properties that matter for the comparison against SmolVLA:

  * Language costs 0 ms at inference. Instructions are known strings, so their
    embeddings are computed once and cached. SmolVLA re-runs a 500M VLM every
    chunk — measured at ~908 ms of fixed prefix, 63% of its latency, and a floor
    no amount of quantization removes.
  * The policy stays ~52M params, so it quantizes and runs like ACT.

⚠️ The instruction is REQUIRED here. A language-conditioned policy that silently
accepts `None` and runs anyway is the exact failure this project exists to
detect — it would score like a working model while ignoring the sentence.
"""
import logging

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = "local/aloha_sim_multitask_lang"
FEATURE = "observation.environment_state"

# The two instructions the policy was trained on. Kept explicit so the voice
# layer can refuse anything else rather than silently steering on a stray vector.
TRAINED_INSTRUCTIONS = (
    "Pick up the cube with the right arm and transfer it to the left arm.",
    "Insert the peg into the socket.",
)


class LangACT:
    name = "langact_pytorch_cpu"

    def __init__(self, checkpoint, dataset=DATASET, device="cpu"):
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device).eval()

        cfg = self.policy.config
        if FEATURE not in cfg.input_features:
            raise RuntimeError(
                f"{checkpoint} has no '{FEATURE}' input — this is stock ACT with "
                f"no language slot, not LangACT. Inputs: {list(cfg.input_features)}"
            )
        self.lang_dim = cfg.input_features[FEATURE].shape[0]

        stats = LeRobotDatasetMetadata(dataset).stats
        self.pre, self.post = make_act_pre_post_processors(cfg, dataset_stats=stats)
        self._assert_normalized(stats)

        self._cache = {}
        self._tok = self._mdl = None

    @staticmethod
    def _assert_normalized(stats):
        """Same trap as the ACT baseline: missing stats produce a policy that
        runs, returns finite actions, and fails the task, which reads as a weak
        model rather than as a loading bug."""
        if not stats:
            raise RuntimeError(f"no dataset stats for {DATASET}")
        for key in ("observation.state", "action"):
            if key not in stats:
                raise RuntimeError(f"dataset stats missing '{key}': got {list(stats)}")
            std = torch.as_tensor(stats[key]["std"])
            if not torch.isfinite(std).all() or (std == 0).any():
                raise RuntimeError(f"bad stats for '{key}' — would divide by 0")
        logging.info("LangACT normalization stats verified")

    def embed(self, text):
        """L2-normalised mean-pooled MiniLM embedding, cached per string.

        Must match make_lang_dataset.embed() exactly; a different pooling or a
        missing normalisation puts the token somewhere the policy never trained.
        """
        if text in self._cache:
            return self._cache[text]
        if self._mdl is None:
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(ENCODER)
            self._mdl = AutoModel.from_pretrained(ENCODER).eval()
        enc = self._tok([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self._mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vec = torch.nn.functional.normalize(pooled, dim=-1)[0].numpy().astype(np.float32)
        if vec.shape[0] != self.lang_dim:
            raise RuntimeError(f"embedding dim {vec.shape[0]} != policy {self.lang_dim}")
        self._cache[text] = vec
        return vec

    def warm(self, instructions=TRAINED_INSTRUCTIONS):
        """Pre-compute the embeddings so `act()` never pays for the encoder.
        This is what makes 'language costs 0 ms' true rather than a claim."""
        for t in instructions:
            self.embed(t)
        return self

    def reset(self):
        self.policy.reset()
        self.did_forward = False

    def act(self, obs, instruction=None):
        if instruction is None:
            raise ValueError(
                "LangACT requires an instruction; accepting None would let the "
                "policy score while ignoring language"
            )
        # ACT replays a cached chunk, so only the call that empties the queue
        # actually runs the network. Timing every call averages one ~350 ms
        # forward over ~100 queue pops and reports a latency nothing has.
        self.did_forward = len(self.policy._action_queue) == 0
        batch = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["agent_pos"], dtype=np.float32)
            ),
            "observation.images.top": torch.from_numpy(
                np.asarray(obs["pixels"]["top"], dtype=np.float32) / 255.0
            ).permute(2, 0, 1),
            FEATURE: torch.from_numpy(self.embed(instruction)),
        }
        with torch.no_grad():
            processed = self.pre(batch)
            action = self.policy.select_action(processed)
            action = self.post(action)
        return action.squeeze(0).float().cpu().numpy()


if __name__ == "__main__":
    import argparse
    import sys
    import time

    sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, force=True)  # lerobot grabs the root logger

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="outputs/langact_smoke/checkpoints/000002/pretrained_model")
    a = ap.parse_args()

    import gymnasium as gym
    import gym_aloha  # noqa: F401

    p = LangACT(a.checkpoint).warm()

    # The language token must survive the preprocessor unchanged: norm_map has no
    # ENV entry, so it is identity. If that ever changes, inference would feed a
    # differently-scaled vector than training and quietly degrade.
    v = p.embed(TRAINED_INSTRUCTIONS[0])
    got = p.pre({FEATURE: torch.from_numpy(v),
                 "observation.state": torch.zeros(14),
                 "observation.images.top": torch.zeros(3, 480, 640)})[FEATURE]
    delta = (got.squeeze(0).numpy() - v).max()
    print(f"language token passes preprocessor unchanged: max|delta| = {delta:.2e}")
    assert abs(delta) < 1e-6, "ENV is being normalized — inference != training"

    a0, a1 = (p.embed(t) for t in TRAINED_INSTRUCTIONS)
    print(f"cosine(instruction_0, instruction_1) = {float(a0 @ a1):.4f}")

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    p.reset()

    t = time.perf_counter()
    act0 = p.act(obs, TRAINED_INSTRUCTIONS[0])
    fwd = (time.perf_counter() - t) * 1000
    p.reset()
    act1 = p.act(obs, TRAINED_INSTRUCTIONS[1])

    print(f"forward (queue empty): {fwd:.0f} ms  did_forward={p.did_forward}")
    print(f"action {act0.shape} finite={np.isfinite(act0).all()}")
    # Same observation, same seed, different sentence: if the two actions are
    # identical the language token is not reaching the network at all.
    print(f"max|action(instr0) - action(instr1)| = {np.abs(act0 - act1).max():.6f}")

    t = time.perf_counter()
    for _ in range(1000):
        p.embed(TRAINED_INSTRUCTIONS[0])
    print(f"cached embed: {(time.perf_counter() - t):.3f} ms per 1000 lookups")
    env.close()
