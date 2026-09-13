"""OpenVINO runtime for LangACT.

Same shape as policy_ov.py, plus the language token. The sentence encoder does
not run here: embeddings are cached on the host by policy_langact.LangACT, so
the per-step cost of language is a dict lookup. That is the claim the whole
architecture rests on, and `--report-embed-cost` measures it rather than
asserting it.

The action queue stays on the host, as with ACT: it is Python control flow, and
tracing it would bake one queue state into the graph.
"""
import argparse
import collections
import logging

import numpy as np
import openvino as ov
import torch

from policy_langact import FEATURE, TRAINED_INSTRUCTIONS, LangACT

LOG = logging.getLogger("langact_ov")


class LangACTOpenVINO:
    def __init__(self, xml="ir/langact_int8.xml", device="CPU", base=None,
                 checkpoint=None):
        if base is None:
            if checkpoint is None:
                raise ValueError("pass either an existing LangACT or a checkpoint")
            base = LangACT(checkpoint).warm()
        self.base = base
        self.pre, self.post = base.pre, base.post
        self.n_action_steps = base.policy.config.n_action_steps

        core = ov.Core()
        model = core.read_model(xml)
        # LATENCY, not THROUGHPUT: a robot runs one request at a time, and the
        # number that matters is how long a single decision takes.
        self.compiled = core.compile_model(model, device,
                                           {"PERFORMANCE_HINT": "LATENCY"})
        self.out = self.compiled.output(0)
        self.name = f"langact_ov_{'int8' if 'int8' in xml else 'fp32'}_{device.lower()}"
        self.queue = collections.deque()
        self.did_forward = False

    def reset(self):
        self.queue.clear()
        self.did_forward = False

    def act(self, obs, instruction=None):
        if instruction is None:
            raise ValueError(
                "LangACT requires an instruction; accepting None would let the "
                "policy score while ignoring language"
            )
        self.did_forward = not self.queue
        if not self.queue:
            batch = {
                "observation.state": torch.from_numpy(
                    np.asarray(obs["agent_pos"], dtype=np.float32)
                ),
                "observation.images.top": torch.from_numpy(
                    np.asarray(obs["pixels"]["top"], dtype=np.float32) / 255.0
                ).permute(2, 0, 1),
                # Cached; no encoder runs here.
                FEATURE: torch.from_numpy(self.base.embed(instruction)),
            }
            processed = self.pre(batch)
            image = processed["observation.images.top"].cpu().numpy()
            state = processed["observation.state"].cpu().numpy()
            lang = processed[FEATURE].cpu().numpy()
            chunk = self.compiled([image, state, lang])[self.out]  # (1, chunk, 14)
            for a in chunk[0][: self.n_action_steps]:
                self.queue.append(a)
        action = self.queue.popleft()
        return self.post(torch.from_numpy(np.asarray(action))).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="ir/langact_int8.xml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    import gymnasium as gym
    import gym_aloha  # noqa: F401

    base = LangACT(args.checkpoint).warm()
    ovp = LangACTOpenVINO(args.xml, args.device, base=base)

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    instr = TRAINED_INSTRUCTIONS[0]

    base.reset(); ovp.reset()
    a_torch = base.act(obs, instr)
    a_ov = ovp.act(obs, instr)
    print("torch :", np.round(a_torch[:5], 5))
    print("ov    :", np.round(a_ov[:5], 5))
    print("max|diff| over 14 joints = %.3e" % np.max(np.abs(a_torch - a_ov)))

    # The runtime must still be listening to the sentence. Same observation,
    # same seed, other instruction: identical actions would mean the language
    # input is dead in this configuration even if it was live in PyTorch.
    ovp.reset()
    a_other = ovp.act(obs, TRAINED_INSTRUCTIONS[1])
    delta = np.max(np.abs(a_ov - a_other))
    print("max|action(instr0) - action(instr1)| = %.3e" % delta)
    if delta == 0.0:
        raise SystemExit("FAIL: this OpenVINO runtime ignores the instruction")

    # Substantiate "language costs 0 ms": the encoder ran during warm(), so the
    # per-step price of a sentence is a dict hit.
    t = time.perf_counter()
    for _ in range(10000):
        base.embed(instr)
    per = (time.perf_counter() - t) / 10000 * 1e6
    print(f"cached embedding lookup: {per:.3f} us/step")
    env.close()
