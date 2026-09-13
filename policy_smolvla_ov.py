"""SmolVLA running its flow-matching sampler on OpenVINO.

Everything outside the traced graph stays on the host and is shared with the
PyTorch path, so the ONLY difference between configurations is where the matrix
multiplies happen:

  * tokenisation + preprocessing  -> host (identical objects)
  * action queue                  -> host (Python control flow)
  * sample_actions                -> OpenVINO IR

Noise is drawn on the host with a seeded generator and passed in, so a rollout
is reproducible and PyTorch/OpenVINO can be compared on identical noise.
"""
import argparse
import collections
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402
import torch  # noqa: E402

from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from policy_smolvla import CAMERA_KEY, SmolVLA  # noqa: E402


class SmolVLAOpenVINO:
    def __init__(self, xml="ir/smolvla_int8.xml", device="GPU",
                 checkpoint="checkpoint_smolvla8k", seed=0, base=None,
                 n_action_steps=None):
        self.base = base or SmolVLA(checkpoint)
        self.policy = self.base.policy
        cfg = self.policy.config
        # How many actions to execute open-loop before re-observing. Trained
        # chunk is 50, but nothing forces us to consume all of it: replanning
        # more often trades latency (which OpenVINO bought us) for closed-loop
        # correction. Overridable WITHOUT retraining.
        self.n_action_steps = n_action_steps or cfg.n_action_steps
        self.chunk_size = cfg.chunk_size
        self.max_action_dim = cfg.max_action_dim
        self.action_dim = cfg.action_feature.shape[0]

        core = ov.Core()
        self.compiled = core.compile_model(
            core.read_model(xml), device, {"PERFORMANCE_HINT": "LATENCY"}
        )
        self.out = self.compiled.output(0)
        self.name = (f"smolvla_ov_{'int8' if 'int8' in xml else 'fp32'}"
                     f"_{device.lower()}_n{self.n_action_steps}")
        self.rng = np.random.default_rng(seed)
        self.queue = collections.deque()
        self.did_forward = False

    def reset(self):
        self.queue.clear()
        self.did_forward = False

    def _inputs(self, obs, instruction):
        batch = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["agent_pos"], dtype=np.float32)
            ),
            CAMERA_KEY: torch.from_numpy(
                np.asarray(obs["pixels"]["top"], dtype=np.float32) / 255.0
            ).permute(2, 0, 1),
            "task": instruction,
        }
        processed = self.base.pre(batch)
        prepared = self.policy._prepare_batch(processed)
        images, img_masks = self.policy.prepare_images(prepared)
        state = self.policy.prepare_state(prepared)
        noise = torch.from_numpy(
            self.rng.standard_normal(
                (state.shape[0], self.chunk_size, self.max_action_dim)
            ).astype(np.float32)
        )
        flat = list(images) + list(img_masks) + [
            prepared[OBS_LANGUAGE_TOKENS],
            prepared[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            noise,
        ]
        return [t.float().numpy() if t.dtype.is_floating_point else t.numpy()
                for t in flat]

    def act(self, obs, instruction):
        self.did_forward = not self.queue
        if not self.queue:
            chunk = self.compiled(self._inputs(obs, instruction))[self.out]
            # Unpad: the graph emits max_action_dim, the robot has action_dim.
            chunk = chunk[0][: self.n_action_steps, : self.action_dim]
            for a in chunk:
                self.queue.append(a)
        action = self.queue.popleft()
        return self.base.post(
            torch.from_numpy(np.asarray(action))
        ).squeeze(0).float().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="ir/smolvla_int8.xml")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--checkpoint", default="checkpoint_smolvla8k")
    a = ap.parse_args()

    import gym_aloha  # noqa: F401
    import gymnasium as gym

    from bench import INSTRUCTION, WRONG_INSTRUCTION

    base = SmolVLA(a.checkpoint)
    ovp = SmolVLAOpenVINO(a.xml, a.device, base=base)

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)

    # Verify the OpenVINO path still reads language — quantization could in
    # principle wash out the instruction signal, which would silently turn the
    # voice interface back into decoration.
    ovp.reset()
    right = ovp.act(obs, INSTRUCTION)
    ovp.rng = np.random.default_rng(0)  # same noise for a fair comparison
    ovp.reset()
    wrong = ovp.act(obs, WRONG_INSTRUCTION)
    d = float(np.max(np.abs(right - wrong)))
    print(f"OpenVINO {a.device}: max|diff| between instructions = {d:.6f}")
    print("LANGUAGE STILL READ AFTER INT8" if d > 1e-4
          else "WARNING: instruction signal lost in quantization")
    env.close()
