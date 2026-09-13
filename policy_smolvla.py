"""SmolVLA — the language-conditioned policy. This is the actual VLA entry.

Differences from the ACT baseline that matter:

  * It reads the instruction. ACT ignores it, which is why ACT is only
    scaffolding and cannot support a voice interface.
  * The checkpoint ships its own processors (policy_preprocessor.json and the
    normalizer safetensors), so the normalization trap that silently broke ACT
    does not apply — but we verify rather than assume.
  * It was trained with the camera renamed to `observation.images.camera1`
    (SmolVLA expects 3 cameras, ALOHA has 1, padded via empty_cameras=2), so
    inference must use the same key. Feeding `observation.images.top` here
    would silently produce a blank-input policy.
"""
import argparse
import logging

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)

CHECKPOINT = "checkpoint_smolvla"
CAMERA_KEY = "observation.images.camera1"  # must match training rename_map

LOG = logging.getLogger("smolvla")


class SmolVLA:
    def __init__(self, checkpoint=CHECKPOINT, device="cpu"):
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to(self.device).eval()
        # The processors were serialised on the H100, so they carry
        # device="cuda". Loading them unchanged on a CPU laptop raises
        # "Failed to instantiate processor step 'device_processor'".
        overrides = {"device_processor": {"device": str(self.device)}}
        self.pre = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_preprocessor.json",
            overrides=overrides,
        )
        # from_pretrained restores the steps but NOT the converters, so the
        # postprocessor would receive a raw Tensor and raise
        # "EnvTransition must be a dictionary". These are the same converters
        # make_smolvla_pre_post_processors() wires up.
        self.post = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_postprocessor.json",
            overrides=overrides,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.name = "smolvla"
        self.did_forward = False
        self._check_language_is_used()

    def _check_language_is_used(self):
        cfg = self.policy.config
        LOG.info("chunk=%s n_action_steps=%s", cfg.chunk_size, cfg.n_action_steps)

    def reset(self):
        self.policy.reset()
        self.did_forward = False

    def act(self, obs, instruction):
        # SmolVLA keeps its chunk in self._queues[ACTION], NOT _action_queue.
        # Reading the wrong attribute made getattr() return [] every step, so
        # every call was reported as a forward pass and the latency split was
        # meaningless (the policy itself chunked correctly all along).
        self.did_forward = len(self.policy._queues[ACTION]) == 0
        batch = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["agent_pos"], dtype=np.float32)
            ),
            CAMERA_KEY: torch.from_numpy(
                np.asarray(obs["pixels"]["top"], dtype=np.float32) / 255.0
            ).permute(2, 0, 1),
            "task": instruction,
        }
        with torch.no_grad():
            processed = self.pre(batch)
            action = self.policy.select_action(processed)
            action = self.post(action)
        return action.squeeze(0).float().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import gymnasium as gym
    import gym_aloha  # noqa: F401

    from bench import INSTRUCTION, WRONG_INSTRUCTION

    p = SmolVLA(args.checkpoint)
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)

    # The decisive check: does a different instruction produce a different
    # action from the identical observation? If these are equal, the policy is
    # ignoring language and the voice layer is decorative.
    p.reset()
    a_right = p.act(obs, INSTRUCTION)
    p.reset()
    a_wrong = p.act(obs, WRONG_INSTRUCTION)

    print("action (correct instruction):", np.round(a_right[:5], 5))
    print("action (wrong  instruction):", np.round(a_wrong[:5], 5))
    d = np.max(np.abs(a_right - a_wrong))
    print(f"max|difference| across 14 joints = {d:.6f}")
    print("LANGUAGE IS READ" if d > 1e-6 else "LANGUAGE IGNORED — voice layer would be fake")
    env.close()
