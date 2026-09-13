"""OpenVINO runtime for the ACT policy.

Reuses the exact pre/post processors from the PyTorch baseline, so the only
thing that differs between configurations is where the matrix multiplies
happen. The action queue is reimplemented here on the host because it is
Python control flow, not part of the exported graph.
"""
import argparse
import collections
import logging

import numpy as np
import openvino as ov
import torch

from policy_act import ACTBaseline

LOG = logging.getLogger("ov")


class ACTOpenVINO:
    def __init__(self, xml="ir/act_int8.xml", device="CPU", baseline=None):
        base = baseline or ACTBaseline()
        self.pre, self.post = base.pre, base.post
        self.n_action_steps = base.policy.config.n_action_steps

        core = ov.Core()
        model = core.read_model(xml)
        # LATENCY hint: one request at a time, minimise per-inference wall clock.
        # THROUGHPUT would look better in a bar chart and be wrong for a robot.
        self.compiled = core.compile_model(
            model, device, {"PERFORMANCE_HINT": "LATENCY"}
        )
        self.out = self.compiled.output(0)
        self.name = f"act_ov_{'int8' if 'int8' in xml else 'fp32'}_{device.lower()}"
        self.queue = collections.deque()
        self.did_forward = False

    def reset(self):
        self.queue.clear()
        self.did_forward = False

    def act(self, obs, instruction=None):
        self.did_forward = not self.queue
        if not self.queue:
            batch = {
                "observation.state": torch.from_numpy(
                    np.asarray(obs["agent_pos"], dtype=np.float32)
                ),
                "observation.images.top": torch.from_numpy(
                    np.asarray(obs["pixels"]["top"], dtype=np.float32) / 255.0
                ).permute(2, 0, 1),
            }
            processed = self.pre(batch)
            image = processed["observation.images.top"].cpu().numpy()
            state = processed["observation.state"].cpu().numpy()
            chunk = self.compiled([image, state])[self.out]  # (1, chunk, 14)
            for a in chunk[0][: self.n_action_steps]:
                self.queue.append(a)
        action = self.queue.popleft()
        return self.post(torch.from_numpy(np.asarray(action))).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="ir/act_int8.xml")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    import gymnasium as gym
    import gym_aloha  # noqa: F401

    base = ACTBaseline()
    ovp = ACTOpenVINO(args.xml, args.device, baseline=base)
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)

    base.reset()
    ovp.reset()
    a_torch = base.act(obs)
    a_ov = ovp.act(obs)
    print("torch :", np.round(a_torch[:5], 5))
    print("ov    :", np.round(a_ov[:5], 5))
    print("max|diff| over 14 joints = %.3e" % np.max(np.abs(a_torch - a_ov)))
    env.close()
