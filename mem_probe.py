"""Find out why the SmolVLA rollout dies silently.

No traceback, no exit message, process gone = almost certainly the OS killing
it for memory (this laptop has 8 GB). Measure RSS at each stage rather than
guessing, and report the peak against what is actually free.
"""
import logging
import os

import psutil
import torch

logging.disable(logging.WARNING)
proc = psutil.Process(os.getpid())


def rss(tag):
    mb = proc.memory_info().rss / 1e6
    avail = psutil.virtual_memory().available / 1e6
    print(f"{tag:32s} RSS {mb:7.0f} MB | system available {avail:7.0f} MB", flush=True)
    return mb


rss("baseline")

import gymnasium as gym  # noqa: E402
import gym_aloha  # noqa: F401,E402
import numpy as np  # noqa: E402

rss("after imports")

from policy_smolvla import SmolVLA  # noqa: E402

p = SmolVLA()
rss("after policy load")

print("dtype:", next(p.policy.parameters()).dtype)
n = sum(q.numel() for q in p.policy.parameters())
print(f"params: {n/1e6:.0f}M -> {n*4/1e9:.2f} GB at fp32")

env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
obs, _ = env.reset(seed=0)
rss("after env")

from bench import INSTRUCTION  # noqa: E402

p.reset()
for i in range(6):
    a = p.act(obs, INSTRUCTION)
    obs, r, te, tr, info = env.step(a)
    rss(f"after step {i+1} (fwd={p.did_forward})")

env.close()
print("SURVIVED 6 STEPS")
