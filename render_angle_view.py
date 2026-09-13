"""Render rollouts from the 'angle' MuJoCo camera instead of 'top'.

The policy is trained on and controlled by the 'top' camera observation only
(unchanged) — this script renders a SEPARATE camera purely for visualization,
since 'top' frames what the policy sees (narrow, single-arm-centric) and
'angle' frames the whole bimanual workspace, which is what a viewer needs to
see the actual handoff between both arms. Additive: does not modify
render_rollout.py or any training/eval path.
"""
import argparse
import pathlib
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np


def rollout_angle_frames(policy, seed, instruction, max_steps=300):
    import gym_aloha  # noqa: F401
    import gymnasium as gym

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=seed)
    policy.reset()
    if hasattr(policy, "rng"):
        policy.rng = np.random.default_rng(seed)
    physics = env.unwrapped._env.physics

    frames, success = [], False
    for _ in range(max_steps):
        action = policy.act(obs, instruction)
        obs, reward, term, trunc, info = env.step(action)
        frames.append(physics.render(height=480, width=640, camera_id="angle"))
        if term or info.get("is_success"):
            success = True
            break
        if trunc:
            break
    env.close()
    return frames, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_langact/pretrained_model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--out-pickle", default="scratch_angle_clips.pkl")
    a = ap.parse_args()

    from policy_langact import LangACT
    print("loading policy...")
    p = LangACT(a.checkpoint).warm()
    instr = "Pick up the cube with the right arm and transfer it to the left arm."

    clips = []
    for s in a.seeds:
        print(f"rollout seed={s} (angle camera) ...")
        frames, ok = rollout_angle_frames(p, s, instr)
        print(f"  seed {s}: success={ok} frames={len(frames)}")
        clips.append((s, frames, ok))

    with open(a.out_pickle, "wb") as fh:
        pickle.dump(clips, fh)
    print("wrote", a.out_pickle, "total frames:", sum(len(f) for _, f, _ in clips))


if __name__ == "__main__":
    main()
