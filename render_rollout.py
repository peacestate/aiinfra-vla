"""Record rollouts to MP4, with the instruction burned into the frame.

The instruction-swap result (30% -> 0%) is the strongest claim in this project
and until now it existed only as numbers in a terminal. This renders it: the
same seed, the same weights, the same noise, run twice with different sentences,
side by side. A viewer can see the words change the behaviour.

The instruction is drawn ON the video deliberately — a demo that shows arms
moving while a voiceover claims language did it proves nothing.
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402


def label(frame, lines, colour=(255, 255, 255)):
    """Draw text onto an RGB frame using PIL (no OpenCV dependency)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()

    pad = 8
    h = 26 * len(lines) + pad
    d.rectangle([0, 0, img.width, h], fill=(0, 0, 0))
    y = 4
    for i, line in enumerate(lines):
        d.text((pad, y), line, fill=colour if i == 0 else (200, 200, 200),
               font=font if i == 0 else small)
        y += 26
    return np.asarray(img)


def rollout_frames(policy, seed, instruction, max_steps=300, tag="", raw=False):
    import gym_aloha  # noqa: F401
    import gymnasium as gym

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=seed)
    policy.reset()
    if hasattr(policy, "rng"):
        policy.rng = np.random.default_rng(seed)  # identical noise across runs

    frames, success = [], False
    for _ in range(max_steps):
        action = policy.act(obs, instruction)
        obs, reward, term, trunc, info = env.step(action)
        frames.append(np.asarray(obs["pixels"]["top"], dtype=np.uint8))
        if term or info.get("is_success"):
            success = True
            break
        if trunc:
            break
    env.close()

    if raw:
        # Caller wants to draw its own banner; labelling here would double it.
        return frames, success
    verdict = "SUCCESS" if success else "FAILED"
    colour = (120, 255, 120) if success else (255, 120, 120)
    out = [label(f, [f'{tag}"{instruction}"', f"seed {seed}  -  {verdict}"], colour)
           for f in frames]
    return out, success


def side_by_side(a, b):
    """Pad the shorter clip with its last frame so both end together."""
    n = max(len(a), len(b))
    a = a + [a[-1]] * (n - len(a))
    b = b + [b[-1]] * (n - len(b))
    return [np.concatenate([x, y], axis=1) for x, y in zip(a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5, help="a seed that SUCCEEDS with the correct instruction")
    ap.add_argument("--out", default="media/instruction_swap.mp4")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=25)
    a = ap.parse_args()

    import imageio.v2 as imageio
    import pathlib

    from bench import INSTRUCTION, WRONG_INSTRUCTION
    from policy_smolvla_ov import SmolVLAOpenVINO

    print("loading policy ...", flush=True)
    p = SmolVLAOpenVINO("ir/smolvla_int8.xml", "GPU", n_action_steps=a.n_action_steps)

    print(f"rollout with CORRECT instruction (seed {a.seed}) ...", flush=True)
    right, ok_right = rollout_frames(p, a.seed, INSTRUCTION, tag="CORRECT: ")

    print(f"rollout with WRONG instruction (seed {a.seed}) ...", flush=True)
    wrong, ok_wrong = rollout_frames(p, a.seed, WRONG_INSTRUCTION, tag="WRONG:   ")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frames = side_by_side(right, wrong)
    imageio.mimsave(a.out, frames, fps=a.fps, macro_block_size=1)

    print(f"\nwrote {a.out}  ({len(frames)} frames, {len(frames)/a.fps:.1f}s)")
    print(f"  correct instruction: {'SUCCESS' if ok_right else 'FAILED'}")
    print(f"  wrong   instruction: {'SUCCESS' if ok_wrong else 'FAILED'}")
    if ok_right and not ok_wrong:
        print("  -> this is the shot: same seed, same noise, only the sentence differs")
    else:
        print("  -> not the clean contrast; try another seed that succeeds")


if __name__ == "__main__":
    main()
