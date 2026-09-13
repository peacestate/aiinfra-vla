"""Runtime-agnostic benchmark: latency AND closed-loop success, same seeds.

A policy is anything with `.reset()` and `.act(obs, instruction) -> (14,) array`.
Swapping PyTorch for OpenVINO IR must not change a single line below, so the
two runtimes are scored on identical episodes.
"""
import argparse, json, os, statistics, time
from pathlib import Path

import gymnasium as gym
import numpy as np
import gym_aloha  # noqa: F401  (registers gym_aloha/* envs)

TASK = "gym_aloha/AlohaTransferCube-v0"
# The second task the multi-task policy was trained on. ACT's public checkpoint
# cannot attempt this at all — it was trained on transfer-cube only and has no
# instruction input to switch with. One policy covering both is the actual
# argument for a VLA over a task-specific policy.
TASK_INSERT = "gym_aloha/AlohaInsertion-v0"

# Verbatim from the training data — a paraphrase is a different test.
INSTRUCTION = "Pick up the cube with the right arm and transfer it to the left arm."
# The other task the policy was trained on. Feeding this while running the
# transfer-cube environment is the control condition for language grounding:
# if success does not drop, the policy is ignoring the instruction and any
# voice interface built on it is decoration.
WRONG_INSTRUCTION = "Insert the peg into the socket."

# Which LangACT checkpoint to score. Set per run so the success-vs-steps curve
# (50k / 100k / 150k) is a matter of pointing at a different directory, not of
# editing code between measurements.
LANGACT_CKPT = os.environ.get("LANGACT_CKPT", "checkpoint_langact/pretrained_model")


def rollout(policy, seed, max_steps=400, instruction=INSTRUCTION, task=None):
    """One episode. Returns (success, per-step inference latencies in ms)."""
    env = gym.make(task or TASK, obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=seed)
    policy.reset()
    latencies, success = [], False
    try:
        for _ in range(max_steps):
            t0 = time.perf_counter()
            action = policy.act(obs, instruction)
            ms = (time.perf_counter() - t0) * 1000.0
            # `did_forward` distinguishes a real network evaluation from a cached
            # replay. Policies that never chunk report every call as a forward.
            latencies.append((ms, getattr(policy, "did_forward", True)))
            obs, reward, terminated, truncated, info = env.step(action)
            # gym-aloha reports task completion via reward==4 / terminated
            if terminated or info.get("is_success"):
                success = True
                break
            if truncated:
                break
    finally:
        env.close()
    return success, latencies


def _stats(values):
    if not values:
        return None
    v = sorted(values)
    return {
        "n": len(v),
        "p50": statistics.median(v),
        "p95": v[min(int(0.95 * len(v)), len(v) - 1)],
        "mean": statistics.fmean(v),
    }


def evaluate(policy, seeds, label, instruction=INSTRUCTION, task=None):
    successes, all_lat, per_seed = 0, [], {}
    for s in seeds:
        ok, lat = rollout(policy, s, instruction=instruction, task=task)
        successes += ok
        # Which seeds failed, not just how many. A rate cannot answer "did INT8
        # fail the same episodes as fp32" or "where in the task does it break",
        # and both questions need the per-seed record kept at measurement time.
        per_seed[s] = bool(ok)
        all_lat.extend(lat)
        fwd = [ms for ms, is_fwd in lat if is_fwd]
        print(f"  seed {s:>3}: {'SUCCESS' if ok else 'fail   '}  "
              f"({len(lat)} steps, {len(fwd)} forward passes, "
              f"{statistics.median(fwd) if fwd else float('nan'):.1f} ms/forward)")

    forward = [ms for ms, is_fwd in all_lat if is_fwd]
    replay = [ms for ms, is_fwd in all_lat if not is_fwd]
    result = {
        "config": label,
        "seeds": list(seeds),
        "episodes": len(seeds),
        "successes": successes,
        "success_rate": successes / len(seeds),
        "per_seed": per_seed,
        # The number that OpenVINO/INT8 actually changes.
        "forward_ms": _stats(forward),
        # What the control loop feels, chunking included. Never compare this
        # across policies with different chunk sizes.
        "replay_ms": _stats(replay),
        "amortized_ms": _stats([ms for ms, _ in all_lat]),
        "steps": len(all_lat),
    }
    f = result["forward_ms"]
    print(f"\n{label}: success {successes}/{len(seeds)} ({result['success_rate']:.0%})")
    print(f"  forward pass : p50 {f['p50']:.1f} ms  p95 {f['p95']:.1f} ms  (n={f['n']})")
    if replay:
        r = result["replay_ms"]
        print(f"  cached replay: p50 {r['p50']:.1f} ms  (n={r['n']})")
    print(f"  amortized    : p50 {result['amortized_ms']['p50']:.1f} ms/step")
    return result


class RandomPolicy:
    """Control condition. Its success rate is the number a real policy must beat;
    if a 'working' policy scores this, it is not working."""

    def __init__(self):
        self.space = gym.make(TASK).action_space

    def reset(self):
        pass

    def act(self, obs, instruction):
        return self.space.sample()


def _act_baseline():
    from policy_act import ACTBaseline
    return ACTBaseline()


def _langact():
    from policy_langact import LangACT
    # warm() pre-computes the instruction embeddings, so the latencies measured
    # here are the policy's, not the sentence encoder's.
    return LangACT(LANGACT_CKPT).warm()


def _langact_ov(xml, device):
    def make():
        from policy_langact_ov import LangACTOpenVINO
        return LangACTOpenVINO(xml, device, checkpoint=LANGACT_CKPT)
    return make


def _ov(xml, device):
    def make():
        from policy_ov import ACTOpenVINO
        return ACTOpenVINO(xml, device)
    return make


def _smolvla():
    from policy_smolvla import SmolVLA
    return SmolVLA()


def _smolvla_ov(xml, device, n_action_steps=None):
    def make():
        from policy_smolvla_ov import SmolVLAOpenVINO
        return SmolVLAOpenVINO(xml, device, n_action_steps=n_action_steps)
    return make


POLICIES = {
    "random": RandomPolicy,
    "smolvla": _smolvla,
    "smolvla_ov_int8_gpu": _smolvla_ov("ir/smolvla_int8.xml", "GPU"),
    "smolvla_ov_int8_cpu": _smolvla_ov("ir/smolvla_int8.xml", "CPU"),
    # Replan frequency sweep — no retraining, pure inference-time knob.
    "smolvla_n25": _smolvla_ov("ir/smolvla_int8.xml", "GPU", 25),
    "smolvla_n10": _smolvla_ov("ir/smolvla_int8.xml", "GPU", 10),
    "smolvla_n5": _smolvla_ov("ir/smolvla_int8.xml", "GPU", 5),
    "act": _act_baseline,
    # Same architecture and latency class as ACT, but language-conditioned.
    "langact": _langact,
    "langact_ov_fp32_cpu": _langact_ov("ir/langact_fp32.xml", "CPU"),
    "langact_ov_int8_cpu": _langact_ov("ir/langact_int8.xml", "CPU"),
    "langact_ov_int8_gpu": _langact_ov("ir/langact_int8.xml", "GPU"),
    "ov_fp32_cpu": _ov("ir/act_fp32.xml", "CPU"),
    "ov_int8_cpu": _ov("ir/act_int8.xml", "CPU"),
    "ov_int8_gpu": _ov("ir/act_int8.xml", "GPU"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="random", choices=sorted(POLICIES))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--task", choices=["transfer", "insert"], default="transfer",
                    help="insert = the second trained task; ACT cannot do it")
    ap.add_argument("--swap-instruction", action="store_true",
                    help="feed the peg-insertion instruction while running the "
                         "transfer-cube env; success should collapse if the "
                         "policy actually reads language")
    args = ap.parse_args()

    seeds = range(args.seed0, args.seed0 + args.episodes)
    if args.task == "insert":
        task, instruction = TASK_INSERT, WRONG_INSTRUCTION
        if args.swap_instruction:
            instruction = INSTRUCTION  # swap = the transfer sentence, wrong here
    else:
        task = TASK
        instruction = WRONG_INSTRUCTION if args.swap_instruction else INSTRUCTION
    label = f"{args.policy}_{args.task}" + ("_wrongtext" if args.swap_instruction else "")
    print(f"instruction: {instruction!r}")
    result = evaluate(POLICIES[args.policy](), seeds, label, instruction, task)
    result["instruction"] = instruction
    result["task"] = task

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"{label}.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out / f'{label}.json'}")


if __name__ == "__main__":
    main()
