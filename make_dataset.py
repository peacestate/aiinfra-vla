"""Merge the two ALOHA sim datasets into one multi-task dataset.

Why this exists: `aloha_sim_transfer_cube_human` contains exactly ONE task
string across all 50 episodes. A policy fine-tuned on it cannot learn to use
language, because language never varies — it is a constant, and constants carry
no information. The voice interface would then be decoration: say anything, or
nothing, and the arms do the same thing.

Merging in `aloha_sim_insertion_human` gives two instructions over the same
robot, camera and action space, so the instruction becomes a real input. It
also makes the claim testable: feed the wrong instruction and success should
collapse. See bench.py --swap-instruction.

LeRobot's MultiLeRobotDataset is disabled upstream
("The MultiLeRobotDataset isn't supported for now."), so the merge is physical.
"""
import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

SOURCES = [
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_insertion_human",
]
OUTPUT = "local/aloha_sim_multitask"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-repo-id", default=OUTPUT)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    dsets = []
    for repo in SOURCES:
        print(f"loading {repo} ...")
        ds = LeRobotDataset(repo)
        print(f"  episodes={ds.meta.total_episodes} frames={ds.meta.total_frames}")
        print(f"  tasks={list(ds.meta.tasks.index)}")
        dsets.append(ds)

    print(f"\nmerging -> {args.output_repo_id}")
    merged = merge_datasets(dsets, args.output_repo_id, output_dir=args.output_dir)

    tasks = list(merged.meta.tasks.index)
    print(f"\nmerged: episodes={merged.meta.total_episodes} "
          f"frames={merged.meta.total_frames}")
    print(f"tasks ({len(tasks)}):")
    for t in tasks:
        print(f"  - {t}")

    # The whole point of the merge. If this fails, the voice layer is theatre.
    if len(tasks) < 2:
        raise SystemExit(
            f"FAIL: merged dataset has {len(tasks)} task string(s). Language "
            "cannot be learned from a constant — do not train on this."
        )
    exp = sum(d.meta.total_frames for d in dsets)
    if merged.meta.total_frames != exp:
        raise SystemExit(
            f"FAIL: merged frames {merged.meta.total_frames} != {exp} expected"
        )
    print("\nOK: 2 distinct instructions, frame count conserved.")
    print(f"root: {merged.root}")


if __name__ == "__main__":
    main()
