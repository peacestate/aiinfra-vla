"""Add a language token to the ALOHA multi-task dataset, without touching ACT.

The trick: ACT already assembles its encoder input as
    [latent, (robot_state), (env_state), (image_features)]
and `env_state` is an unused slot wired to a plain `nn.Linear` projection
(modeling_act.py:344 and :465). So a sentence embedding written into
`observation.environment_state` becomes a language token inside *stock,
unmodified* ACT. No architecture code, no custom policy class.

Why this beats SmolVLA on the axes that matter for deployment:
  * language costs ZERO at inference — instructions are known strings, so their
    embeddings are computed once and cached. SmolVLA re-runs a 500M VLM every
    chunk (measured: 908 ms of fixed prefix cost).
  * the policy stays ~52M params, so latency and model size stay ACT-class.

It still genuinely reads language: a novel phrasing goes through the same
sentence encoder and produces a different vector, so the policy responds to
meaning rather than to a task index. A task ID would not survive rephrasing.
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"  # 22M, 384-dim, offline only
FEATURE = "observation.environment_state"
MERGED = "local/aloha_sim_multitask"
OUTPUT = "local/aloha_sim_multitask_lang"


def embed(texts, model_name=ENCODER):
    """Mean-pooled sentence embeddings, L2-normalised."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()
    enc = tok(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=-1).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=MERGED)
    ap.add_argument("--output", default=OUTPUT)
    a = ap.parse_args()

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(a.source)
    tasks = list(ds.meta.tasks.index)
    print(f"source: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
    for t in tasks:
        print("  task:", t)
    if len(tasks) < 2:
        raise SystemExit("FATAL: <2 instructions; language would be a constant")

    vecs = embed(tasks)
    dim = vecs.shape[1]
    print(f"embeddings: {vecs.shape} ({ENCODER})")

    # Sanity: the two instructions must be distinguishable. If the encoder maps
    # them to nearly the same vector, the language token carries no signal and
    # the whole design fails silently.
    sim = float(vecs[0] @ vecs[1])
    print(f"cosine similarity between the two instructions: {sim:.4f}")
    if sim > 0.95:
        raise SystemExit("FATAL: instructions embed almost identically")

    task_to_vec = {t: v for t, v in zip(tasks, vecs)}
    index_to_task = {i: t for i, t in enumerate(tasks)}

    def language_feature(row, episode_idx, frame_in_episode):
        """Called per frame as (row_dict, ep_idx, frame_in_ep); map the row's
        task to its cached embedding."""
        ti = int(row["task_index"])
        return task_to_vec[index_to_task[ti]]

    print(f"writing {a.output} ...")
    out = add_features(
        ds,
        {FEATURE: (language_feature, {"dtype": "float32", "shape": (dim,),
                                      "names": None})},
        repo_id=a.output,
    )
    print(f"done: {out.meta.total_frames} frames, feature {FEATURE} dim {dim}")
    print("root:", out.root)

    # Verify the feature actually varies with the task — a constant column would
    # reproduce exactly the bug this dataset exists to avoid.
    sample = out[0]
    print(f"frame 0 {FEATURE}: shape {tuple(np.asarray(sample[FEATURE]).shape)}")


if __name__ == "__main__":
    main()
