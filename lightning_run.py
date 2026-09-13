"""Launch the SmolVLA fine-tune on Lightning AI.

Budget discipline (8 credits total):
  * `max_runtime` is ALWAYS set. A hung job with no cap is how a budget
    disappears overnight with nothing to show.
  * The probe measures steps/sec before the real run is sized. No guessing.
  * `save_freq` is frequent, so exhausting credits still leaves a usable
    checkpoint instead of nothing.
  * The training config was validated on CPU locally, so the GPU never debugs.

Usage:
    python lightning_run.py probe          # 200 steps, capped, measures rate
    python lightning_run.py train --steps N --batch B
    python lightning_run.py status --name NAME
    python lightning_run.py logs --name NAME
    python lightning_run.py stop --name NAME
"""
import argparse
import pathlib
import warnings

warnings.filterwarnings("ignore")

from lightning_sdk import Job, Machine  # noqa: E402

IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
TEAMSPACE = "anuragmisgra192000/default-project"
# H100 exists only on the Nebius cloud account; the default AWS cluster returns
# "accelerator lit-h100-1 not found". The sgnoir-lora-h100 studio proved this.
CLOUD = "lightning-nebius-prod"

# Kept identical between probe and real run so the measured rate transfers.
SETUP = r"""
set -eu  # container shell is dash, not bash: no pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Fail fast if the GPU is not what we are paying for.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch;assert torch.cuda.is_available();print('GPU:',torch.cuda.get_device_name(0))"

# lerobot depends on evdev, which is source-only and needs a compiler.
# The pytorch *runtime* image has no gcc; installing one is cheaper than
# pulling the multi-GB -devel image on billed GPU time.
apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc python3-dev >/dev/null

pip install -q --upgrade pip
# transformers 5.x breaks lerobot's ACT config:
#   TypeError: non-default argument 'backbone_cfg' follows default argument
pip install -q "lerobot" "transformers<5" num2words accelerate

python - <<'PYEOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
srcs = ["lerobot/aloha_sim_transfer_cube_human", "lerobot/aloha_sim_insertion_human"]
ds = [LeRobotDataset(r) for r in srcs]
m = merge_datasets(ds, "local/aloha_sim_multitask")
tasks = list(m.meta.tasks.index)
print("MERGED episodes=%d frames=%d tasks=%d" % (
    m.meta.total_episodes, m.meta.total_frames, len(tasks)))
for t in tasks:
    print("  task:", t)
# A single instruction teaches the model that language is a constant, which
# would make the whole voice layer decorative. Refuse to train on that.
assert len(tasks) >= 2, "FATAL: merged dataset has <2 instructions"
PYEOF
"""

TRAIN = r"""
time lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --dataset.repo_id=local/aloha_sim_multitask \
  --rename_map='{{"observation.images.top": "observation.images.camera1"}}' \
  --batch_size={batch} \
  --steps={steps} \
  --save_freq={save_freq} \
  --log_freq={log_freq} \
  --eval_freq=0 \
  --num_workers=8 \
  --output_dir=/teamspace/studios/this_studio/outputs/{out} \
  --job_name={out} \
  --wandb.enable=false \
  --seed=1000
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
           --format=csv || true
"""


def build_command(steps, batch, save_freq, log_freq, out):
    return SETUP + TRAIN.format(
        steps=steps, batch=batch, save_freq=save_freq, log_freq=log_freq, out=out
    )


def launch(name, steps, batch, save_freq, log_freq, out, max_runtime):
    cmd = build_command(steps, batch, save_freq, log_freq, out)
    print(f"launching '{name}': {steps} steps @ batch {batch} on H100")
    print(f"hard cap: {max_runtime}s ({max_runtime/3600:.2f} h)")
    job = Job.run(
        name=name,
        machine=Machine.H100,
        image=IMAGE,
        command=cmd,
        teamspace=TEAMSPACE,
        cloud=CLOUD,
        max_runtime=max_runtime,
        interruptible=False,
    )
    print(f"submitted: {job.name}")
    return job


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="200 steps to measure steps/sec")
    p.add_argument("--name", default="smolvla-probe")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    # 45 min covers image pull + pip + dataset merge + 200 steps with margin.
    p.add_argument("--max-runtime", type=int, default=2700)

    t = sub.add_parser("train", help="the real run, sized from the probe")
    t.add_argument("--name", default="smolvla-train")
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--save-freq", type=int, default=1000)
    t.add_argument("--max-runtime", type=int, required=True)

    for n in ("status", "logs", "stop"):
        s = sub.add_parser(n)
        s.add_argument("--name", required=True)

    a = ap.parse_args()

    if a.cmd == "probe":
        launch(a.name, a.steps, a.batch, save_freq=a.steps, log_freq=20,
               out="probe", max_runtime=a.max_runtime)
    elif a.cmd == "train":
        launch(a.name, a.steps, a.batch, save_freq=a.save_freq, log_freq=100,
               out="smolvla_multitask", max_runtime=a.max_runtime)
    else:
        job = Job(name=a.name, teamspace=TEAMSPACE)
        if a.cmd == "status":
            print("status:", job.status)
            print("machine:", getattr(job, "machine", "?"))
        elif a.cmd == "logs":
            # job.logs is a _Logs object, and Windows cp1252 cannot encode the
            # box-drawing characters pip emits. Write UTF-8 to a file instead.
            L = job.logs
            txt = L if isinstance(L, str) else "\n".join(str(x) for x in L)
            out = pathlib.Path(f"{a.name}.log")
            out.write_text(txt, encoding="utf-8")
            print(f"wrote {out} ({txt.count(chr(10))+1} lines)")
        elif a.cmd == "stop":
            job.stop()
            print("stopped", a.name)


if __name__ == "__main__":
    main()
