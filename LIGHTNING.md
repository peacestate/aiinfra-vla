# Lightning AI H100 run — SmolVLA multi-task fine-tune

**Budget: 8 credits.** Everything below was validated on CPU locally first, so
the GPU session is throughput only. Do not debug here.

## 0. Before starting the machine

Have these ready so the GPU is never idle while you think:

- HF token (for pulling `lerobot/smolvla_base` and the two source datasets)
- This repo's `make_dataset.py` and `train_smolvla.sh`

## 1. Environment (CPU-time, start the GPU *after* this if billing allows)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "lerobot" "transformers<5" num2words accelerate
# transformers 5.x breaks lerobot's ACT config:
#   TypeError: non-default argument 'backbone_cfg' follows default argument
```

## 2. Rebuild the merged dataset (do NOT upload 45k frames)

It is faster to re-merge from the Hub than to transfer the local copy.

```bash
python make_dataset.py
# expect: episodes=100 frames=45000, 2 task strings, "OK"
```

The script hard-fails if fewer than 2 instructions survive — a single-instruction
dataset teaches SmolVLA to ignore language, which would make the voice layer
meaningless.

## 3. Measure throughput BEFORE committing the budget

Run 200 steps, read steps/sec, then compute the affordable step count.

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false --policy.empty_cameras=2 \
  --dataset.repo_id=local/aloha_sim_multitask \
  --rename_map='{"observation.images.top": "observation.images.camera1"}' \
  --batch_size=64 --steps=200 --save_freq=200 --log_freq=20 --eval_freq=0 \
  --num_workers=8 --output_dir=outputs/probe --job_name=probe \
  --wandb.enable=false
```

Then: `affordable_steps = (credits_remaining * seconds_per_credit) * steps_per_sec`,
and take **70%** of it. Leave headroom — a run that dies at 95% with no
checkpoint is worth nothing.

## 4. The real run

`save_freq` is deliberately frequent: if credits run out, the latest checkpoint
is still a usable model.

```bash
STEPS=<from step 3> BATCH=64 SAVE_FREQ=1000 bash train_smolvla.sh
```

## 5. Bring the checkpoint home

Only `pretrained_model/` is needed (~900 MB); skip `training_state/` unless you
intend to resume.

```bash
huggingface-cli upload <your-user>/smolvla-aloha-multitask \
  outputs/smolvla_multitask/checkpoints/<last>/pretrained_model
```

## Windows notes (local only — none apply on Lightning)

- `--policy.path=lerobot/smolvla_base` is mangled to `lerobot\smolvla_base`;
  use a local `snapshot_download` path instead.
- Checkpoint `last` symlink needs Developer Mode; training itself completes.
- `torchcodec` is unavailable, so video decoding falls back to `pyav`.
