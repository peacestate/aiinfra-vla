#!/usr/bin/env bash
# SmolVLA multi-task fine-tune.
#
# BUDGET RULE: never debug on the paid GPU. Run smoke_train.sh on CPU first —
# it proves the config parses, the dataset loads, the loss computes and a
# checkpoint writes. Only then start this.
#
# Sizing: STEPS is set from a measured steps/sec over the first few hundred
# steps, not guessed. save_freq is deliberately frequent so that running out of
# credits still leaves a usable checkpoint rather than nothing.
set -euo pipefail

DATASET="${DATASET:-local/aloha_sim_multitask}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-64}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OUT="${OUT:-outputs/smolvla_multitask}"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="${DATASET}" \
  --batch_size="${BATCH}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=100 \
  --eval_freq=0 \
  --num_workers=8 \
  --output_dir="${OUT}" \
  --job_name=smolvla_aloha_multitask \
  --wandb.enable=false \
  --seed=1000
