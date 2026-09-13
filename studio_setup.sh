#!/usr/bin/env bash
# One-shot Studio setup. RUN THIS ON CPU, never on H100.
#
# Every line here exists because it failed on billed GPU time once already.
# Prints SETUP_OK on success; anything else means do not switch to H100 yet.
#
#   nohup bash studio_setup.sh > setup.log 2>&1 &
#   tail -f setup.log        # wait for SETUP_OK

set -eu   # container shell is dash: no `pipefail`

echo "=== 1/5 toolchain ==="
# lerobot depends on evdev, which is source-only and needs a compiler.
apt-get update -qq
apt-get install -y -qq --no-install-recommends gcc python3-dev >/dev/null
echo "gcc: $(gcc --version | head -1)"

echo "=== 2/5 python deps ==="
pip install -q --upgrade pip
# transformers 5.x breaks lerobot's ACT config:
#   TypeError: non-default argument 'backbone_cfg' follows default argument
pip install -q "lerobot" "transformers<5" num2words accelerate

echo "=== 3/5 ABI repairs ==="
# The base image ships scipy 1.11 (numpy 1.x ABI) and scikit-learn 1.3, while
# lerobot pulls numpy 2.x. Both then fail at import with confusing errors:
#   "numpy.core.multiarray failed to import"
#   "numpy.dtype size changed, may indicate binary incompatibility"
pip install -q -U "scipy>=1.14" scikit-learn
python -c "import numpy,scipy,sklearn,torch,lerobot; print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__)"
python -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "=== 4/5 dataset (2 instructions, or the voice layer is decoration) ==="
python - <<'PY'
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
# A single instruction teaches the model that language is a constant.
assert len(tasks) >= 2, "FATAL: merged dataset has <2 instructions"
assert m.meta.total_frames == 45000, f"unexpected frame count {m.meta.total_frames}"
PY

echo "=== 5/5 pre-fetch weights (so GPU time is never spent downloading) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
print("cached:", snapshot_download("lerobot/smolvla_base"))
PY

echo
echo "SETUP_OK"
