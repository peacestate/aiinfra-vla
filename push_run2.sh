#!/usr/bin/env bash
# Push a LangACT continuation run to Kaggle.
#
#   export KAGGLE_USER=youruser
#   bash push_run2.sh                                          # run 2, resumes run 1
#   bash push_run2.sh langact-train-3 youruser/langact-train-2  # run 3, resumes run 2
#
# Each run is a NEW slug that mounts the PREVIOUS run's output via kernel_sources:
# a kernel cannot list its own output as a source. Resuming this way keeps the
# ~630 MB checkpoint inside Kaggle, off a laptop whose TLS drops long transfers.
#
# Quota does NOT trigger anything. It refreshes weekly (00:00 UTC) but the kernel
# still has to be pushed by hand. Check what is left first:
#   curl -s -H "Authorization: Bearer $(tr -d ' \r\n' < ~/.kaggle/access_token)" \
#        https://www.kaggle.com/api/v1/kernels/quota
set -eu

USER="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
SLUG="${1:-langact-train-2}"
SOURCE="${2:-$USER/langact-train}"
HERE="$(cd "$(dirname "$0")" && pwd)"

DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT
cp "$HERE/kaggle_langact.py" "$DIR/"

# machine_shape is what pins the T4. Without it Kaggle hands out a P100, which is
# sm_60 and unsupported by current torch — the kernel aborts in ~6 seconds.
cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "$USER/$SLUG",
  "title": "$SLUG",
  "code_file": "kaggle_langact.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$USER/langact-aloha-multitask-lang"],
  "competition_sources": [],
  "kernel_sources": ["$SOURCE"],
  "model_sources": []
}
JSON

echo "pushing $USER/$SLUG (resuming from $SOURCE)"
kaggle kernels push -p "$DIR" --accelerator NvidiaTeslaT4

cat <<NOTE

Now confirm it actually starts. An out-of-quota account does NOT error: it leaves
the kernel QUEUED forever, or reports a successful push having created nothing.
A healthy kernel reaches RUNNING within ~2 minutes:

  kaggle kernels status $USER/<slug>

The log is only readable AFTER the run completes, so watch progress in the browser.
Fetch the log afterwards from the API (kernels output downloads every file and the
big checkpoint starves the log):

  curl -s -H "Authorization: Bearer \$(tr -d ' \r\n' < ~/.kaggle/access_token)" \\
    "https://www.kaggle.com/api/v1/kernels/output?userName=$USER&kernelSlug=<slug>"
NOTE
