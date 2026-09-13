# GPU budget protocol

Learned by burning ~7 of 8 credits to get ~30 minutes of training.
H100 on Lightning bills at **~5.1 credits/hour** (measured, two readings).

## The rule

**The GPU trains. It does not install, debug, download, queue, or think.**

Everything else happens on CPU (cheap) or locally (free), and is *proven working*
before the machine is switched to H100.

## Pre-flight, in order — none of this costs GPU time

1. **Validate the training config locally on CPU**, 2 steps, batch 1. This alone
   caught four blockers that would each have cost H100 minutes:
   - `--policy.repo_id` required unless `--policy.push_to_hub=false`
   - Windows mangles `lerobot/smolvla_base` into a backslash path -> local snapshot
   - SmolVLA expects 3 cameras, ALOHA has 1 -> `--rename_map` + `--policy.empty_cameras=2`
   - checkpoint `last` symlink needs admin on Windows (Linux-only concern)
2. **Start the Studio on CPU**, run `studio_setup.sh`, wait for `SETUP_OK`.
3. **Pre-download model weights and datasets on CPU** — they persist across the
   machine switch on the Studio's disk.
4. **Only then** `switch_machine(H100)` and immediately start training.

## Use a Studio, not a Job

A Job re-pays image pull + apt + pip + dataset merge on *billed GPU time*, every
run. A Studio keeps its filesystem across a machine switch, so setup is paid once
at CPU rates. Switching an existing Studio to H100 also got capacity instantly
when a fresh Job sat queued for 22 minutes.

## Always arm an autostop

Never leave training unattended without a step budget. An H100 that finishes at
3am and idles costs more than the training did — and a run that exhausts credits
mid-training strands the checkpoint on a machine that can no longer be started.

`autostop.py` waits for a target checkpoint, kills training, pushes to HF **while
the machine is still alive**, then stops the Studio. Push before shutdown, always.

## Checkpoint often, and get one out early

`--save_freq=1000`. Pull or push the first checkpoint as soon as it exists. If
credits run out, the latest checkpoint is still a usable model instead of nothing.

## Environment fixes (Lightning base image)

All of these are in `studio_setup.sh`:

| symptom | cause | fix |
| --- | --- | --- |
| `set: Illegal option -o pipefail` | container shell is **dash** | `set -eu` |
| `error: command 'gcc' failed` | no compiler; `evdev` is source-only | `apt-get install -y gcc python3-dev` |
| `numpy.core.multiarray failed to import` | scipy 1.11 vs numpy 2.2 ABI | `pip install -U "scipy>=1.14"` |
| `numpy.dtype size changed` (sklearn) | scikit-learn 1.3 built for numpy 1.x | `pip install -U scikit-learn` |
| `libavutil.so.56: cannot open` | torchcodec wants ffmpeg 4 | `--dataset.video_backend=pyav` |
| `Venv creation is not allowed` | Studios allow 1 conda env | new Studio, not a venv |
| `accelerator lit-h100-1 not found` | H100 only on **lightning-nebius-prod** | pass that cloud account |

## Gotchas that cost time

- `Studio.status` is an **enum**; `str(s.status) == "Running"` — comparing the
  enum to a string is always False and silently reports "finished".
- `download_folder` does not expand `~`, and reports success having downloaded
  nothing. Use a path relative to the Studio root.
- Studios auto-sleep. Start and run in the same call.
