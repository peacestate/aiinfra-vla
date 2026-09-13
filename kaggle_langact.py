"""LangACT training kernel for Kaggle (T4).

Runs stock, unmodified LeRobot ACT on `local/aloha_sim_multitask_lang`, where a
384-dim MiniLM sentence embedding occupies `observation.environment_state` — the
unused slot ACT already projects with a plain nn.Linear. See make_lang_dataset.py.

Why Kaggle and not an H100: at ~265 GFLOPs/step this model never saturates a T4,
let alone an H100. The bottleneck is MP4 decode on the CPU, so the faster card
buys wall clock proportional to nothing. See README.

Three rules this kernel exists to enforce, all learned the hard way:

  1. The GPU trains. Run this once on CPU first (it detects the absence of a GPU
     and does a 2-step smoke instead) so that installs and dataset staging are
     never billed against GPU quota.
  2. Stop on our terms. Kaggle kills a kernel that exhausts quota, and a killed
     kernel may never persist /kaggle/working. We self-terminate at a time budget
     comfortably inside the quota remaining.
  3. Push before shutdown, always. Checkpoints go to the Hub as they appear, over
     Kaggle's link rather than a laptop's. A run that dies between pushes loses
     at most SAVE_FREQ steps.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

T0 = time.time()

# --- knobs ------------------------------------------------------------------
STEPS = int(os.environ.get("LANGACT_STEPS", 150_000))
BATCH = int(os.environ.get("LANGACT_BATCH", 8))
WORKERS = int(os.environ.get("LANGACT_WORKERS", 4))  # Kaggle gives ~4 vCPU
SAVE_FREQ = int(os.environ.get("LANGACT_SAVE_FREQ", 5000))
# Budget well inside the quota actually remaining, so we stop rather than be killed.
# 10h, not the 11.7h a 15k->150k run needs: Kaggle kills a GPU session at
# 12h, and being killed loses the drain-to-checkpoint stop entirely.
BUDGET_S = int(os.environ.get("LANGACT_BUDGET_S", 36000))  # 10h
# On hitting the budget, keep training until the NEXT checkpoint lands rather
# than terminating on the spot. Run 1 stopped at step 19,735 with its last save
# at 15,000 and threw away 4,735 steps — about 25 minutes of GPU.
DRAIN_GRACE_S = int(os.environ.get("LANGACT_DRAIN_GRACE_S", 1800))
KEEP_CKPTS = 2  # /kaggle/working is capped; each ACT checkpoint is ~630 MB
# Milestones are never pruned, so success-vs-steps can be evaluated after the
# fact. LangACT splits every batch across 2 tasks and so sees half the per-task
# data of the cube-only ACT baseline; 200k is where its cube exposure matches.
MILESTONES = {50_000, 100_000, 150_000}  # 50k is a free extra point on the curve

LEROBOT = "lerobot==0.4.4"
TRANSFORMERS = "transformers==4.57.6"  # 5.x breaks LeRobot's ACT config

DS_IN = pathlib.Path("/kaggle/input/langact-aloha-multitask-lang")
DS_ID = "local/aloha_sim_multitask_lang"
DS_HOME = pathlib.Path.home() / ".cache/huggingface/lerobot" / DS_ID
OUT = pathlib.Path("/kaggle/working/langact")
HF_REPO = os.environ.get("LANGACT_HF_REPO", "earthpulse/langact-aloha-multitask")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


# --- 0. fail in the first second, not the tenth hour ------------------------
def validate_knobs():
    """Reject knob combinations that would silently produce no usable curve.

    LeRobot only writes a checkpoint at multiples of SAVE_FREQ. A milestone that
    is not such a multiple is never created, so the prune exemption protects
    nothing and the gap is only discovered once the run has finished and the GPU
    hours are spent. Ten hours of training to learn that 50,000 % 3,000 != 0 is
    the expensive way to find out.
    """
    bad = sorted(m for m in MILESTONES if m % SAVE_FREQ)
    if bad:
        raise SystemExit(
            f"FATAL: milestones {bad} are not multiples of SAVE_FREQ={SAVE_FREQ}, "
            f"so no checkpoint is ever written at those steps and prune() would "
            f"protect files that do not exist. Pick milestones on the save grid."
        )
    unreachable = sorted(m for m in MILESTONES if m > STEPS)
    if unreachable:
        log(f"WARNING: milestones {unreachable} exceed STEPS={STEPS:,} and will "
            f"never be reached in this run")
    if KEEP_CKPTS < 1:
        raise SystemExit(f"FATAL: KEEP_CKPTS={KEEP_CKPTS} would delete every checkpoint")

    # ~630 MB per ACT checkpoint (207 MB weights + optimizer state) against the
    # 20 GB /kaggle/working cap. Report it so a future knob change that would
    # fill the disk is visible up front rather than as a death mid-save.
    retained = len([m for m in MILESTONES if m <= STEPS]) + KEEP_CKPTS
    log(f"knobs OK: STEPS={STEPS:,} SAVE_FREQ={SAVE_FREQ:,} "
        f"milestones={sorted(MILESTONES)} -> <={retained} checkpoints "
        f"(~{retained * 0.63:.1f} GB of the 20 GB cap)")


# --- 1. what machine did we get? -------------------------------------------
def describe_machine():
    try:
        import torch
    except ImportError:
        log("torch not importable yet")
        return False, None
    gpu = torch.cuda.is_available()
    log(f"torch {torch.__version__}  cuda={gpu}")
    if gpu:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        log(f"gpu: {name}  sm_{cap[0]}{cap[1]}  count={torch.cuda.device_count()}")
        # P100 (sm_60) no longer works with current torch builds; T4 is sm_75.
        if cap[0] < 7:
            raise SystemExit(f"FATAL: {name} is sm_{cap[0]}{cap[1]}; select T4, not P100")
    return gpu, torch.__version__


# --- 2. install, pinning torch so pip cannot swap it out on billed time ------
def install(torch_version):
    con = pathlib.Path("/kaggle/working/constraints.txt")
    lines = []
    if torch_version:
        lines.append(f"torch=={torch_version.split('+')[0]}")
    try:
        import torchvision

        lines.append(f"torchvision=={torchvision.__version__.split('+')[0]}")
    except ImportError:
        pass
    con.write_text("\n".join(lines) + "\n")
    log(f"constraints: {lines}")
    # If LeRobot genuinely cannot live with the image's torch, pip fails here in
    # seconds instead of quietly pulling ~2 GB of wheels at GPU rates.
    run([sys.executable, "-m", "pip", "install", "-q", "-c", str(con),
         LEROBOT, TRANSFORMERS, "huggingface_hub"])


# --- 3. stage the dataset where LeRobot expects it --------------------------
def stage_dataset():
    if DS_HOME.exists() and (DS_HOME / "meta/info.json").exists():
        log(f"dataset already staged at {DS_HOME}")
        return
    src = DS_IN
    if not (src / "meta/info.json").exists():
        # Don't guess the mount name: find the dataset root by its marker file.
        root = pathlib.Path("/kaggle/input")
        log(f"{DS_IN} is not the root; /kaggle/input contains: "
            f"{sorted(p.name for p in root.iterdir()) if root.exists() else 'NOTHING'}")
        hits = sorted(root.glob("*/**/meta/info.json")) if root.exists() else []
        if not hits:
            raise SystemExit("FATAL: no LeRobot dataset (meta/info.json) under /kaggle/input")
        src = hits[0].parent.parent
        log(f"discovered dataset root: {src}")
    DS_HOME.parent.mkdir(parents=True, exist_ok=True)
    log(f"staging {src} -> {DS_HOME}")
    # /kaggle/input is read-only and LeRobot writes into the dataset root.
    shutil.copytree(src, DS_HOME, dirs_exist_ok=True)
    info = DS_HOME / "meta/info.json"
    if not info.exists():
        found = sorted(p.name for p in DS_HOME.iterdir())
        raise SystemExit(f"FATAL: no meta/info.json after staging; got {found}")
    import json

    meta = json.loads(info.read_text())
    log(f"dataset: {meta.get('total_episodes')} episodes, "
        f"{meta.get('total_frames')} frames")


# --- 4. token: required for GPU runs, optional for the CPU smoke ------------
def hf_token(required):
    tok = os.environ.get("HF_WRITE_TOKEN", "").strip()
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient

            # An *unset* Kaggle secret comes back as an empty string, not an error.
            tok = (UserSecretsClient().get_secret("HF_WRITE_TOKEN") or "").strip()
        except Exception as e:
            log(f"kaggle_secrets unavailable: {e}")
    if not tok:
        # Mounts are nested under /kaggle/input/datasets/<user>/<slug>, so glob.
        for p in sorted(pathlib.Path("/kaggle/input").glob("**/hf_token.txt")):
            tok = p.read_text().strip()
            break
    if not tok:
        msg = "no HF token (secret 'HF_WRITE_TOKEN' or an hf_token.txt dataset)"
        if required:
            raise SystemExit(f"FATAL: {msg}; checkpoints would be stranded")
        # Not fatal: the checkpoint still survives as this kernel's own output,
        # which the next run mounts via kernel_sources. HF is the second copy.
        log(f"note: {msg} — relying on Kaggle kernel output alone")
        return None
    log(f"HF token found (len={len(tok)})")
    return tok


# --- 5. resume from whatever the last session pushed ------------------------
def find_mounted_checkpoint():
    """Newest checkpoint among any attached kernel outputs.

    Attaching the previous run's kernel output via `kernel_sources` keeps the
    whole resume loop inside Kaggle: no credential, and no 630 MB round trip
    through a laptop whose TLS drops sustained transfers.
    """
    import json

    root = pathlib.Path("/kaggle/input")
    if not root.exists():
        return None, 0
    best, best_step = None, -1
    for h in sorted(root.glob("**/checkpoints/*/training_state/training_step.json")):
        try:
            step = json.loads(h.read_text()).get("step", 0)
        except Exception:
            continue
        if step > best_step:
            best, best_step = h.parent.parent, step
    if best is None:
        log("no mounted checkpoint; starting from scratch")
        return None, 0
    log(f"found mounted checkpoint at step {best_step}: {best}")
    return best, best_step


def resume_from(local, step):
    """Copy a checkpoint into OUT and return its train_config.json path."""
    log(f"resuming at step {step}")
    dest = OUT / "checkpoints" / f"{step:06d}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local, dest, dirs_exist_ok=True)
    last = OUT / "checkpoints/last"
    # `last` may be a symlink (Linux) or a real directory (the Windows fallback
    # below). unlink() raises on a directory, and leaving it in place makes
    # LeRobot's own save fail with FileExistsError mid-run.
    if last.is_symlink() or last.is_file():
        last.unlink()
    elif last.is_dir():
        shutil.rmtree(last, ignore_errors=True)
    try:
        last.symlink_to(dest, target_is_directory=True)
    except OSError:
        shutil.copytree(dest, last, dirs_exist_ok=True)
    return dest / "pretrained_model/train_config.json"


# --- 6. training ------------------------------------------------------------
def train_cmd(gpu, resume_config):
    exe = shutil.which("lerobot-train") or "lerobot-train"
    if resume_config:
        # The resumed train_config carries the ORIGINAL --steps (100000). Without
        # this override a resume silently stops at the old target instead of the
        # one we asked for.
        return [exe, f"--config_path={resume_config}", "--resume=true",
                f"--steps={STEPS}"]
    return [
        exe,
        f"--dataset.repo_id={DS_ID}",
        "--dataset.video_backend=pyav",  # torchcodec wants an ffmpeg 4 the image lacks
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--policy.device={'cuda' if gpu else 'cpu'}",
        f"--output_dir={OUT}",
        "--job_name=langact",
        f"--steps={STEPS if gpu else 2}",
        f"--batch_size={BATCH if gpu else 1}",
        f"--num_workers={WORKERS if gpu else 0}",
        f"--save_freq={SAVE_FREQ if gpu else 2}",
        "--save_checkpoint=true",
        "--log_freq=100",
        "--eval_freq=0",  # never spend GPU minutes on rollouts
        "--wandb.enable=false",
        "--seed=1000",
    ]


def read_lines(stream, chunk=2048):
    """Yield lines split on BOTH \\r and \\n.

    tqdm redraws its bar with carriage returns and no newline, so iterating the
    pipe line-by-line blocks for the entire run — the monitor would never tick,
    and neither checkpoint pushes nor the time budget would ever fire. Splitting
    on \\r keeps the loop alive on every bar redraw.
    """
    buf = ""
    while True:
        data = stream.read(chunk)
        if not data:
            break
        buf += data
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        for p in parts:
            if p.strip():
                yield p
    if buf.strip():
        yield buf


STEP_RE = re.compile(r"step:\s*([\d.]+)([KM]?)")
TQDM_RE = re.compile(r"\b(\d+)/(\d+)\s*\[")  # "Training:  12%|..| 1200/100000 ["


def parse_step(line):
    m = STEP_RE.search(line)
    if m:
        n = float(m.group(1))
        return int(n * {"": 1, "K": 1e3, "M": 1e6}[m.group(2)])
    # LeRobot abbreviates ("step:12K"), so the tqdm counter is the exact source
    # and, unlike the log line, it appears on every redraw rather than every
    # log_freq steps.
    m = TQDM_RE.search(line)
    if m and int(m.group(2)) == STEPS:
        return int(m.group(1))
    return None


def push(token, ckpt):
    """True only if the checkpoint really reached the Hub."""
    if not token:
        return False
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        # A fine-grained token scoped to this one repo may lack repo-creation
        # rights entirely. That is fine when the repo already exists, so a 403
        # here must not abort the upload that follows.
        try:
            api.create_repo(HF_REPO, private=True, exist_ok=True)
        except Exception as e:
            log(f"create_repo skipped ({type(e).__name__}); assuming it exists")
        api.upload_folder(folder_path=str(ckpt), repo_id=HF_REPO,
                          commit_message=f"checkpoint {ckpt.name}")
        log(f"pushed {ckpt.name} -> {HF_REPO}")
        return True
    except Exception as e:
        log(f"WARNING: push of {ckpt.name} failed: {e}")
        return False


def prune(pushed, token):
    """Keep the newest KEEP_CKPTS. /kaggle/working is capped at 20 GB and each
    ACT checkpoint is ~630 MB, so an unpruned long run fills the disk and the
    kernel dies with the checkpoint it was trying to save."""
    dirs = sorted(d for d in (OUT / "checkpoints").glob("[0-9]*") if d.is_dir())
    for d in dirs[:-KEEP_CKPTS]:
        # Milestones outlive the rolling window: they are the success-vs-steps
        # curve, and once pruned they cannot be recovered without retraining.
        if int(d.name) in MILESTONES:
            continue
        # With a token, only drop what is safely on the Hub. Without one, the
        # kernel output is the only copy, so keeping the newest is all we can do.
        if token and d.name not in pushed:
            continue
        shutil.rmtree(d, ignore_errors=True)
        log(f"pruned {d.name}")


def main():
    validate_knobs()  # before anything is installed or any GPU time is spent
    gpu, tv = describe_machine()
    install(tv)
    gpu, tv = describe_machine()  # re-read: install may have moved torch
    stage_dataset()
    token = hf_token(required=False)

    if not gpu:
        log("NO GPU — running the 2-step preflight smoke instead")

    # Do NOT create OUT here: LeRobot refuses to start when the output directory
    # already exists and resume is False. Only the resume path may create it.
    resume_config = None
    if gpu:
        ckpt, step = find_mounted_checkpoint()
        if ckpt is not None:
            try:
                resume_config = resume_from(ckpt, step)
            except Exception as e:
                # Never let a broken resume burn the whole GPU session. Falling
                # back to scratch costs steps; crashing costs the entire slot.
                log(f"WARNING: resume failed ({type(e).__name__}: {e}) — scratch")
                shutil.rmtree(OUT, ignore_errors=True)
                resume_config = None

    cmd = train_cmd(gpu, resume_config)
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace")

    pushed, seen, first_step, first_t, last_step = set(), set(), None, None, 0
    stopping = False
    logged_at = -1       # last step bucket already reported, to stop double-counting
    drain_from = None    # how many checkpoints existed when the budget expired
    for line in read_lines(proc.stdout):
        print(line.rstrip(), flush=True)
        step = parse_step(line)
        if step is not None:
            last_step = step
            # Skip warmup: the first steps pay for cudnn autotune and a cold
            # page cache, and would make the extrapolation pessimistic.
            if first_step is None and step >= 200:
                first_step, first_t = step, time.time()
            elif first_step is not None and step > first_step:
                # Report once per 1000-step bucket. tqdm redraws many times at the
                # SAME step, so logging on every redraw re-reports a growing wall
                # clock against an unchanged step count — run 1 sawtoothed from
                # 303 to 322 ms/step and back purely from this.
                bucket = step // 1000
                if step % 1000 == 0 and bucket != logged_at:
                    logged_at = bucket
                    ms = (time.time() - first_t) / (step - first_step) * 1000
                    eta = (STEPS - step) * ms / 3.6e6
                    log(f"MEASURED {ms:.1f} ms/step -> {STEPS} steps in "
                        f"{STEPS * ms / 3.6e6:.2f} h; {eta:.2f} h remaining")

        for d in sorted(p for p in (OUT / "checkpoints").glob("[0-9]*") if p.is_dir()):
            # `seen` stops a failing push being retried on every bar redraw;
            # `pushed` records only what actually landed, so prune never deletes
            # a checkpoint on the strength of a push that did not happen.
            if d.name not in seen and (d / "training_state/training_step.json").exists():
                seen.add(d.name)
                if push(token, d):
                    pushed.add(d.name)
                prune(pushed, token)

        if not stopping and time.time() - T0 > BUDGET_S:
            if drain_from is None:
                drain_from = len(seen)
                log(f"TIME BUDGET {BUDGET_S}s reached at step {last_step} — "
                    f"draining to the next checkpoint (grace {DRAIN_GRACE_S}s)")
            # Terminating here would discard every step since the last save: run 1
            # lost 4,735 of them. Wait for one more checkpoint, but never past the
            # grace, or we would trade a saved run for an out-of-quota kill.
            if len(seen) > drain_from or time.time() - T0 > BUDGET_S + DRAIN_GRACE_S:
                log(f"stopping at step {last_step} on our terms "
                    f"(checkpoints={sorted(seen)})")
                stopping = True
                proc.terminate()

    proc.wait()
    log(f"training exited rc={proc.returncode} at step {last_step}")

    # Whatever the exit reason, get the newest checkpoint out before we die.
    for d in sorted(p for p in (OUT / "checkpoints").glob("[0-9]*") if p.is_dir()):
        if d.name not in pushed and push(token, d):
            pushed.add(d.name)

    log(f"DONE steps={last_step} pushed={sorted(pushed)} "
        f"elapsed={(time.time() - T0) / 60:.1f} min")

    # A non-zero rc is expected only when *we* sent the terminate at the budget.
    # Any other non-zero rc is a failed run, and must not be reported as success:
    # the kernel exiting 0 with a green tick is exactly how a dead run hides.
    if proc.returncode != 0 and not stopping:
        raise SystemExit(f"FATAL: training failed with rc={proc.returncode} "
                         f"at step {last_step}")
    # The run is only useful if a checkpoint survives for the next session to
    # mount. Without a token that copy is the kernel output, so check the disk.
    on_disk = sorted(p.name for p in (OUT / "checkpoints").glob("[0-9]*") if p.is_dir())
    if gpu and not on_disk:
        raise SystemExit("FATAL: GPU run produced no checkpoint on disk")
    log(f"checkpoints retained in kernel output: {on_disk}")
    print("LANGACT_OK")


if __name__ == "__main__":
    main()
