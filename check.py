"""One command to answer: is it still training, how far, and when will it finish?

    python check.py            # progress
    python check.py --stop     # stop training and shut the GPU down now
    python check.py --pull     # download the newest checkpoint locally

Windows note: stdout is forced to replace unencodable characters, because the
training progress bar emits block glyphs that cp1252 cannot encode and that
raises UnicodeEncodeError mid-print.
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(errors="replace")

from lightning_sdk import Studio  # noqa: E402

STUDIO = "vla-bimanual"
TEAMSPACE = "default-project"
USER = "anuragmisgra192000"
REMOTE = "~/vla/outputs/smolvla_multitask/checkpoints"           # for s.run (shell)
REMOTE_REL = "vla/outputs/smolvla_multitask/checkpoints"          # for download_folder


def is_running(s):
    return str(s.status) == "Running"


def studio():
    return Studio(name=STUDIO, teamspace=TEAMSPACE, user=USER)


def safe(text):
    return text.encode("ascii", "replace").decode("ascii")


def progress(s):
    if not is_running(s):
        # The watcher stops the studio when training exits, so Stopped is the
        # normal "finished" signal, not an error.
        print(f"studio: {s.status}  -> training has ended (watcher stopped it)")
        print("run `python check.py --pull` to fetch the final checkpoint")
        return
    out = s.run(
        "cd ~/vla && "
        "grep -aoE 'step:[0-9K]+ .*loss:[0-9.]+' train.log | tail -3; "
        "echo '---'; "
        "grep -aoE '[0-9]+/20000 \\[[^]]*\\]' train.log | tail -1; "
        "echo '---'; "
        "ls " + REMOTE + " 2>/dev/null | tail -3; "
        "echo '---'; "
        "pgrep -f lerobot-train >/dev/null && echo TRAINING || echo FINISHED; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
    )
    print(f"studio: {s.status} on {s.machine}")
    print(safe(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true",
                    help="kill training and stop the GPU immediately")
    ap.add_argument("--pull", action="store_true",
                    help="download the newest checkpoint to ./checkpoint")
    a = ap.parse_args()
    s = studio()

    if a.stop:
        if is_running(s):
            print(safe(s.run("pkill -f lerobot-train || true; sleep 20; echo killed")))
            s.stop()
        print("studio stopped — no further GPU charges")
        return

    if a.pull:
        if not is_running(s):
            print("studio is stopped; starting on CPU to copy files (cheap)")
            from lightning_sdk import Machine
            s.start(machine=Machine.CPU)
        last = s.run(f"ls {REMOTE} | sort -n | tail -1").strip().splitlines()[-1]
        print("newest checkpoint:", last)
        s.download_folder(f"{REMOTE_REL}/{last}/pretrained_model", "checkpoint")
        got = [f for f in pathlib.Path("checkpoint").rglob("*") if f.is_file()]
        if not any(f.suffix == ".safetensors" for f in got):
            raise SystemExit("FAIL: no weights downloaded (%d files)" % len(got))
        mb = sum(f.stat().st_size for f in got) / 1e6
        print("downloaded %d files, %.0f MB -> ./checkpoint" % (len(got), mb))
        return

    progress(s)


if __name__ == "__main__":
    main()
