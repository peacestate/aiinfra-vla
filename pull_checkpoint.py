"""Download the checkpoint file-by-file, retrying through local TLS failures.

This laptop's TLS interception raises DECRYPTION_FAILED_OR_BAD_RECORD_MAC on
sustained transfers. `download_folder` gives up on the whole folder if any file
fails, so we go file-by-file and retry each one independently — a small file
that already landed is never re-fetched.
"""
import argparse
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(errors="replace")

from lightning_sdk import Studio  # noqa: E402

STUDIO = "vla-bimanual"
TEAMSPACE = "default-project"
USER = "anuragmisgra192000"
SH = "~/vla/outputs/smolvla_multitask/checkpoints"
REL = "vla/outputs/smolvla_multitask/checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", help="checkpoint dir, e.g. 001000 (default: newest)")
    ap.add_argument("--out", default="checkpoint")
    ap.add_argument("--attempts", type=int, default=8)
    a = ap.parse_args()

    s = Studio(name=STUDIO, teamspace=TEAMSPACE, user=USER)
    if str(s.status) != "Running":
        raise SystemExit("studio is not running; start it first")

    step = a.step or s.run(f"ls {SH} | grep -E '^[0-9]+$' | sort -n | tail -1").strip().split()[-1]
    listing = s.run(f"ls -1 {SH}/{step}/pretrained_model")
    files = [f.strip() for f in listing.splitlines() if f.strip() and "." in f]
    print(f"checkpoint {step}: {len(files)} files")

    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)
    failed = []
    for f in files:
        dest = out / f
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {f}: already have it ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        for i in range(a.attempts):
            try:
                s.download_file(f"{REL}/{step}/pretrained_model/{f}", str(dest))
                print(f"  {f}: OK ({dest.stat().st_size/1e6:.1f} MB)")
                break
            except Exception as e:
                msg = str(e)[:70]
                print(f"  {f}: attempt {i+1}/{a.attempts} failed ({msg})")
                if dest.exists():
                    dest.unlink()
                time.sleep(min(2 ** i, 30))
        else:
            failed.append(f)

    if failed:
        print(f"\nFAILED: {failed}")
        print("Local TLS is dropping these. Get a write-scoped HF token and")
        print("push from the studio instead — the studio's network is clean.")
        raise SystemExit(1)
    print("\nall files downloaded to", out)


if __name__ == "__main__":
    main()
