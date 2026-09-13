"""Download a large file through a TLS middlebox that strangles long streams.

Observed on this machine: short HTTPS requests succeed, but sustained transfers
die with "[SSL] record layer failure" / "DECRYPTION_FAILED_OR_BAD_RECORD_MAC" —
across lightning.ai, huggingface.co and speechmatics.com alike. Local HTTPS
inspection, not the servers. So: never hold a connection open long. Fetch small
byte ranges, each its own short request.

Two correctness rules learned the hard way:

  1. WRITE AT EXPLICIT OFFSETS, never append. An earlier append-based version
     let two concurrent resumes both append, producing a 1010 MB file where the
     real one is 906.7 MB — silently corrupt, and it still "looked complete".
  2. TAKE A LOCK. Resuming while another copy is still running is the situation
     that caused (1).

Verifies the final size and refuses to report success on a mismatch.
"""
import argparse
import os
import pathlib
import sys
import time

import requests


def load_env(path=".env"):
    env = {}
    p = pathlib.Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def fetch_range(url, headers, start, end, attempts=10):
    h = dict(headers)
    h["Range"] = f"bytes={start}-{end}"
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=h, timeout=60)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            data = r.content
            want = end - start + 1
            if len(data) != want:
                raise RuntimeError(f"short read {len(data)} != {want}")
            return data
        except Exception as e:
            last = e
            time.sleep(min(1.5 ** i, 10))
    raise RuntimeError(f"range {start}-{end} failed after {attempts}: {last}")


def probe_size(url, headers):
    """Total size, via HEAD if allowed and otherwise from Content-Range.

    Kaggle's signed kaggleusercontent URLs answer HEAD without a Content-Length,
    so asking for one byte and reading `Content-Range: bytes 0-0/<total>` is the
    reliable way to size the object — and it doubles as proof that the server
    honours ranges at all, which this whole downloader depends on.
    """
    try:
        r = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        if r.status_code < 400 and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    h = dict(headers)
    h["Range"] = "bytes=0-0"
    r = requests.get(url, headers=h, allow_redirects=True, timeout=60)
    if r.status_code != 206 or "Content-Range" not in r.headers:
        raise SystemExit(
            f"server will not serve byte ranges (HTTP {r.status_code}); this "
            "downloader cannot work against it"
        )
    return int(r.headers["Content-Range"].split("/")[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="earthpulse/smolvla-aloha-multitask")
    ap.add_argument("--file", default="model.safetensors")
    ap.add_argument("--url", default=None,
                    help="fetch this exact URL instead of building an HF one "
                         "(e.g. a pre-signed Kaggle kernel-output link)")
    ap.add_argument("--out", default="checkpoint_smolvla8k/model.safetensors")
    ap.add_argument("--chunk-mb", type=float, default=2.0)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(out.suffix + ".lock")
    progress = out.with_suffix(out.suffix + ".progress")

    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit(
            f"another download holds {lock}. Kill it first, or delete the lock.\n"
            "Two concurrent downloads previously corrupted this file."
        )

    try:
        if a.url:
            # A pre-signed URL carries its own credentials in the query string;
            # attaching a bearer token as well gets the request rejected.
            url = a.url
            headers = {"User-Agent": "aiinfra-vla/0.1"}
        else:
            env = load_env()
            token = env.get("HF_WRITE_TOKEN", "").strip()
            headers = {"Authorization": f"Bearer {token}",
                       "User-Agent": "aiinfra-vla/0.1"}
            url = f"https://huggingface.co/{a.repo}/resolve/main/{a.file}"

        size = probe_size(url, headers)

        # Progress is tracked in a sidecar file, NOT inferred from file length,
        # so a partially written file can never be mistaken for a complete one.
        done = 0
        if out.exists() and progress.exists():
            try:
                done = int(progress.read_text().strip())
            except ValueError:
                done = 0
        if done >= size:
            print(f"already complete: {size/1e6:.1f} MB")
            return

        # Pre-allocate so every chunk can be written at its true offset.
        mode = "r+b" if out.exists() and out.stat().st_size == size else "wb"
        if mode == "wb":
            with open(out, "wb") as fh:
                fh.truncate(size)
            done = 0
        print(f"downloading {size/1e6:.1f} MB from offset {done/1e6:.1f} MB")

        chunk = int(a.chunk_mb * 1e6)
        t0 = time.time()
        with open(out, "r+b") as fh:
            while done < size:
                end = min(done + chunk - 1, size - 1)
                data = fetch_range(url, headers, done, end)
                fh.seek(done)
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
                done = end + 1
                progress.write_text(str(done))
                el = time.time() - t0
                print(f"\r  {done/1e6:7.1f}/{size/1e6:.1f} MB ({100.0*done/size:5.1f}%) "
                      f"{done/1e6/max(el, 1e-9):.2f} MB/s",
                      end="", file=sys.stderr, flush=True)

        actual = out.stat().st_size
        if actual != size:
            raise SystemExit(f"\nFAIL: wrote {actual} bytes, expected {size}")
        print(f"\ndone: {out} ({actual/1e6:.1f} MB, size verified)")
        progress.unlink(missing_ok=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
