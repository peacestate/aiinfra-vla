"""Measure SmolVLA inference latency across runtimes on identical input.

PyTorch on this i3 was measured at ~14 s per inference over 878 real rollout
steps. This is the number the Intel stack is supposed to move.

LATENCY hint, not THROUGHPUT: a robot runs one inference at a time, and
THROUGHPUT would flatter the numbers while describing a workload nobody has.
"""
import argparse
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402
import torch  # noqa: E402


def measure(fn, warmup, iters, label):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(int(0.95 * len(times)), len(times) - 1)]
    print(f"{label:28s} p50 {p50:9.1f} ms   p95 {p95:9.1f} ms   (n={iters})",
          flush=True)
    return {"label": label, "p50": p50, "p95": p95, "n": iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_smolvla")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--skip-torch", action="store_true",
                    help="PyTorch takes ~14 s per call; skip when only comparing IRs")
    a = ap.parse_args()

    from export_smolvla import build_example
    from policy_smolvla import SmolVLA
    from bench import INSTRUCTION

    print("loading policy ...", flush=True)
    p = SmolVLA(a.checkpoint)
    ex = build_example(p, INSTRUCTION)
    images, img_masks, lang_tokens, lang_masks, state, noise = ex
    flat = list(images) + list(img_masks) + [lang_tokens, lang_masks, state, noise]
    flat = [t.float() if t.dtype.is_floating_point else t for t in flat]
    arrays = [t.numpy() for t in flat]

    results = []

    if not a.skip_torch:
        from export_smolvla import SampleActions
        w = SampleActions(p.policy.model, len(images)).eval().float()
        with torch.no_grad():
            results.append(measure(lambda: w(*flat), 0, max(1, a.iters // 2),
                                   "PyTorch fp32 CPU"))

    core = ov.Core()
    print("OpenVINO devices:", core.available_devices, flush=True)

    for tag, xml in (("fp32", "ir/smolvla_fp32.xml"), ("int8", "ir/smolvla_int8.xml")):
        for dev in ("CPU", "GPU"):
            if dev not in core.available_devices:
                continue
            try:
                cm = core.compile_model(core.read_model(xml), dev,
                                        {"PERFORMANCE_HINT": "LATENCY"})
                out = cm.output(0)
                results.append(measure(lambda: cm(arrays)[out], a.warmup, a.iters,
                                       f"OpenVINO {tag} {dev}"))
            except Exception as e:
                print(f"OpenVINO {tag} {dev}: FAILED ({type(e).__name__}: "
                      f"{str(e)[:90]})", flush=True)

    if results:
        base = results[0]["p50"]
        print("\nspeedup vs", results[0]["label"])
        for r in results[1:]:
            print(f"  {r['label']:26s} {base / r['p50']:6.1f}x")


if __name__ == "__main__":
    main()
