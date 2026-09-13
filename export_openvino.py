"""Export a LeRobot policy to OpenVINO IR, then INT8-compress it with NNCF.

Exports the *chunk predictor* — the pure tensor-in/tensor-out network — not the
stateful `select_action` wrapper. The action queue is Python control flow and
belongs on the host; tracing it would bake one queue state into the graph.

Parity is checked against the PyTorch reference before anything is written, so
a silently-wrong graph cannot reach the benchmark.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import nncf
import openvino as ov
import torch

from policy_act import ACTBaseline

LOG = logging.getLogger("export")


class ACTChunkWrapper(torch.nn.Module):
    """Functional view of ACT: (image, state) -> action chunk.

    LeRobot policies consume a dict; OpenVINO wants positional tensors. This
    adapter is the whole bridge.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, state):
        batch = {"observation.images.top": image, "observation.state": state}
        return self.policy.predict_action_chunk(batch)


def example_inputs(cfg, batch=1):
    c, h, w = cfg.input_features["observation.images.top"].shape
    state_dim = cfg.input_features["observation.state"].shape[0]
    return (
        torch.zeros(batch, c, h, w, dtype=torch.float32),
        torch.zeros(batch, state_dim, dtype=torch.float32),
    )


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="ir")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max abs deviation from the PyTorch reference for fp32 IR")
    ap.add_argument("--int8-tol", type=float, default=5e-2,
                    help="advisory only for INT8; success rate is the real gate")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)

    LOG.info("loading ACT (with verified normalization stats)...")
    baseline = ACTBaseline()
    policy = baseline.policy
    wrapper = ACTChunkWrapper(policy).eval()

    ex_img, ex_state = example_inputs(policy.config)

    with torch.no_grad():
        reference = wrapper(ex_img, ex_state).cpu().numpy()
    LOG.info("torch reference chunk: %s", reference.shape)

    LOG.info("converting to OpenVINO IR (fp32)...")
    ov_model = ov.convert_model(
        wrapper,
        example_input=(ex_img, ex_state),
        input=[("image", ex_img.shape), ("state", ex_state.shape)],
    )
    fp32_path = out / "act_fp32.xml"
    ov.save_model(ov_model, fp32_path, compress_to_fp16=False)
    LOG.info("wrote %s", fp32_path)

    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    got = compiled([ex_img.numpy(), ex_state.numpy()])[compiled.output(0)]
    d = max_abs_diff(reference, got)
    LOG.info("fp32 IR parity vs torch: max|diff| = %.3e", d)
    if d > args.tol:
        raise SystemExit(
            f"FAIL: fp32 IR deviates by {d:.3e} > tol {args.tol:.1e}. "
            "The exported graph is not the model — do not benchmark it."
        )
    LOG.info("fp32 IR parity OK")

    LOG.info("compressing weights to INT8 (NNCF INT8_ASYM)...")
    # NOTE: Intel's Pi0.5 walkthrough passes `all_layers=True`, but NNCF 3.3.0
    # rejects it for INT8 modes ("INT8 modes do not support all_layers") — it is
    # an INT4-only knob now. INT8 already covers all compressible layers.
    int8_model = nncf.compress_weights(
        core.read_model(fp32_path),
        mode=nncf.CompressWeightsMode.INT8_ASYM,
    )
    int8_path = out / "act_int8.xml"
    ov.save_model(int8_model, int8_path)
    LOG.info("wrote %s", int8_path)

    compiled8 = core.compile_model(int8_model, "CPU")
    got8 = compiled8([ex_img.numpy(), ex_state.numpy()])[compiled8.output(0)]
    d8 = max_abs_diff(reference, got8)
    LOG.info("int8 parity vs torch: max|diff| = %.3e", d8)
    if d8 > args.int8_tol:
        LOG.warning(
            "INT8 deviates by %.3e (> %.1e advisory). Numeric drift alone does not "
            "condemn it — the gate is closed-loop success rate, so run bench.py.",
            d8, args.int8_tol,
        )

    for p in (fp32_path, int8_path):
        mb = (p.with_suffix(".bin").stat().st_size) / 1e6
        LOG.info("%-16s weights %7.1f MB", p.name, mb)


if __name__ == "__main__":
    main()
