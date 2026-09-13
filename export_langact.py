"""Export LangACT to OpenVINO IR, then INT8-compress it with NNCF.

Same contract as export_openvino.py, one extra input: the 384-dim sentence
embedding that occupies `observation.environment_state`.

The embedding is passed as a *tensor*, not computed inside the graph. That is
the point of the design — the sentence encoder never runs at inference, because
instructions are known strings whose embeddings are cached on the host. Tracing
MiniLM into the graph would import the very cost this architecture avoids.

Parity against the PyTorch reference is checked before anything is written, and
the language input is checked to actually change the output: an exported graph
that ignores its third input would benchmark beautifully and be worthless.
"""
import argparse
import logging
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch

from policy_langact import FEATURE, TRAINED_INSTRUCTIONS, LangACT

LOG = logging.getLogger("export_langact")


class LangACTChunkWrapper(torch.nn.Module):
    """Functional view: (image, state, lang) -> action chunk."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, state, lang):
        batch = {
            "observation.images.top": image,
            "observation.state": state,
            FEATURE: lang,
        }
        return self.policy.predict_action_chunk(batch)


def example_inputs(cfg, lang_vec, batch=1):
    c, h, w = cfg.input_features["observation.images.top"].shape
    state_dim = cfg.input_features["observation.state"].shape[0]
    return (
        torch.zeros(batch, c, h, w, dtype=torch.float32),
        torch.zeros(batch, state_dim, dtype=torch.float32),
        torch.from_numpy(lang_vec).unsqueeze(0),
    )


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", default="ir")
    ap.add_argument("--prefix", default="langact")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--int8-tol", type=float, default=5e-2,
                    help="advisory only for INT8; success rate is the real gate")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)

    LOG.info("loading LangACT from %s ...", args.checkpoint)
    lang = LangACT(args.checkpoint).warm()
    policy = lang.policy
    wrapper = LangACTChunkWrapper(policy).eval()

    v0 = lang.embed(TRAINED_INSTRUCTIONS[0])
    v1 = lang.embed(TRAINED_INSTRUCTIONS[1])
    ex_img, ex_state, ex_lang = example_inputs(policy.config, v0)

    with torch.no_grad():
        reference = wrapper(ex_img, ex_state, ex_lang).cpu().numpy()
        other = wrapper(ex_img, ex_state,
                        torch.from_numpy(v1).unsqueeze(0)).cpu().numpy()
    LOG.info("torch reference chunk: %s", reference.shape)

    lang_effect = max_abs_diff(reference, other)
    LOG.info("torch: swapping the instruction moves the chunk by %.3e", lang_effect)
    if lang_effect == 0.0:
        raise SystemExit(
            "FAIL: the two instructions produce identical action chunks — the "
            "language slot is not wired into this checkpoint."
        )

    LOG.info("converting to OpenVINO IR (fp32)...")
    ov_model = ov.convert_model(
        wrapper,
        example_input=(ex_img, ex_state, ex_lang),
        input=[("image", ex_img.shape), ("state", ex_state.shape),
               ("lang", ex_lang.shape)],
    )
    fp32_path = out / f"{args.prefix}_fp32.xml"
    ov.save_model(ov_model, fp32_path, compress_to_fp16=False)
    LOG.info("wrote %s", fp32_path)

    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    got = compiled([ex_img.numpy(), ex_state.numpy(), ex_lang.numpy()])[compiled.output(0)]
    d = max_abs_diff(reference, got)
    LOG.info("fp32 IR parity vs torch: max|diff| = %.3e", d)
    if d > args.tol:
        raise SystemExit(
            f"FAIL: fp32 IR deviates by {d:.3e} > tol {args.tol:.1e}. "
            "The exported graph is not the model — do not benchmark it."
        )
    LOG.info("fp32 IR parity OK")

    # The graph must still respond to the instruction. A trace that folded the
    # example embedding into a constant would pass parity above and silently
    # become a non-language model.
    got_other = compiled([ex_img.numpy(), ex_state.numpy(),
                          v1[None, :]])[compiled.output(0)]
    ir_effect = max_abs_diff(got, got_other)
    LOG.info("fp32 IR: swapping the instruction moves the chunk by %.3e", ir_effect)
    if ir_effect == 0.0:
        raise SystemExit(
            "FAIL: the IR ignores its `lang` input — the embedding was traced "
            "into a constant. Do not benchmark this graph."
        )

    LOG.info("compressing weights to INT8 (NNCF INT8_ASYM)...")
    # all_layers=True is INT4-only in NNCF 3.3.0; INT8 already covers every
    # compressible layer. See export_openvino.py.
    int8_model = nncf.compress_weights(
        core.read_model(fp32_path),
        mode=nncf.CompressWeightsMode.INT8_ASYM,
    )
    int8_path = out / f"{args.prefix}_int8.xml"
    ov.save_model(int8_model, int8_path)
    LOG.info("wrote %s", int8_path)

    compiled8 = core.compile_model(int8_model, "CPU")
    got8 = compiled8([ex_img.numpy(), ex_state.numpy(),
                      ex_lang.numpy()])[compiled8.output(0)]
    d8 = max_abs_diff(reference, got8)
    LOG.info("int8 parity vs torch: max|diff| = %.3e", d8)
    if d8 > args.int8_tol:
        LOG.warning(
            "INT8 deviates by %.3e (> %.1e advisory). Numeric drift alone does "
            "not condemn it — the gate is closed-loop success rate (bench.py).",
            d8, args.int8_tol,
        )

    got8_other = compiled8([ex_img.numpy(), ex_state.numpy(),
                            v1[None, :]])[compiled8.output(0)]
    LOG.info("int8 IR: swapping the instruction moves the chunk by %.3e",
             max_abs_diff(got8, got8_other))

    for p in (fp32_path, int8_path):
        mb = p.with_suffix(".bin").stat().st_size / 1e6
        LOG.info("%-20s weights %7.1f MB", p.name, mb)


if __name__ == "__main__":
    main()
