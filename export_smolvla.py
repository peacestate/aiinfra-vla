"""Export SmolVLA's flow-matching sampler to OpenVINO IR, then INT8 it.

Why this is the headline: SmolVLA in PyTorch on this i3 takes ~14 s per
inference (measured, 878 steps). A robot cannot close a loop on that. The whole
submission is what Intel's stack does to that number.

What is exported: `model.sample_actions(images, img_masks, lang_tokens,
lang_masks, state, noise)` — pure tensors in, action chunk out. Tokenisation
stays on the host because it is not a tensor op, and the action queue stays on
the host because it is Python control flow.

`noise` is passed explicitly rather than sampled inside, for two reasons:
deterministic parity checking against the PyTorch reference, and keeping RNG
out of the traced graph.
"""
import argparse
import logging
import pathlib

import numpy as np
import nncf
import openvino as ov
import torch

from policy_smolvla import SmolVLA

LOG = logging.getLogger("export-smolvla")


class SampleActions(torch.nn.Module):
    """Functional view of the flow-matching sampler.

    prepare_images() returns LISTS of per-camera tensors (here 3: the real
    ALOHA top camera plus the 2 padded empties SmolVLA expects). A traced graph
    needs flat positional tensors, so the lists are unpacked as arguments and
    rebuilt inside forward().
    """

    def __init__(self, model, n_cams):
        super().__init__()
        self.model = model
        self.n_cams = n_cams

    def forward(self, *args):
        n = self.n_cams
        images = list(args[:n])
        img_masks = list(args[n:2 * n])
        lang_tokens, lang_masks, state, noise = args[2 * n:]
        return self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )


def build_example(p, instruction):
    """Run one real observation through the policy's own preprocessing so the
    exported graph sees exactly the tensors inference will feed it."""
    import gymnasium as gym
    import gym_aloha  # noqa: F401
    from policy_smolvla import CAMERA_KEY

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    batch = {
        "observation.state": torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)),
        CAMERA_KEY: torch.from_numpy(
            np.asarray(obs["pixels"]["top"], dtype=np.float32) / 255.0
        ).permute(2, 0, 1),
        "task": instruction,
    }
    env.close()

    processed = p.pre(batch)
    policy = p.policy
    prepared = policy._prepare_batch(processed)
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    images, img_masks = policy.prepare_images(prepared)
    state = policy.prepare_state(prepared)
    lang_tokens = prepared[OBS_LANGUAGE_TOKENS]
    lang_masks = prepared[OBS_LANGUAGE_ATTENTION_MASK]

    cfg = policy.config
    noise = torch.zeros(
        (state.shape[0], cfg.chunk_size, cfg.max_action_dim), dtype=torch.float32
    )
    return images, img_masks, lang_tokens, lang_masks, state, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_smolvla8k")
    ap.add_argument("--outdir", default="ir")
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--num-steps", type=int, default=None,
                    help="flow-matching denoise steps (config default 10). "
                         "Latency = prefix + num_steps x expert, so this is "
                         "the main latency lever; it is baked into the graph.")
    ap.add_argument("--suffix", default="", help="tag for the IR filenames")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from bench import INSTRUCTION

    LOG.info("loading %s ...", a.checkpoint)
    p = SmolVLA(a.checkpoint)
    if a.num_steps:
        LOG.info("num_steps %d -> %d", p.policy.config.num_steps, a.num_steps)
        p.policy.config.num_steps = a.num_steps
    ex = build_example(p, INSTRUCTION)
    images, img_masks, lang_tokens, lang_masks, state, noise = ex
    n_cams = len(images)
    LOG.info("cameras: %d (1 real + %d padded)", n_cams, n_cams - 1)

    flat = list(images) + list(img_masks) + [lang_tokens, lang_masks, state, noise]
    for i, t in enumerate(flat):
        LOG.info("  arg%-2d %-16s %s", i, tuple(t.shape), t.dtype)

    wrapper = SampleActions(p.policy.model, n_cams).eval().float()

    # bfloat16 does not convert cleanly; OpenVINO wants fp32 on the way in.
    ex = tuple(t.float() if t.dtype.is_floating_point else t for t in flat)

    with torch.no_grad():
        reference = wrapper(*ex).cpu().numpy()
    LOG.info("torch reference chunk: %s", reference.shape)

    out = pathlib.Path(a.outdir)
    out.mkdir(exist_ok=True)

    LOG.info("converting to OpenVINO IR ...")
    ov_model = ov.convert_model(wrapper, example_input=ex)
    fp32 = out / f"smolvla_fp32{a.suffix}.xml"
    ov.save_model(ov_model, fp32, compress_to_fp16=False)
    LOG.info("wrote %s", fp32)

    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    got = compiled([t.numpy() for t in ex])[compiled.output(0)]
    d = float(np.max(np.abs(reference - got)))
    LOG.info("fp32 IR parity vs torch: max|diff| = %.3e", d)
    if d > a.tol:
        raise SystemExit(f"FAIL: fp32 IR deviates by {d:.3e} — do not benchmark it")

    LOG.info("compressing to INT8 ...")
    int8 = nncf.compress_weights(core.read_model(fp32), mode=nncf.CompressWeightsMode.INT8_ASYM)
    int8_path = out / f"smolvla_int8{a.suffix}.xml"
    ov.save_model(int8, int8_path)

    c8 = core.compile_model(int8, "CPU")
    got8 = c8([t.numpy() for t in ex])[c8.output(0)]
    LOG.info("int8 parity vs torch: max|diff| = %.3e",
             float(np.max(np.abs(reference - got8))))

    for f in (fp32, int8_path):
        LOG.info("%-20s %7.1f MB", f.name, f.with_suffix(".bin").stat().st_size / 1e6)


if __name__ == "__main__":
    main()
