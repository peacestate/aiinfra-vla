# Bimanual VLA on Intel — AI Infra Summit Hackathon (online track)

**Track:** Intel — Bimanual VLA Manipulation (online)
**Bonus:** Speechmatics — Best Use of Speechmatics (stacks with the track)
**Build window:** 2026-09-10 → submission ~2026-09-17 01:00 UTC

## The claim

A language-conditioned bimanual manipulation policy, running in real time on a
**$400 Intel laptop with no discrete GPU** — 12th Gen Core i3-1215U, 8 GB RAM,
Intel UHD iGPU — driven by voice.

SmolVLA (450M) is reported at ~2,028 ms per inference on CPU. At that rate a
robot cannot close its control loop. We export it to OpenVINO IR, compress the
weights to INT8 with NNCF, and measure what that buys on real Intel silicon.

**Hardware disclosure.** This is a 12th Gen Core i3 (Alder Lake) — CPU and
Intel UHD iGPU only, **no NPU**. Newer Core Ultra chips (Meteor Lake/Lunar
Lake) add an NPU as a third inference target; none of the numbers in this
repo touch one, because this machine doesn't have one. Every CPU and iGPU
number here is real, measured, and reproducible with `benchmark_app` on this
exact hardware — we're stating the gap plainly rather than implying broader
Core Ultra coverage than what was actually tested.

## The measurement that matters

Latency alone is a half-claim: any quantization makes a model faster, and a
broken policy is very fast indeed. Every configuration is therefore scored on
**both** axes over the **same fixed seeds**:

| axis | how |
| --- | --- |
| latency | ms/inference, p50 + p95, `benchmark_app -hint latency` |
| capability | closed-loop task success rate on `AlohaTransferCube-v0`, N seeds |

A quantization that halves latency and drops success rate is reported as a
failure, not a win.

## Configurations under test

| # | policy | runtime | device |
| --- | --- | --- | --- |
| 0 | ACT (52M) | PyTorch | CPU — baseline floor, non-language-conditioned |
| 1 | SmolVLA (450M) | PyTorch fp32 | CPU |
| 2 | SmolVLA | OpenVINO IR fp32 | CPU |
| 3 | SmolVLA | OpenVINO IR INT8 (NNCF `INT8_ASYM`) | CPU |
| 4 | SmolVLA | OpenVINO IR INT8 | GPU (Intel UHD) |

## Environment

`gym_aloha/AlohaTransferCube-v0` — MuJoCo bimanual ALOHA. Success = one arm
picks the cube and **transfers it to the other arm**, so the task cannot be
solved by a single arm. 14-D action space = 2 arms x 7 joints.

Verified running locally: 79.4 ms/step (12.6 Hz) with 480x640 top-camera
rendering, Python 3.11.14.

## Voice layer (Speechmatics)

Speech → text → the natural-language instruction the VLA is conditioned on.
Not a bolted-on transcript box: the spoken phrase is the policy's actual
conditioning input, so changing what you say changes what the arms do.

## Setup

Python 3.11 is required — `labmaze` (via `dm_control`) has no cp313 Windows
wheel and falls back to a bazel source build that fails.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe gym-aloha
uv pip install --python .venv/Scripts/python.exe torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe smoke_env.py   # expect SMOKE_OK
uv pip install --python .venv/Scripts/python.exe lerobot==0.4.4 transformers==4.57.6 \
    huggingface_hub sentence-transformers openvino nncf
```

**Get the trained LangACT@200k checkpoint** (public, no token needed — this is
the exact checkpoint every result in this README was measured on):

```bash
huggingface-cli download earthpulse/langact-aloha-multitask \
    --local-dir checkpoint_langact/pretrained_model
```

Then reproduce any number in this README directly, e.g.:

```bash
python bench.py --policy langact --episodes 20 --seed0 0
python bench.py --policy langact --episodes 20 --seed0 0 --swap-instruction
benchmark_app -m ir/langact_int8.xml -hint latency -d GPU   # after export_langact.py
```

## Results — ACT, 20 seeds, i3-1215U

| config | success | forward p50 | failed seeds |
| --- | --- | --- | --- |
| random (control) | 0/20 (0%) | — | all |
| ACT PyTorch, CPU | 14/20 (70%) | 351 ms | 2,5,7,8,12,15 |
| OpenVINO INT8, CPU | 13/20 (65%) | 289 ms | 2,5,7,8,12,**13**,15 |
| OpenVINO INT8, Intel iGPU | **14/20 (70%)** | **85.8 ms** | 2,5,7,8,12,15 |

INT8 weights: 131.8 MB → 33.5 MB (3.9x).

**The episode-level result is the real one.** INT8 on the iGPU failed on exactly
the same six episodes as PyTorch — all 20 outcomes identical, not merely the
same average. INT8 on CPU diverged on one episode (seed 13); same IR, different
runtime kernels and accumulation order, so the two are not bit-identical.
Reporting that is more useful than smoothing it into "70% vs 65%".

⚠️ Latency figures above were collected while other work ran on the same 6-core
CPU and are provisional — one clean idle-machine pass is owed before publication.
Success rates are seed-deterministic and unaffected.

## Results — LangACT (ours, language-conditioned ACT), 20 seeds, i3-1215U

Trained to 200,000 steps (`kaggle_langact.py`, resumed across 5 Kaggle T4
sessions). Same benchmark harness, same 20 fixed seeds as the ACT table above.

| config | success | forward p50 | failed seeds |
| --- | --- | --- | --- |
| LangACT PyTorch, CPU (fp32) | 14/20 (70%) | 541.8 ms | 0,3,10,12,15,16 |
| OpenVINO INT8, CPU | 15/20 (75%) | 371.0 ms | 0,2,3,12,16 |
| OpenVINO INT8, Intel iGPU | 13/20 (65%) | **90.5 ms** | 0,2,3,12,13,15,16 |

INT8 weights: 132.6 MB → 33.7 MB (3.9x).

**Official OpenVINO `benchmark_app -hint latency` numbers** (pure model
inference, no Python/env-step overhead — this is the formal tool, not our own
timer, and is the number the "forward p50" column above is measuring loosely
around):

| runtime | median | average | min | max | throughput |
| --- | --- | --- | --- | --- | --- |
| fp32, CPU | 239.1 ms | 254.9 ms | 227.2 ms | 634.9 ms | 3.92 FPS |
| INT8, CPU | 195.9 ms | 237.7 ms | 160.9 ms | 430.9 ms | 4.20 FPS |
| **INT8, Intel iGPU** | **71.0 ms** | 71.7 ms | 57.5 ms | 232.4 ms | **13.91 FPS** |

Each row ran `benchmark_app`'s synchronous-latency mode for 60 seconds
(236-835 iterations depending on runtime); these are the numbers a judge can
reproduce byte-for-byte with `benchmark_app -m ir/langact_int8.xml -hint
latency -d GPU`, no application code involved.

**We are not claiming INT8-CPU "beats" fp32 because it scored higher (75% vs
70%).** Four seeds (0, 3, 12, 16) fail on every runtime — genuine task
failures, runtime-independent. Beyond that core four, each runtime disagrees
with the others on 1-3 additional seeds (seed 2 and 13 only fail under
quantization; seeds 10 and 15 only fail in fp32 PyTorch). At n=20 a spread of
65-75% across three runtimes of the *same* exported model is within sampling
noise, not a quantization effect — the same pattern the ACT table above
already established (`ov_int8_cpu` 13/20 vs `ov_int8_gpu` 14/20 on identical
IR). We report PyTorch fp32 (70%) as the accuracy figure and the iGPU number
(90.5 ms) as the latency figure; the two should not be read as "the fast
version is also the best version" or vice versa — they are separate axes,
never conflated per the measurement principle at the top of this README.

### The swap test: language does not survive to closed-loop behavior

Tested at 125k, 150k, and 200k steps. At 150k and 200k, the swapped-instruction
run reproduced the **exact same per-seed success/failure pattern** as the
correct-instruction run — not just the same rate, the same 13 (150k) or 14
(200k) seeds succeeding, every time. Swapping "pick up the cube..." for
"insert the peg..." never changed a single episode's outcome.

This is not an early-training artifact: task success climbed the entire way
from 20% (15k steps) to 70% (200k steps, tying the ACT ceiling) while language
sensitivity stayed at exactly zero throughout. Root cause: in
`aloha_sim_multitask_lang`, every training episode pairs one task with one
instruction 1:1 — the visual scene already fully determines the correct
behavior, so there is no training signal that ever rewards reading the
sentence. Confirmed at the numeric level too: exporting to OpenVINO, swapping
the instruction *does* move the raw single-step output (fp32: 2.47e-2, INT8:
3.02e-2 — the export's built-in "is language wired in" parity gate does not
fail). The weight is real and reacts; it is just too small relative to
vision/state to survive 100+ steps of closed-loop feedback and ever change the
final trajectory.

**We are reporting this as the honest result, not retrying until it looks
better.** A language slot can be added to stock ACT for free — 0.2% more
params, 0 ms inference cost (cached embedding, not a forward pass), zero cost
to task performance — but this dataset never gives the model a reason to use
it. That is a real, useful finding about behavior-cloning dataset design,
surfaced only because the benchmark was built to test the actual claim
("language steers the arms") instead of stopping at accuracy.

### Robustness on randomized seeds (not the fixed evaluation set)

All results above use a fixed seed range (0-19) deliberately, so every
checkpoint, runtime, and instruction variant is compared on identical
episodes — a fixed set is what makes the whole swap-test methodology valid at
all (same noise, same scene, only the instruction differs).

As a separate check that this isn't overfit to that specific range, we also
ran LangACT@200k on 10 seeds drawn from OS entropy (`random.sample`, not
0-19, not chosen after seeing results):

```
seeds = [12310, 97979, 15614, 87902, 88268, 3418, 78982, 47294, 99370, 57955]
```

**Result: 8/10 (80%)** — consistent with, and slightly above, the 70% fixed-
seed result (n=10 is small; this is within noise, not an improvement). This
is the evidence that the fixed-seed numbers above generalize rather than
being an artifact of that particular range.

**Video: `media/voice_control_randomized10_seeds.mp4`** — all 10 randomized
episodes back-to-back, each labeled with its seed and a SUCCESS/FAIL banner.
The 2 genuine failures (seeds 78982, 99370) are included and labeled, not cut
— this is the actual footage behind the 8/10 number above, not a highlight
reel of the successes.

## Voice control (Speechmatics) — how we actually make it talk-to-able

The swap test above rules out one design: feeding speech into the policy's
internal language slot and expecting it to steer behavior. It doesn't, and
retraining to fix that needs new data this build window doesn't have time for
(see "The swap test" above). So the voice layer is built around what is
actually true instead of what was hoped: **voice picks which trained skill
runs, and the skill executes on its own measured competence.**

`voice.py` (Speechmatics STT + TTS, both live) turns speech into text and maps
it onto one of two known instructions by keyword overlap — visibly, with the
score shown, never a hidden classifier's unfalsifiable judgment call. It
refuses outright if nothing matches rather than guessing and moving the arm.

`voice_control.py` is the last step: it takes that routed decision and
actually runs the robot.

```bash
python voice_control.py --say "pick up the cube and pass it to your other hand"
python voice_control.py --say "insert the peg into the socket"
```

```
TTS  -> audio/spoken.wav
STT  -> 'Pick up the cube and pass it to your other hand.'
cues -> {'transfer': 3, 'insert': 0}
MAP  -> [transfer] 'Pick up the cube with the right arm and transfer it to the left arm.'
SKILL -> [transfer] running rollout, seed=1 ...
RUN  -> SUCCESS (241 steps, 1511.5 ms/forward avg)
TTS  -> audio/readback.wav ('Task complete. Cube transferred.')
```

Asking for the task that was directly measured at 0/20 (see below) gets an
honest decline instead of a silent failure or a lie:

```
STT  -> 'Insert the peg into the socket.'
cues -> {'transfer': 0, 'insert': 4}
MAP  -> [insert] 'Insert the peg into the socket.'
SKILL -> [insert] NOT READY (measured 0/20 on its own task)
```

(`voice_control.py` passes the routed instruction string through to the
policy for logging/completeness, since the API expects one — but per the swap
test, that string is not what determines behavior. The routing decision above
it is. This distinction is the entire point: **the words genuinely control
what the arm does — through explicit, auditable routing — even though the
neural network's own internal language pathway does not.**)

**LangACT can only do one task well.** Evaluated directly on
`AlohaInsertion-v0` with its own correct instruction (not swapped) — the real
test of whether it learned insertion, not a language-grounding probe — it
scores **0/20 (0%)**. Despite training on both tasks, only cube-transfer
became a usable skill; insertion demos (50 episodes, split with cube-transfer
across 200k steps) were not enough to learn a working policy. The router
above reflects that measured limit rather than attempting a task it cannot do.

## SmolVLA — the language-conditioned policy

Fine-tuned `lerobot/smolvla_base` on the merged 2-instruction dataset:
8,000 steps, batch 64, H100, loss 0.480 -> 0.033. Checkpoint:
`earthpulse/smolvla-aloha-multitask` (private).

**Language grounding is verified, not assumed.** Same observation, two
instructions:

```
action (correct instruction): [-0.00259 -0.99613  1.18982 -0.0037  -0.30867]
action (wrong   instruction): [-0.00119 -0.98446  1.19325 -0.00725 -0.34169]
max|difference| across 14 joints = 0.137674 rad  ->  LANGUAGE IS READ
```

If those two vectors had matched, the policy would be ignoring the instruction
and the Speechmatics layer would be decoration. They differ by ~7.9 degrees of
joint command, so the spoken instruction genuinely steers the arms.

## Results — SmolVLA (the VLA entry), latency on the i3-1215U

All five measured back-to-back on the same idle machine, identical input,
`PERFORMANCE_HINT: LATENCY` (a robot runs one inference at a time; THROUGHPUT
would flatter the numbers while describing a workload nobody has).

| runtime | device | p50 | speedup |
| --- | --- | --- | --- |
| PyTorch fp32 | CPU | 6,876 ms | — |
| OpenVINO fp32 | CPU | 5,652 ms | 1.2x |
| OpenVINO INT8 | CPU | 3,526 ms | 2.0x |
| OpenVINO fp32 | Intel iGPU | 1,521 ms | 4.5x |
| **OpenVINO INT8** | **Intel iGPU** | **1,328 ms** | **5.2x** |

IR weights: 1,574 MB -> 395 MB (4x).

**A 450M language-conditioned VLA, 6.9 s -> 1.33 s per decision, on a $400
laptop with no discrete GPU.**

⚠️ An earlier draft of this table quoted ~14,000 ms for PyTorch. That figure was
taken during a rollout while installs ran on the same 6-core CPU, and it
overstated the speedup roughly 2x. Latency measured under contention is not a
result — every number above was re-taken on an idle machine.

### What had to be true for the export to work

`sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise)` is
pure tensors in, tensors out. Three things stay OUTSIDE the graph on purpose:
tokenisation (not a tensor op), the action queue (Python control flow), and the
noise sample (passed in explicitly, so parity against PyTorch is deterministic
and RNG never enters the trace). `prepare_images()` returns *lists* of
per-camera tensors, which a traced graph cannot accept, so they are unpacked to
positional args and rebuilt inside `forward()`.

## The headline result: language is load-bearing

SmolVLA (8k steps) on OpenVINO INT8 / Intel iGPU, `AlohaTransferCube-v0`,
10 fixed seeds, identical noise. The only variable is the sentence.

| instruction given | success |
| --- | --- |
| "Pick up the cube with the right arm and transfer it to the left arm." | **3/10 (30%)** |
| "Insert the peg into the socket." (wrong task) | **0/10 (0%)** |

Under the null hypothesis that the policy ignores language, scoring 0/10 when
the true rate is 0.30 has probability 0.7^10 = **2.8%**. So the instruction is
doing real work, not decorating the demo — which is exactly what a voice
interface has to be able to claim.

Reproduce:

```bash
python bench.py --policy smolvla_ov_int8_gpu --episodes 10
python bench.py --policy smolvla_ov_int8_gpu --episodes 10 --swap-instruction
```

## Which model wins what

| | ACT | SmolVLA |
| --- | --- | --- |
| task success | **14/20 (70%)** | 3/10 (30%) |
| reads language | no | **yes (30% -> 0% on swap)** |
| params | 52M | 450M |
| PyTorch CPU latency | 351 ms | 6,876 ms |
| best Intel config | 85.8 ms (INT8 iGPU) | 1,328 ms (INT8 iGPU) |
| speedup | 4.1x | 5.2x |

**ACT wins the task outright, and it is reported that way.** ACT is purpose-built
for this benchmark and ships a fully-trained public checkpoint; SmolVLA got
8,000 steps on a hackathon budget. But ACT takes no language input at all, so it
cannot be spoken to — a voice interface on ACT would be a lie, since the words
could not reach the policy. The two results are separate claims, both measured.

## Status

- [x] Bimanual sim running locally, benchmarked
- [x] ACT baseline, closed-loop success rate over 20 seeds
- [x] OpenVINO export + INT8, with a parity gate before benchmarking
- [x] Benchmark harness (latency + success, fixed seeds, forward vs replay split)
- [x] Multi-task dataset merged (100 eps / 45k frames / 2 instructions)
- [x] SmolVLA training config validated on CPU (2 steps, checkpoint written)
- [x] LangACT trained 15k→200k steps (5 chained Kaggle T4 runs), ties ACT ceiling (70%)
- [x] Instruction-swap grounding test (`bench.py --swap-instruction`) — run at 3 checkpoints, decisive null result
- [x] LangACT → OpenVINO → INT8 → same harness (132.6MB → 33.7MB, parity verified)
- [x] Speechmatics voice conditioning — via explicit task routing (`voice_control.py`), not internal language conditioning (measured not to work)
- [ ] Clean idle-machine latency re-run
- [ ] SmolVLA fine-tune on Lightning H100 (see LIGHTNING.md)
- [ ] SmolVLA → OpenVINO → INT8 → same harness
- [ ] Demo video, slides, cover image, hosted demo URL

## Traps hit (all of them "success" that wasn't)

1. **ACT checkpoint normalization silently dropped.** Loading it warns about
   "unexpected keys" and returns a policy with no normalization — LeRobot 0.4.x
   moved it to a processor pipeline. Exit code 0, plausible actions, 0% success.
   Would have been read as "weak baseline".
2. **Latency averaged across a chunked policy.** ACT runs the net once per 100
   steps; timing every call reported 8.7 ms for a 300 ms model.
3. **A script exited 0 printing nothing.** `lerobot` configures the root logger
   on import, making a later `logging.basicConfig` a no-op.
4. **Single-instruction dataset.** Training SmolVLA on it teaches the model that
   language is a constant, making any voice interface decorative.
5. `transformers` 5.x breaks LeRobot's ACT config (`non-default argument
   'backbone_cfg' follows default argument`) — pin `<5`.
6. Intel's Pi0.5 guide passes `all_layers=True` to `compress_weights`; NNCF
   3.3.0 rejects it for INT8 modes.
