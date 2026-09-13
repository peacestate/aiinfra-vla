# Bimanual VLA on Intel

**Tagline:** A measurement, not a demo — bimanual robot manipulation on Intel
hardware, with a rigorous test of whether voice control actually works.

## Track
Intel — Bimanual VLA Manipulation (online) + Speechmatics bonus (Best Use of
Speechmatics)

## The problem

A voice-controlled bimanual robot arm needs two things, not one:
1. It has to actually complete the physical task (one arm picks up an object,
   hands it to the other arm).
2. Its behavior has to actually depend on the spoken words — otherwise the
   voice interface is decoration, and the robot does the same thing no
   matter what you say.

Most projects measure #1 and assume #2. This project measures both,
rigorously, and reports what it actually finds — including when the answer
isn't the one we hoped for.

## What it does

We trained **LangACT** — stock ACT (a 52M-parameter bimanual manipulation
policy) with a language instruction written into an internal slot the
architecture already has but never uses. This costs 0.2% more parameters and
0ms of inference time (the instruction is a known string, embedded once and
cached — never re-run through a network).

Trained from 15,000 to 200,000 steps across 5 chained Kaggle T4 sessions
(each resuming the previous checkpoint), it reaches **70% closed-loop success**
on the standard `AlohaTransferCube-v0` bimanual benchmark (MuJoCo simulation,
20 fixed seeds) — tying the no-language ACT baseline exactly.

Then we ran the test that actually matters: **the instruction-swap test.**
Feed the model the *wrong* instruction while running the same task. If
language steers behavior, success should collapse. We ran this at three
separate checkpoints (125k, 150k, 200k steps). At 150k and 200k, the swapped-
instruction run produced the **exact same per-seed outcomes** as the correct
instruction — not just the same success rate, but literally the same
episodes succeeding and failing, every time.

We diagnosed why: in the training dataset, every episode pairs one task with
one instruction 1:1, so the visual scene alone fully determines the correct
behavior. Nothing in training ever rewards the model for actually reading the
sentence. Confirmed even at the numeric level — exporting to OpenVINO,
swapping the instruction *does* move the raw model output slightly (the
weight is real and reacts), but that perturbation is too small to survive
100+ steps of closed-loop visual feedback and ever change the outcome.

**Rather than hide this, we built the voice interface around it.** Since
in-model language conditioning doesn't work, speech is routed explicitly
instead: Speechmatics STT transcribes the command, keyword matching resolves
it to a known task, and that task then actually runs. If you ask for a task
the system can't do (peg insertion — measured at 0/20, a real capability
gap), it says so and declines, rather than attempting and silently failing.
This is a real, working, honestly-scoped voice-controlled robot — Speechmatics
STT and TTS both genuinely in the loop, the spoken words genuinely
determining what the arm does or doesn't do.

## Intel optimization

Exported to OpenVINO IR and compressed to INT8 with NNCF (132.6MB → 33.7MB,
3.9x smaller), parity-checked against the PyTorch reference before
benchmarking. Official `benchmark_app -hint latency` numbers on our own Core
i3 (Alder Lake) laptop — CPU and Intel UHD iGPU, no NPU, disclosed plainly:

| runtime | median latency | throughput |
|---|---|---|
| fp32, CPU | 239.1 ms | 3.92 FPS |
| INT8, CPU | 195.9 ms | 4.20 FPS |
| **INT8, Intel iGPU** | **71.0 ms** | **13.91 FPS** |

## Challenges we ran into

- A laptop-side TLS interception issue caused sustained downloads (model
  checkpoints, Kaggle kernel outputs) to fail unpredictably — worked around
  with a chunked, resumable downloader.
- Diagnosing *why* the swap test failed took real investigation: ruling out
  "not enough training" (task success kept climbing right through 200k
  steps) versus the real cause, a dataset confound where language is always
  redundant with vision.
- The documented Hugging Face checkpoint-download command
  (`huggingface-cli download`) turned out to crash on Windows before
  downloading anything — caught during our own fresh-clone reproducibility
  test, fixed to use `hf download` instead.

## Accomplishments we're proud of

- A rigorous instruction-swap methodology, run at three separate training
  checkpoints, that caught a real dataset-design problem a naive
  accuracy-only evaluation would have completely hidden.
- Choosing to report a null result honestly rather than reframe it, and then
  building a genuinely working alternative (voice routing) instead of
  quietly dropping the voice feature.
- Verified, end-to-end reproducibility: a fresh clone, independent checkpoint
  download, and eval run reproduced our exact reported number (14/20, 70%,
  identical failed seeds) bit-for-bit.

## What's next

- A follow-up dataset with a small number of "conflict" episodes (same
  visual scene, different instructions, only one correct) would directly
  test whether the architecture *could* learn real language-dependence given
  the right training signal — this needs new data collection, not more
  training on the current dataset.
- Extending the working task set beyond cube-transfer (peg-insertion
  currently fails at 0/20 and would need substantially more demonstration
  data).

## Built with

Python, PyTorch, LeRobot (ACT), OpenVINO, NNCF, gym-aloha / MuJoCo,
Speechmatics (STT + TTS), Hugging Face Hub, Kaggle (T4 training).

## Links

- **Repository:** https://github.com/peacestate/aiinfra-vla
- **Trained checkpoint:** https://huggingface.co/earthpulse/langact-aloha-multitask
- **Pitch video (100s):** `media/pitch_video.mp4` — narrated over 100% real robot
  footage (no slides): all 10 randomized-seed episodes, picture-in-picture
  showing both the full scene and what the robot's own policy sees, narration
  calling out both real failures honestly as they happen, ending on the true
  8/10 score and the measured latency numbers
- **Raw demo footage (10 randomized seeds, unedited, unnarrated):** `media/voice_control_randomized10_seeds.mp4`
- **Slide deck:** `slides/deck.pdf`
