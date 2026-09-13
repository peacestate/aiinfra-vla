"""The whole system, end to end: speak -> robot acts -> video.

    python demo.py --say "grab the block and pass it to the other arm"
    python demo.py --wav recording.wav

Chain:
    Speechmatics TTS  (only when --say, to make audio without a microphone)
      -> Speechmatics STT   (the real speech-to-text step)
      -> instruction mapping (onto a trained instruction, or refuse)
      -> Speechmatics TTS   (confirm-before-actuate readback)
      -> SmolVLA on OpenVINO INT8 / Intel iGPU
      -> MP4 with the spoken words and the resolved instruction on screen

Two design choices worth defending out loud:

  * The policy knows exactly TWO instructions. Free speech is *mapped* onto one
    of them and the mapping is shown on screen. Pretending to understand open
    vocabulary would be a claim that dies under one question.
  * Unrecognised speech REFUSES to move the arms. A robot that guesses when it
    did not understand is worse than one that stops.
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--say", help="synthesize this phrase, then transcribe it back")
    g.add_argument("--wav", help="use an existing recording")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="media/voice_demo.mp4")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=25)
    ap.add_argument("--no-readback", action="store_true")
    a = ap.parse_args()

    import voice
    from render_rollout import label, rollout_frames

    pathlib.Path("audio").mkdir(exist_ok=True)

    if a.say:
        print(f'[1/5] TTS  synthesising "{a.say}"', flush=True)
        wav = voice.synthesize(a.say, "audio/spoken.wav")
    else:
        wav = a.wav
    print(f"[2/5] STT  transcribing {wav}", flush=True)
    text = voice.transcribe(wav)
    print(f"      heard: {text!r}", flush=True)

    key, instruction, scores = voice.map_instruction(text)
    print(f"[3/5] MAP  cue scores {scores}", flush=True)
    if instruction is None:
        print("      UNRESOLVED — the policy knows only:")
        for k, v in voice.TRAINED.items():
            print(f"        [{k}] {v}")
        print("      Refusing to move the arms on a guess.")
        if not a.no_readback:
            voice.synthesize("I did not understand that instruction. Not moving.",
                             "audio/readback.wav")
        return
    print(f"      resolved -> [{key}] {instruction!r}", flush=True)

    if not a.no_readback:
        msg = f"Heard {key}. Executing."
        voice.synthesize(msg, "audio/readback.wav")
        print(f"[4/5] TTS  readback: {msg!r}", flush=True)

    print("[5/5] ACT  running SmolVLA on OpenVINO INT8 / Intel iGPU ...", flush=True)
    from policy_smolvla_ov import SmolVLAOpenVINO

    p = SmolVLAOpenVINO("ir/smolvla_int8.xml", "GPU",
                        n_action_steps=a.n_action_steps)
    frames, success = rollout_frames(p, a.seed, instruction, raw=True)

    # Re-label with the full provenance: what was said, what it resolved to,
    # and what happened. The chain is the point, not just the arms moving.
    spoken = a.say or pathlib.Path(wav).name
    verdict = "SUCCESS" if success else "FAILED"
    colour = (120, 255, 120) if success else (255, 120, 120)
    frames = [
        label(np.asarray(f),
              [f'you said: "{text}"',
               f"resolved: {instruction}",
               f"SmolVLA INT8 / Intel iGPU  -  seed {a.seed}  -  {verdict}"],
              colour)
        for f in frames
    ]

    import imageio.v2 as imageio

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=a.fps, macro_block_size=1)
    print(f"\nwrote {a.out} ({len(frames)/a.fps:.1f}s) — {verdict}")


if __name__ == "__main__":
    main()
