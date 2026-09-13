"""End-to-end voice-controlled robot demo: speech -> routed task -> real rollout.

This is additive on top of voice.py and bench.py — neither is modified.
voice.py already turns speech into a resolved instruction (or refuses on
ambiguous speech); this script is the missing last step, actually running the
robot on that resolved intent and reporting the true outcome back by voice.

Honesty note, following from a measured result: LangACT's internal language
conditioning does not survive closed-loop rollout (verified at 125k/150k/200k
training steps — swapping the instruction never changes a single episode's
outcome; see README "The swap test"). So this script does NOT feed the spoken
instruction into the policy's language slot expecting it to steer behavior.
Instead, voice picks WHICH SKILL runs — an explicit, auditable routing
decision — and the selected skill executes on its own trained competence.

Also measured: the "insert" skill does not exist yet. LangACT scores 0/20 on
AlohaInsertion-v0 even with the correct instruction — training on it never
produced a working policy, unlike the 70% cube-transfer skill. So the router
below has exactly one real skill wired up, and says so honestly when asked for
the other one, rather than attempting a task it cannot do and silently
failing.
"""
import argparse
import pathlib
import sys

import voice
from bench import rollout, INSTRUCTION

CHECKPOINT = "checkpoint_langact/pretrained_model"

# Which routed keys have an actual working, evaluated skill behind them.
# transfer: 14/20 (70%), ties the ACT ceiling, measured at 200k steps.
# insert: 0/20 (0%) measured directly on AlohaInsertion-v0 — not a skill yet.
WORKING_SKILLS = {"transfer"}


def load_policy():
    from policy_langact import LangACT
    return LangACT(CHECKPOINT).warm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", help="synthesize this phrase, then transcribe and "
                                  "act on it (no microphone needed)")
    ap.add_argument("--wav", help="transcribe an existing wav instead")
    ap.add_argument("--voice", default="jack")
    ap.add_argument("--seed", type=int, default=0, help="rollout seed")
    args = ap.parse_args()

    if args.say:
        wav = voice.synthesize(args.say, "audio/spoken.wav", voice=args.voice)
        print(f"TTS  -> {wav}")
    elif args.wav:
        wav = args.wav
    else:
        raise SystemExit("pass --say TEXT or --wav FILE")

    text = voice.transcribe(wav)
    print(f"STT  -> {text!r}")

    key, instruction, scores = voice.map_instruction(text)
    print(f"cues -> {scores}")

    if instruction is None:
        msg = "I didn't catch a task I know. I can transfer the cube."
        print(f"MAP  -> UNRESOLVED. {msg}")
        voice.synthesize(msg, "audio/readback.wav", voice=args.voice)
        return

    print(f"MAP  -> [{key}] {instruction!r}")

    if key not in WORKING_SKILLS:
        # Measured, not guessed: this skill scores 0/20 on its own task. Saying
        # so and refusing is the honest move — attempting it would silently
        # fail and misrepresent what the system can do.
        msg = (f"I understood you want me to {key}, but I haven't learned that "
               f"skill well enough yet — I can transfer the cube.")
        print(f"SKILL -> [{key}] NOT READY (measured 0/20 on its own task)")
        voice.synthesize(msg, "audio/readback.wav", voice=args.voice)
        return

    print(f"SKILL -> [{key}] running rollout, seed={args.seed} ...")
    policy = load_policy()
    # Instruction is passed for completeness/logging, not because it steers
    # behavior — see the module docstring. The routing decision above is what
    # actually determined which skill runs.
    success, latencies = rollout(policy, args.seed, instruction=instruction)
    fwd = [ms for ms, is_fwd in latencies if is_fwd]
    outcome = "SUCCESS" if success else "fail"
    print(f"RUN  -> {outcome} ({len(latencies)} steps, "
          f"{sum(fwd)/len(fwd) if fwd else float('nan'):.1f} ms/forward avg)")

    msg = f"Task complete. {'Cube transferred.' if success else 'I dropped it, sorry.'}"
    voice.synthesize(msg, "audio/readback.wav", voice=args.voice)
    print(f"TTS  -> audio/readback.wav ({msg!r})")


if __name__ == "__main__":
    main()
