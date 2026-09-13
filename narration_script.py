"""Narration script for the pitch video, one line per seed segment.

Timed to roughly fit each seed's natural episode length (7-10s at natural
speaking pace, ~2.3 words/sec). If narration runs longer than the episode,
the video assembly script holds the final frame to cover it, rather than
cutting the line short.
"""

LINES = [
    (12310, True,  "Ten episodes. Real seeds, picked at random, not cherry-picked. Watch seed one line up the handoff."),
    (97979, True,  "Seed two. Same weights, same policy, a different starting position."),
    (15614, True,  "Seed three succeeds too. Three for three so far."),
    (87902, True,  "Seed four gets there as well."),
    (88268, True,  "Seed five. Five in a row now."),
    (3418,  True,  "Seed six also lands it."),
    (78982, False, "Seed seven fails. The right arm can't get a clean grip here — we're showing you this, not cutting it."),
    (47294, True,  "Seed eight recovers. Back to a success."),
    (99370, False, "Seed nine fails too. Our second real failure, included honestly."),
    (57955, True,  "And seed ten succeeds. Final score: eight out of ten, on seeds we never picked."),
]

OUTRO = ("That ties our no-language baseline. And on Intel's integrated GPU, "
         "this runs in ninety milliseconds — the real number, not a highlight reel.")
