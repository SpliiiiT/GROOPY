"""The curated word-sign vocabulary — THE shared list that ties both tracks together.

"This list is the contract" (docs/data_contract.md): it is simultaneously
  - the Recognition dynamic-word LSTM's output classes, and
  - the Synthesis clip-dictionary keys (which gloss has a video clip).

Change the vocabulary here, once, and both directions update. Any word NOT in this list is
handled by fingerspelling (letters) on both sides, so the system never dead-ends.

Glosses are chosen to overlap the WLASL100 word-level dataset so one download can supply
both LSTM training sequences and Synthesis playback clips.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Curated ~20-word starter vocabulary (all present in WLASL100). Lowercase glosses.
GLOSSES: list[str] = [
    "hello",
    "thanks",
    "please",
    "yes",
    "no",
    "sorry",
    "help",
    "want",
    "need",
    "name",
    "you",
    "me",
    "good",
    "bad",
    "happy",
    "sad",
    "eat",
    "drink",
    "friend",
    "love",
]

# gloss -> clip filename (under synthesis/clips/). One canonical clip per gloss for
# playback; sequence_data.py may use additional per-gloss samples for LSTM training.
CLIP_MAP: dict[str, str] = {g: f"{g}.mp4" for g in GLOSSES}

NUM_WORDS = len(GLOSSES)

# Stable index <-> gloss mapping for the LSTM's softmax classes.
GLOSS_TO_INDEX: dict[str, int] = {g: i for i, g in enumerate(GLOSSES)}
INDEX_TO_GLOSS: dict[int, str] = {i: g for g, i in GLOSS_TO_INDEX.items()}


def has_clip(gloss: str) -> bool:
    """True if this gloss has a word-sign clip (else the caller should fingerspell it)."""
    return gloss.lower() in CLIP_MAP


def resolve(gloss: str, clips_dir: Path) -> Optional[Path]:
    """Return the clip path for a gloss under clips_dir, or None if not in the vocabulary.

    Existence on disk is the caller's concern — this only maps a known gloss to where its
    clip should live, so it stays importable without the clips being downloaded yet.
    """
    name = CLIP_MAP.get(gloss.lower())
    return (clips_dir / name) if name else None
