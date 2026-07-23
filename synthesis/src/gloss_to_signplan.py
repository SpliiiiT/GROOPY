"""Gloss sequence -> SignPlan (the thing the player renders).

For each gloss:
  - if it has a word-sign clip in the shared vocabulary  -> a WordClip step (play the clip)
  - otherwise                                            -> a Fingerspell step (spell it)

Fingerspelling reuses the ASL-alphabet letter images the Recognition track already trains
on (shared.config.LETTERS_DIR), so the system never dead-ends on an unknown word (names,
rare words). This module is pure/headless — it builds the plan; player.py does the I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from shared import vocabulary as vocab
from shared.config import CLIPS_DIR, LETTERS_DIR


@dataclass
class WordClip:
    """Play the sign-video clip for a known gloss."""

    gloss: str                                    # the word, e.g. "hello"
    clip_path: Path                               # where its .mp4 lives
    kind: str = field(default="word", init=False) # tag for the player/UI (not a constructor arg)
    # Sentiment emphasis (Decision A2, see synthesis/src/pipeline.apply_sentiment): extra
    # pause after the clip (ms) and how many times to replay it. Both default to "no effect"
    # so an unmodified plan behaves exactly as before.
    hold_ms: int = 0                              # extra pause after the clip (0 = none)
    repeat: int = 1                               # times to play the clip (1 = once)


@dataclass
class Fingerspell:
    """Spell a word letter-by-letter using ASL-alphabet letter images."""

    word: str                                     # the word being spelled, e.g. "oussama"
    letters: list[str]  # uppercase A-Z, in order
    kind: str = field(default="fingerspell", init=False)   # tag for the player/UI
    # Sentiment emphasis — see WordClip. Not set by apply_sentiment today (fingerspelling is
    # mostly out-of-vocabulary names/rare words, not the emotional content of an utterance),
    # but honoured by player.py for symmetry if ever set.
    hold_ms: int = 0
    repeat: int = 1

    def letter_dirs(self, letters_dir: Path = LETTERS_DIR) -> list[Path]:
        """Per-letter directory of ASL-alphabet images (player picks a frame from each)."""
        return [letters_dir / ltr for ltr in self.letters]   # e.g. .../letters/O, .../letters/U, ...


Step = Union[WordClip, Fingerspell]               # a plan step is one of these two


@dataclass
class SignPlan:
    """An ordered list of render steps produced from a gloss sequence."""

    source_text: Optional[str]                    # the original text (for reference/logging)
    glosses: list[str]                            # the gloss sequence this plan came from
    steps: list[Step]                             # the ordered steps the player will render

    @property
    def fingerspelled_glosses(self) -> list[str]:
        return [s.word for s in self.steps if isinstance(s, Fingerspell)]   # which words got spelled

    def summary(self) -> str:
        # Human-readable one-liner, e.g. "hello [O-U-S-S-A-M-A] thanks" — handy for the UI/logs
        parts = []
        for s in self.steps:
            parts.append(s.gloss if isinstance(s, WordClip) else f"[{'-'.join(s.letters)}]")
        return " ".join(parts)


def _letters_of(gloss: str) -> list[str]:
    """A-Z letters of a gloss, uppercased, for fingerspelling (non-letters dropped)."""
    return [c.upper() for c in gloss if c.isalpha() and c.isascii()]   # keep only A-Z, uppercase


def build_sign_plan(
    glosses: list[str],
    source_text: Optional[str] = None,
    clips_dir: Path = CLIPS_DIR,
    letters_dir: Path = LETTERS_DIR,
) -> SignPlan:
    """Turn a gloss list into a renderable SignPlan (word clips + fingerspell fallback)."""
    steps: list[Step] = []
    for g in glosses:
        clip = vocab.resolve(g, clips_dir)        # is this gloss in the 20-word vocabulary (has a clip)?
        if clip is not None:
            steps.append(WordClip(gloss=g, clip_path=clip))   # known word -> play its clip
        else:
            letters = _letters_of(g)              # unknown word -> break into letters
            if letters:  # skip empties (e.g. stray punctuation that slipped through)
                steps.append(Fingerspell(word=g, letters=letters))   # -> fingerspell it
    return SignPlan(source_text=source_text, glosses=list(glosses), steps=steps)
