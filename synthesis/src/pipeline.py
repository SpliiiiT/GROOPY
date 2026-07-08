"""Synthesis orchestration: (speech | text) -> [sentiment] -> gloss -> SignPlan.

This is the single entry point the desktop demo (and any UI) calls. It ties together the
text-to-gloss rules, the clip/fingerspell planner, and the optional sentiment annotation.

The sentiment SEAM lives here as `apply_sentiment` — currently a PASS-THROUGH no-op. When
you and your partner decide what sentiment should drive (a displayed label, slower/emphasised
playback, avatar expression), implement it there; nothing else in the pipeline needs to
change. The computed Sentiment is already carried on the Result today.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.config import CLIPS_DIR, LETTERS_DIR
from shared.contract import Sentiment

from .gloss_to_signplan import SignPlan, build_sign_plan
from .text_to_gloss import text_to_gloss


@dataclass
class Result:
    """Everything a UI needs to render one utterance."""

    source_text: str
    glosses: list[str]
    plan: SignPlan
    sentiment: Optional[Sentiment] = None


def apply_sentiment(plan: SignPlan, sentiment: Optional[Sentiment]) -> SignPlan:
    """SEAM (currently no-op): let sentiment modulate the SignPlan.

    Deliberately does nothing yet — the team hasn't decided sentiment's behavioural role.
    Candidates when you do: attach per-step emphasis/speed, reorder, or add facial-marker
    cues (avatar path only). Return the (possibly modified) plan.
    """
    return plan


def _analyze(text: str) -> Optional[Sentiment]:
    """Call the sentiment module if importable; never let it break synthesis."""
    try:
        from sentiment import analyze
    except Exception:
        return None
    try:
        return analyze(text)
    except Exception:
        return None


def synthesize(
    text: Optional[str] = None,
    wav: Optional[str] = None,
    with_sentiment: bool = True,
    drop_stopwords: bool = True,
    clips_dir: Path = CLIPS_DIR,
    letters_dir: Path = LETTERS_DIR,
) -> Result:
    """Produce a renderable Result from typed text or a speech wav.

    Provide exactly one of `text` or `wav`. If `wav` is given, it is transcribed via asr.py
    (an optional dependency) and the resulting text feeds the same downstream path.
    """
    if wav is not None:
        from .asr import transcribe  # lazy: ASR backend is optional
        text = transcribe(wav)
    if text is None:
        raise ValueError("synthesize() needs either text= or wav=.")

    glosses = text_to_gloss(text, drop_stopwords=drop_stopwords)
    plan = build_sign_plan(glosses, source_text=text, clips_dir=clips_dir, letters_dir=letters_dir)

    sentiment = _analyze(text) if with_sentiment else None
    plan = apply_sentiment(plan, sentiment)  # no-op today; documented seam

    return Result(source_text=text, glosses=glosses, plan=plan, sentiment=sentiment)
