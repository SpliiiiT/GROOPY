"""Synthesis orchestration: (speech | text) -> [sentiment] -> gloss -> SignPlan.

This is the single entry point the desktop demo (and any UI) calls. It ties together the
text-to-gloss rules, the clip/fingerspell planner, and the optional sentiment annotation.

The sentiment SEAM lives here as `apply_sentiment` — Decision A2 (see docs/sentiment_options.md):
strong, non-neutral sentiment adds emphasis (a held pause + a replay) to the WordClip steps of
the plan, like tone of voice. A3 (avatar facial expression) is still out of scope — no avatar
output exists to drive.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.config import CLIPS_DIR, LETTERS_DIR
from shared.contract import Sentiment

from .gloss_to_signplan import SignPlan, WordClip, build_sign_plan
from .text_to_gloss import text_to_gloss

# Decision A2 tuning: only emphasise CONFIDENT non-neutral sentiment (score is [0,1]
# confidence/intensity of the label, per shared/contract.Sentiment) — a borderline call
# shouldn't visibly change playback.
EMPHASIS_SCORE_THRESHOLD = 0.75
EMPHASIS_HOLD_MS = 400  # extra pause held after the clip finishes
EMPHASIS_REPEAT = 2     # replay the clip this many times


@dataclass
class Result:
    """Everything a UI needs to render one utterance."""

    source_text: str
    glosses: list[str]
    plan: SignPlan
    sentiment: Optional[Sentiment] = None


def apply_sentiment(plan: SignPlan, sentiment: Optional[Sentiment]) -> SignPlan:
    """Decision A2: strong, non-neutral sentiment emphasises the plan's WordClip steps
    (held pause + replay) — like tone of voice. Mutates and returns `plan`; a weak/neutral/
    missing sentiment leaves it untouched, so this stays a no-op in exactly those cases.
    """
    if sentiment is None or sentiment.label == "neutral":
        return plan
    if sentiment.score < EMPHASIS_SCORE_THRESHOLD:
        return plan
    for step in plan.steps:
        if isinstance(step, WordClip):
            step.hold_ms = EMPHASIS_HOLD_MS
            step.repeat = EMPHASIS_REPEAT
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
