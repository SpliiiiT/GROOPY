"""analyze(text) -> Sentiment.

Backend seam for the partner. The default backend is a tiny, dependency-free lexicon
classifier so the pipeline runs out of the box; swap in a real model (e.g. a HuggingFace
`transformers` pipeline) by implementing `_Backend.score` and setting `set_backend(...)`.

Returns the shared `Sentiment(label, score)` from shared/contract.py — the same object the
Token contract carries — so there is one sentiment type across the whole system.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol

from shared.contract import Sentiment

# --- Tiny built-in lexicon (stub). Not exhaustive — a placeholder the partner replaces. ---
_POSITIVE = {
    "good", "great", "happy", "love", "like", "thanks", "thank", "nice", "awesome",
    "wonderful", "glad", "excited", "please", "yes", "friend", "fun", "well", "best",
}
_NEGATIVE = {
    "bad", "sad", "hate", "angry", "no", "sorry", "terrible", "awful", "wrong", "sick",
    "tired", "hurt", "pain", "worst", "cant", "never", "unhappy", "upset",
}
_NEGATORS = {"not", "no", "never", "dont", "cant", "wont"}

_WORD_RE = re.compile(r"[a-z']+")


class _Backend(Protocol):
    def score(self, text: str) -> Sentiment: ...


class LexiconBackend:
    """Dependency-free rule-based sentiment. Handles simple negation ('not good')."""

    def score(self, text: str) -> Sentiment:
        words = _WORD_RE.findall(text.casefold())
        pos = neg = 0
        negate = False
        for w in words:
            if w in _NEGATORS:
                negate = True
                continue
            if w in _POSITIVE:
                neg, pos = (neg + 1, pos) if negate else (neg, pos + 1)
            elif w in _NEGATIVE:
                pos, neg = (pos + 1, neg) if negate else (pos, neg + 1)
            negate = False  # negation only affects the next sentiment word

        total = pos + neg
        if total == 0:
            return Sentiment(label="neutral", score=0.5)
        if pos == neg:
            return Sentiment(label="neutral", score=0.5)
        label = "positive" if pos > neg else "negative"
        score = max(pos, neg) / total  # confidence/intensity in [0.5, 1.0]
        return Sentiment(label=label, score=score)


_backend: _Backend = LexiconBackend()


def set_backend(backend: _Backend) -> None:
    """Swap the sentiment backend (e.g. a real transformer model). Partner hook."""
    global _backend
    _backend = backend


def analyze(text: Optional[str]) -> Optional[Sentiment]:
    """Return Sentiment for `text`, or None for empty input."""
    if not text or not text.strip():
        return None
    return _backend.score(text)
