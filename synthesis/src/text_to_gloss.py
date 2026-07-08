"""English text -> ASL gloss sequence.

PoC-level ONLY. Real ASL grammar reorders and drops words, uses classifiers, spatial
grammar, and non-manual markers — none of that is modelled here. This is a deliberately
simple, deterministic, dependency-free rule set good enough to drive the clip player:

    lowercase -> strip punctuation -> tokenise on whitespace -> drop a few function words
    -> map a couple of common synonyms -> keep the rest as glosses.

Words that survive but have no clip in the vocabulary are still returned as glosses; the
downstream planner (gloss_to_signplan) decides to fingerspell them.
"""
from __future__ import annotations

import re
import unicodedata

# Function words that ASL typically omits. Small on purpose — over-dropping hurts meaning.
_STOPWORDS = {"a", "an", "the", "is", "are", "am", "to", "of", "do", "does"}

# Light synonym/normalisation map toward glosses that exist in the vocabulary.
_SYNONYMS = {
    "hi": "hello",
    "hey": "hello",
    "thank": "thanks",
    "thankyou": "thanks",
    "wanna": "want",
    "gonna": "want",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalise(text: str) -> str:
    """Casefold, strip accents, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_to_gloss(text: str, drop_stopwords: bool = True) -> list[str]:
    """Return an ordered list of ASL glosses for `text`.

    drop_stopwords: omit common function words (default True, closer to ASL). Set False to
    keep every word (useful if you'd rather fingerspell/preserve everything).
    """
    words = normalise(text).split()
    glosses: list[str] = []
    for w in words:
        if drop_stopwords and w in _STOPWORDS:
            continue
        glosses.append(_SYNONYMS.get(w, w))
    return glosses
