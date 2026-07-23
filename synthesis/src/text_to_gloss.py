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
    "hi": "hello",         # so "hi" plays the hello clip instead of being fingerspelled
    "hey": "hello",
    "thank": "thanks",
    "thankyou": "thanks",
    "wanna": "want",
    "gonna": "want",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)   # matches anything that isn't a word char or whitespace


def normalise(text: str) -> str:
    """Casefold, strip accents, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)         # decompose accented chars into base + combining mark
    text = "".join(c for c in text if not unicodedata.combining(c))   # drop the combining marks (strip accents)
    text = text.casefold()                             # aggressive lowercase (handles more scripts than .lower())
    text = _PUNCT_RE.sub(" ", text)                    # punctuation -> spaces
    return re.sub(r"\s+", " ", text).strip()           # collapse repeated spaces, trim ends


def text_to_gloss(text: str, drop_stopwords: bool = True) -> list[str]:
    """Return an ordered list of ASL glosses for `text`.

    drop_stopwords: omit common function words (default True, closer to ASL). Set False to
    keep every word (useful if you'd rather fingerspell/preserve everything).
    """
    words = normalise(text).split()                    # clean then split into tokens
    glosses: list[str] = []
    for w in words:
        if drop_stopwords and w in _STOPWORDS:         # skip function words ASL usually omits
            continue
        glosses.append(_SYNONYMS.get(w, w))            # map to a known synonym, else keep the word as-is
    return glosses
