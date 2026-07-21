"""Sentiment bake-off candidates — each implements analyze.py's `_Backend` Protocol
(`.score(text) -> Sentiment`), so any of them can be dropped straight into
`analyze.set_backend()` once picked.

  scratch          TF-IDF + Logistic Regression, trained from scratch on IMDB (train_scratch.py)
  distilbert       Pretrained transformer, general-purpose (SST-2 movie reviews, binary)
  twitter_roberta  Pretrained transformer, tuned on short informal text, natively 3-class

Binary candidates (scratch, distilbert) derive "neutral" from a probability band around 0.5,
since IMDB has no neutral label to train against — see _label_from_prob.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Dict, Optional

from shared.contract import Sentiment

from .config import MODELS_DIR

NEUTRAL_BAND = 0.15  # |p_positive - 0.5| below this -> "neutral" instead of committing to a side


def _label_from_prob(p_positive: float) -> tuple[str, float]:
    """Binary P(positive) -> 3-way (label, score) with a neutral band around 0.5."""
    dist_from_center = abs(p_positive - 0.5)
    if dist_from_center < NEUTRAL_BAND:
        return "neutral", round(1.0 - dist_from_center / NEUTRAL_BAND, 3)
    label = "positive" if p_positive > 0.5 else "negative"
    score = p_positive if label == "positive" else 1.0 - p_positive
    return label, round(score, 3)


class ScratchBackend:
    """TF-IDF + Logistic Regression, trained from scratch on IMDB. See train_scratch.py."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        model_path = model_path or (MODELS_DIR / "scratch_sentiment.pkl")
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found — run `python -m sentiment.src.train_scratch` first."
            )
        with open(model_path, "rb") as f:
            self.vectorizer, self.clf = pickle.load(f)

    def score(self, text: str) -> Sentiment:
        x = self.vectorizer.transform([text])
        p_positive = float(self.clf.predict_proba(x)[0][1])
        label, score = _label_from_prob(p_positive)
        return Sentiment(label=label, score=score)


class PretrainedBackend:
    """Wraps a HuggingFace PyTorch sequence-classification model behind the analyze.py Protocol.

    HuggingFace `transformers` dropped TensorFlow model support in recent major versions (only
    PyTorch classes are available now), so this — unlike the rest of the project — runs on
    PyTorch (CPU-only), isolated to the sentiment track.
    """

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self._torch = torch

    def score(self, text: str) -> Sentiment:
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0).numpy()
        idx = int(probs.argmax())

        if len(self.id2label) == 2:
            # Binary model: recover P(positive), then apply the same neutral-band logic as
            # the scratch model so both binary candidates behave consistently live.
            positive_idx = next(i for i, l in self.id2label.items() if "pos" in l.lower())
            label, score = _label_from_prob(float(probs[positive_idx]))
        else:
            label = self.id2label[idx].lower()
            score = round(float(probs[idx]), 3)
        return Sentiment(label=label, score=score)


REGISTRY: Dict[str, Callable[[], object]] = {
    "scratch": lambda: ScratchBackend(),
    "distilbert": lambda: PretrainedBackend("distilbert-base-uncased-finetuned-sst-2-english"),
    "twitter_roberta": lambda: PretrainedBackend("cardiffnlp/twitter-roberta-base-sentiment-latest"),
}

# evaluate.py's bake-off (IMDB accuracy/latency/size scorecard) technically picks "distilbert" —
# but IMDB has zero neutral examples, so it never penalises a model for failing at neutral
# detection. A second check on realistic app-style sentences (eval_realistic.py) found
# distilbert gets 0/6 neutral examples right (it was only ever trained to force pos/neg), while
# twitter_roberta scores 100% overall on that same set. That's the one that should actually ship.
# See docs/sentiment_options.md and sentiment/results/realistic_eval.json for the full story.
RECOMMENDED_MODEL = "twitter_roberta"


def build(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown sentiment model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name]()


def load_bakeoff_winner():
    """Build whichever candidate evaluate.py's IMDB-accuracy scorecard picked (raw bake-off
    result, currently "distilbert" — see results/winner.json). NOT the deployment
    recommendation; see RECOMMENDED_MODEL / load_recommended_backend() for that."""
    import json

    from .config import RESULTS_DIR

    winner_path = RESULTS_DIR / "winner.json"
    if not winner_path.exists():
        raise FileNotFoundError(
            f"{winner_path} not found — run `python -m sentiment.src.evaluate` first."
        )
    name = json.loads(winner_path.read_text())["model"]
    return build(name)


def load_recommended_backend():
    """Build RECOMMENDED_MODEL — the candidate that actually handles realistic app input well
    (see the module-level comment above), not just the raw IMDB-accuracy scorecard winner.

    Opt into it instead of the default LexiconBackend:
        from sentiment.src import analyze, models
        analyze.set_backend(models.load_recommended_backend())
    """
    return build(RECOMMENDED_MODEL)
