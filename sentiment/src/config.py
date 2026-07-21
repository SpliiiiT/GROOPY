"""Central configuration for the Sentiment bake-off (mirrors recognition/src/config.py)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "sentiment" / "results"
MODELS_DIR = REPO_ROOT / "sentiment" / "models"

for _d in (RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# IMDB via tf.keras.datasets.imdb — no extra download infra (bundled with TF, which the
# project already depends on), decoded back to raw text via its own word index.
IMDB_NUM_WORDS = 20000
TEST_SIZE = 500  # held-out slice for the bake-off's fixed protocol — kept small so scoring
                  # 2 pretrained transformers on CPU stays fast (~1-2 min each, not tens of min)

# accuracy/latency/size, like the CNN/word bake-offs. NOTE: the latency weight here (0.3) was
# set assuming sentiment behaves like the per-frame vision models, where 20-80ms differences
# are directly felt. It doesn't: sentiment runs once per typed/spoken utterance (a button
# click), where even 80ms is imperceptible — this weighting doesn't actually matter for the
# deployment decision. What DOES matter, and this scorecard can't see at all, is that IMDB
# (the accuracy metric's source) has no neutral examples, so it can't reward/penalise neutral
# detection — see models.py's RECOMMENDED_MODEL and docs/sentiment_options.md for how that
# was actually caught and corrected.
SENTIMENT_SCORECARD_WEIGHTS = {
    "accuracy": 0.5,
    "latency": 0.3,
    "size": 0.2,
}
