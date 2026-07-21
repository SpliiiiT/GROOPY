"""Train the from-scratch sentiment candidate: TF-IDF + Logistic Regression on IMDB.

CPU, seconds to train — unlike the vision bake-offs this needs no GPU/Colab.

Usage:
  python -m sentiment.src.train_scratch
"""
from __future__ import annotations

import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import MODELS_DIR
from .data import load_imdb_text


def main() -> None:
    train_texts, train_labels, _, _ = load_imdb_text()

    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, train_labels)
    train_seconds = time.time() - t0

    out_path = MODELS_DIR / "scratch_sentiment.pkl"
    with open(out_path, "wb") as f:
        pickle.dump((vectorizer, clf), f)
    print(f"trained on {len(train_texts)} reviews in {train_seconds:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
