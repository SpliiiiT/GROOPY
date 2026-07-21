"""Sentiment bake-off evaluation — scratch vs. pretrained, one fixed protocol (CRISP-DM: Evaluation).

For every candidate in models.REGISTRY, scores it on a held-out IMDB test slice:
  - accuracy against the binary ground truth. A "neutral" prediction always counts as wrong —
    IMDB has no neutral-labelled examples, so there's nothing to reward being cautious against;
    this deliberately penalises over-using the neutral band.
  - mean latency per call (ms) — full tokenize+inference for the pretrained models, TF-IDF
    transform + predict for the scratch one.
  - model size (MB) — pretrained transformer parameter count vs. the scratch pickle's byte size.

Then applies the weighted scorecard (recognition/src/scorecard.py — fully generic, reused
as-is) and writes a ranked table + winner.json, mirroring recognition/src/evaluate.py.

Usage:
  python -m sentiment.src.evaluate
  python -m sentiment.src.evaluate --candidates scratch,twitter_roberta   # skip a slow one
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

from recognition.src import scorecard as scorecard_mod  # fully generic, no vision coupling

from .config import RESULTS_DIR, SENTIMENT_SCORECARD_WEIGHTS
from .data import load_imdb_text
from .models import REGISTRY


def _size_mb(backend) -> float:
    if hasattr(backend, "model") and hasattr(backend.model, "num_parameters"):
        return backend.model.num_parameters() * 4 / 1e6  # fp32 params -> MB (approximation)
    return len(pickle.dumps((backend.vectorizer, backend.clf))) / 1e6  # ScratchBackend


def evaluate_candidate(name: str, build_fn, texts, labels) -> dict:
    t0 = time.time()
    backend = build_fn()
    load_seconds = time.time() - t0

    correct = 0
    t0 = time.time()
    for text, true_label in zip(texts, labels):
        result = backend.score(text)
        pred_positive = result.label == "positive"
        true_positive = bool(true_label)
        if result.label != "neutral" and pred_positive == true_positive:
            correct += 1
    latency_ms = (time.time() - t0) / len(texts) * 1000.0

    row = {
        "model": name,
        "accuracy": round(correct / len(texts), 4),
        "latency": round(latency_ms, 2),
        "size": round(_size_mb(backend), 2),
        "load_seconds": round(load_seconds, 2),
    }
    print(
        f"[{name}] acc={row['accuracy']:.4f} lat={row['latency']}ms size={row['size']}MB "
        f"(load {row['load_seconds']}s)"
    )
    return row


def _write_markdown(ranked) -> None:
    lines = [
        "# Sentiment bake-off results",
        "",
        "| Rank | Model | Accuracy | Latency (ms) | Size (MB) | Score |",
        "|------|-------|----------|---------------|-----------|-------|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['accuracy']} | {r['latency']} | "
            f"{r['size']} | **{r['total']}** |"
        )
    lines += [
        "",
        "_Scorecard weights: accuracy 50%, latency 30%, size 20%. Accuracy is on a held-out "
        "IMDB slice (binary ground truth); a 'neutral' prediction always counts as wrong._",
    ]
    (RESULTS_DIR / "bakeoff.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment bake-off evaluation + scorecard.")
    parser.add_argument("--candidates", default=",".join(REGISTRY))
    args = parser.parse_args()
    names = [n for n in args.candidates.split(",") if n]

    _, _, test_texts, test_labels = load_imdb_text()
    print(f"evaluating {len(names)} candidate(s) on {len(test_texts)} held-out reviews\n")

    rows = [evaluate_candidate(name, REGISTRY[name], test_texts, test_labels) for name in names]
    ranked = scorecard_mod.score(rows, weights=SENTIMENT_SCORECARD_WEIGHTS)

    (RESULTS_DIR / "bakeoff.json").write_text(json.dumps(ranked, indent=2))
    _write_markdown(ranked)
    win = ranked[0]
    (RESULTS_DIR / "winner.json").write_text(json.dumps(win, indent=2))
    print(f"\nWINNER: {win['model']}  (total score {win['total']})")


if __name__ == "__main__":
    main()
