"""Word-sign bake-off evaluation + scorecard (the dynamic counterpart to evaluate.py).

For every trained word_<name>.keras it computes on the held-out landmark-sequence TEST set:
  - accuracy, macro-F1
  - mean inference latency (ms per sequence, single-sample, CPU)
  - model size on disk (MB)
Then applies the weighted word scorecard (config.WORD_SCORECARD_WEIGHTS) and writes a ranked
table + word_winner.json. Point the desktop app's --word-model at the winner.

Usage:
  python -m recognition.src.evaluate_word
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from . import scorecard as scorecard_mod
from .config import MODELS_DIR, RESULTS_DIR, WORD_SCORECARD_WEIGHTS
from .sequence_data import load_dataset


def _latency_ms(model, sample, n: int = 50) -> float:
    x = tf.constant(np.expand_dims(sample, 0).astype("float32"))   # one (1,30,258) input
    model(x, training=False)  # warmup                              # build the graph once (not timed)
    t0 = time.time()
    for _ in range(n):                                             # time n forward passes
        model(x, training=False)
    return (time.time() - t0) / n * 1000.0                         # average ms per sequence


def evaluate_model(name: str, model_path: Path, X_te, y_te) -> dict:
    model = tf.keras.models.load_model(model_path)                 # load one trained candidate
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)     # predicted class per test sequence
    acc = accuracy_score(y_te, y_pred)                             # overall accuracy
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)  # macro-F1 (each class equal weight)
    row = {
        "model": name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
        "latency": round(_latency_ms(model, X_te[0]), 2),          # ms/sequence
        "size": round(model_path.stat().st_size / 1e6, 2),         # MB on disk
    }
    print(f"[{name}] acc={row['accuracy']} f1={row['macro_f1']} "
          f"lat={row['latency']}ms size={row['size']}MB")
    return row


def main() -> None:
    _, _, (X_te, y_te), class_names = load_dataset(augment_factor=0)   # clean (un-augmented) test set
    if not len(X_te):
        print("No test sequences. Run data/prepare_asl_citizen.py + train_word first.")
        return

    rows = []
    for model_path in sorted(MODELS_DIR.glob("word_*.keras")):     # every trained word model
        rows.append(evaluate_model(model_path.stem.replace("word_", ""), model_path, X_te, y_te))
    if not rows:
        print("No word_*.keras models found. Train first: python -m recognition.src.train_word")
        return

    ranked = scorecard_mod.score(rows, weights=WORD_SCORECARD_WEIGHTS)   # weighted ranking (acc 60 / lat 20 / size 20)
    (RESULTS_DIR / "word_bakeoff.json").write_text(json.dumps(ranked, indent=2))

    lines = [                                                      # build a markdown results table
        "# Word-sign bake-off results",
        "",
        "| Rank | Model | Acc | Macro-F1 | Latency (ms) | Size (MB) | Score |",
        "|------|-------|-----|----------|--------------|-----------|-------|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(f"| {i} | {r['model']} | {r['accuracy']} | {r['macro_f1']} | "
                     f"{r['latency']} | {r['size']} | **{r['total']}** |")
    lines += ["", "_Weights: accuracy 60%, latency 20%, size 20%. Test set = held-out landmark "
              "sequences (no augmentation)._"]
    (RESULTS_DIR / "word_bakeoff.md").write_text("\n".join(lines))

    win = ranked[0]                                               # the winner
    (RESULTS_DIR / "word_winner.json").write_text(json.dumps(win, indent=2))
    print(f"\nWORD WINNER: {win['model']}  (score {win['total']})")
    print(f"Use it live:  python desktop/app.py --word-model {win['model_path'] if 'model_path' in win else MODELS_DIR / ('word_'+win['model']+'.keras')}")


if __name__ == "__main__":
    main()
