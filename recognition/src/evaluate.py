"""Bake-off evaluation (CRISP-DM: Evaluation).

For every trained model in recognition/models it computes on the held-out TEST set:
  - accuracy, macro precision/recall/F1
  - confusion matrix (saved as PNG)
  - mean inference latency (ms/frame, single-image, CPU by default)
  - model size on disk (MB)

Then it applies the weighted scorecard and writes a ranked bake-off table + a
winner.json. Robustness is a manual [0,1] score from the Grad-CAM/bias review (see
xai_gradcam.py) — ROBUSTNESS_SCORES below holds the values already reported in
docs/results.md sec. 6; any model not in that dict falls back to the 0.5 "unreviewed"
placeholder. Stability needs a live-webcam pass per model and is still unfilled (0.5)
for all candidates — update STABILITY_SCORES the same way once that's done.

Usage:
  python -m recognition.src.evaluate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import scorecard as scorecard_mod
from .config import INPUT_SHAPE, MODELS_DIR, RESULTS_DIR
from .data import make_datasets

# Manual [0,1] scores from the Grad-CAM review (recognition/src/xai_gradcam.py) — see
# docs/results.md sec. 6. efficientnetb0/cnn_scratch are read directly off the heatmaps;
# resnet50/mobilenetv2 are estimated (same rank, not separately Grad-CAM'd). A model not
# listed here falls back to the 0.5 "unreviewed" placeholder rather than guessing.
ROBUSTNESS_SCORES = {
    "efficientnetb0": 0.85,
    "cnn_scratch": 0.65,
    "resnet50": 0.80,
    "mobilenetv2": 0.70,
}
# Manual [0,1] scores from a live-webcam pass — none run yet across all 4 CNN candidates,
# so every model still falls back to the 0.5 placeholder until that review happens.
STABILITY_SCORES: dict = {}


def _size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def _latency_ms(model: tf.keras.Model, n: int = 50) -> float:
    # Use direct __call__ (not model.predict) to measure true per-frame inference
    # time — predict() adds Python/dispatch overhead that would inflate this metric,
    # which feeds 20% of the scorecard.
    dummy = tf.constant(np.random.rand(1, *INPUT_SHAPE).astype("float32"))
    model(dummy, training=False)  # warmup (builds the graph)
    t0 = time.time()
    for _ in range(n):
        model(dummy, training=False)
    return (time.time() - t0) / n * 1000.0


def _plot_confusion(cm, class_names, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_model(name: str, model_path: Path, test_ds, class_names) -> dict:
    model = tf.keras.models.load_model(model_path)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    # Pin labels to the full class set so every metric/matrix spans all 29 classes,
    # even when a small test split or the model's predictions miss some classes.
    # (Without this, sklearn infers labels only from values present in y_true/y_pred,
    # which crashes classification_report and misaligns the confusion-matrix axes.)
    all_labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    _plot_confusion(cm, class_names, RESULTS_DIR / f"confusion_{name}.png")

    report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    (RESULTS_DIR / f"report_{name}.json").write_text(json.dumps(report, indent=2))

    row = {
        "model": name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "latency": round(_latency_ms(model), 2),   # ms/frame
        "size": round(_size_mb(model_path), 2),    # MB
        # manual criteria — see ROBUSTNESS_SCORES/STABILITY_SCORES above; 0.5 = unreviewed
        "robustness": ROBUSTNESS_SCORES.get(name, 0.5),
        "stability": STABILITY_SCORES.get(name, 0.5),
    }
    print(
        f"[{name}] acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} "
        f"lat={row['latency']}ms size={row['size']}MB"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake-off evaluation + scorecard.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    _, _, test_ds, class_names = make_datasets()

    rows = []
    for model_path in sorted(Path(args.models_dir).glob("*.keras")):
        rows.append(evaluate_model(model_path.stem, model_path, test_ds, class_names))

    if not rows:
        print("No .keras models found. Train first: python -m recognition.src.train")
        return

    ranked = scorecard_mod.score(rows)
    (RESULTS_DIR / "bakeoff.json").write_text(json.dumps(ranked, indent=2))
    _write_markdown(ranked)

    win = ranked[0]
    (RESULTS_DIR / "winner.json").write_text(json.dumps(win, indent=2))
    print(f"\nWINNER: {win['model']}  (total score {win['total']})")
    unreviewed_robustness = [r["model"] for r in rows if r["model"] not in ROBUSTNESS_SCORES]
    if unreviewed_robustness:
        print(f"Note: robustness still 0.5 (unreviewed) for: {', '.join(unreviewed_robustness)}.")
    print("Note: stability is still a 0.5 placeholder for every model — needs a live-webcam pass.")


def _write_markdown(ranked) -> None:
    lines = [
        "# Bake-off results",
        "",
        "| Rank | Model | Acc | F1 | Latency (ms) | Size (MB) | Score |",
        "|------|-------|-----|----|--------------|-----------|-------|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['accuracy']} | {r['macro_f1']} | "
            f"{r['latency']} | {r['size']} | **{r['total']}** |"
        )
    lines += [
        "",
        "_Scorecard weights: accuracy 40%, latency 20%, size 15%, robustness 15%, "
        "stability 10%. Robustness is a manual [0,1] score from the Grad-CAM review "
        "(0.5 = not yet reviewed); stability is a manual [0,1] score from a live-webcam "
        "pass (0.5 = not yet run for any candidate)._",
    ]
    (RESULTS_DIR / "bakeoff.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
