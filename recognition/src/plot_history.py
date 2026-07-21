"""Training-curve plots for a bake-off (CRISP-DM: Modeling — visual convergence check).

Reads the per-model history_*.json files written by train.py/train_word.py (each has
{"record": {...}, "history": {"loss": [...], "val_loss": [...], "accuracy": [...],
"val_accuracy": [...]}}) and plots validation accuracy + validation loss per epoch, one
line per model, so you can see which candidate converges faster/higher and which overfits.

Usage:
  python -m recognition.src.plot_history --glob "history_word_*.json" --title "Word bake-off" --out word_training_curves.png
  python -m recognition.src.plot_history --glob "history_*.json" --exclude "history_word_*.json" --title "CNN bake-off" --out cnn_training_curves.png
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

from .config import RESULTS_DIR


def _load_histories(pattern: str, exclude: str | None) -> dict:
    out = {}
    for path in sorted(RESULTS_DIR.glob(pattern)):
        if exclude and fnmatch.fnmatch(path.name, exclude):
            continue
        data = json.loads(path.read_text())
        name = data["record"]["model"]
        out[name] = data["history"]
    return out


def plot_bakeoff_histories(pattern: str, title: str, out_name: str, exclude: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = _load_histories(pattern, exclude)
    if not histories:
        raise SystemExit(f"No history files matched '{pattern}' in {RESULTS_DIR}")

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
    for name, hist in histories.items():
        epochs = range(1, len(hist["val_accuracy"]) + 1)
        ax_acc.plot(epochs, hist["val_accuracy"], marker="o", markersize=3, label=name)
        ax_loss.plot(epochs, hist["val_loss"], marker="o", markersize=3, label=name)

    ax_acc.set_title("Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    ax_loss.set_title("Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path = RESULTS_DIR / out_name
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bake-off training curves from history_*.json.")
    parser.add_argument("--glob", default="history_*.json")
    parser.add_argument("--exclude", default=None, help="fnmatch pattern to skip, e.g. 'history_word_*.json'")
    parser.add_argument("--title", default="Bake-off training curves")
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()
    plot_bakeoff_histories(args.glob, args.title, args.out, args.exclude)


if __name__ == "__main__":
    main()
