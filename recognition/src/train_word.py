"""Word-sign bake-off trainer — the SAME fixed protocol for every sequence-model candidate.

The dynamic counterpart to train.py (the CNN bake-off). Trains LSTM / GRU / BiLSTM /
Transformer on the shared landmark-sequence dataset under one identical protocol (same split,
augmentation, epochs, optimizer, callbacks), so evaluate_word.py can compare them fairly and
a scorecard picks the winner.

Usage:
  python -m recognition.src.train_word --model all --epochs 60 --augment 8
  python -m recognition.src.train_word --model bilstm --epochs 80
"""
from __future__ import annotations

import argparse
import json
import time

import tensorflow as tf

from shared.config import FRAME_FEATURES, SEQ_LEN
from shared.vocabulary import NUM_WORDS

from . import word_models
from .config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MODELS_DIR,
    REDUCE_LR_PATIENCE,
    RESULTS_DIR,
    SEED,
)
from .sequence_data import load_dataset


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
        ),                                                       # stop early, keep the best epoch
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=REDUCE_LR_PATIENCE, min_lr=1e-6
        ),                                                       # drop LR when val loss stalls
    ]


def train_one(name, epochs, batch_size, data) -> dict:
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = data
    tf.keras.utils.set_random_seed(SEED)                         # reproducible init/shuffle for every candidate
    model = word_models.build(name, NUM_WORDS, SEQ_LEN, FRAME_FEATURES)   # build the requested sequence model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),       # Adam @ 1e-3
        loss="sparse_categorical_crossentropy",                  # integer-label multi-class loss
        metrics=["accuracy"],
    )
    t0 = time.time()
    hist = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,   # single-phase training (trained from scratch)
        batch_size=batch_size, callbacks=_callbacks(), verbose=2,
    )
    train_seconds = time.time() - t0

    test_acc = float(model.evaluate(X_te, y_te, verbose=0)[1]) if len(X_te) else float("nan")   # held-out test accuracy
    model_path = MODELS_DIR / f"word_{name}.keras"
    model.save(model_path)                                       # persist the model

    history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}   # training curves
    record = {                                                   # summary row for the report/plots
        "model": name,
        "params": int(model.count_params()),
        "train_seconds": round(train_seconds, 1),
        "best_val_accuracy": round(max(history.get("val_accuracy", [0.0])), 4),
        "test_accuracy": round(test_acc, 4),
        "epochs_run": len(history.get("loss", [])),
        "model_path": str(model_path),
    }
    (RESULTS_DIR / f"history_word_{name}.json").write_text(
        json.dumps({"record": record, "history": history}, indent=2)
    )
    print(f"[{name}] saved -> {model_path}  val={record['best_val_accuracy']} "
          f"test={record['test_accuracy']}  params={record['params']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the word-sign bake-off candidates.")
    parser.add_argument("--model", default="all", help="word model name or 'all'")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augment", type=int, default=8,
                        help="augmented copies per train sample (0 = off). Vital for tiny data.")
    args = parser.parse_args()

    names = list(word_models.REGISTRY) if args.model == "all" else [args.model]   # which candidates
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), class_names = load_dataset(
        augment_factor=args.augment                              # grow the tiny train set ×augment
    )
    print(f"train={len(X_tr)} (augment x{args.augment}) val={len(X_val)} test={len(X_te)} "
          f"classes={len(class_names)}\nCandidates: {names}")
    data = ((X_tr, y_tr), (X_val, y_val), (X_te, y_te))

    summary = [train_one(n, args.epochs, args.batch_size, data) for n in names]   # train each candidate
    (RESULTS_DIR / "word_train_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nTraining complete. Run:  python -m recognition.src.evaluate_word")


if __name__ == "__main__":
    main()
