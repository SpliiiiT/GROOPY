"""Trainer for the dynamic word-sign LSTM (CRISP-DM: Modeling, sequence track).

Mirrors the static trainer's protocol (seeds, EarlyStopping/ReduceLROnPlateau, history
JSON, .keras save) but single-phase — there's no transfer-learning backbone to unfreeze.

Usage:
  python -m recognition.src.train_word --epochs 40
"""
from __future__ import annotations

import argparse
import json
import time

import tensorflow as tf

from shared.config import FRAME_FEATURES, SEQ_LEN
from shared.vocabulary import NUM_WORDS

from .config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MODELS_DIR,
    REDUCE_LR_PATIENCE,
    RESULTS_DIR,
    SEED,
)
from .models.lstm_word import build_lstm_word
from .sequence_data import load_dataset


def _set_seeds() -> None:
    tf.keras.utils.set_random_seed(SEED)


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=REDUCE_LR_PATIENCE, min_lr=1e-6
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dynamic word-sign LSTM.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augment", type=int, default=8,
                        help="augmented copies per train sample (0 = off). Vital for tiny data.")
    args = parser.parse_args()

    _set_seeds()
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), class_names = load_dataset(
        augment_factor=args.augment
    )
    print(f"train={len(X_tr)} (augment x{args.augment}) val={len(X_val)} test={len(X_te)} "
          f"classes={len(class_names)}")

    model = build_lstm_word(NUM_WORDS, SEQ_LEN, FRAME_FEATURES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0 = time.time()
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=_callbacks(),
        verbose=2,
    )
    train_seconds = time.time() - t0

    test_acc = float("nan")
    if len(X_te):
        _, test_acc = model.evaluate(X_te, y_te, verbose=0)

    model_path = MODELS_DIR / "lstm_word.keras"
    model.save(model_path)

    history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    record = {
        "model": "lstm_word",
        "params": int(model.count_params()),
        "train_seconds": round(train_seconds, 1),
        "best_val_accuracy": max(history.get("val_accuracy", [0.0])),
        "test_accuracy": round(float(test_acc), 4),
        "epochs_run": len(history.get("loss", [])),
        "model_path": str(model_path),
    }
    (RESULTS_DIR / "history_lstm_word.json").write_text(
        json.dumps({"record": record, "history": history}, indent=2)
    )
    print(f"[lstm_word] saved -> {model_path}  best_val_acc={record['best_val_accuracy']:.4f} "
          f"test_acc={record['test_accuracy']}")


if __name__ == "__main__":
    main()
