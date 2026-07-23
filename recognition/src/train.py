"""Unified trainer — the SAME protocol for every bake-off candidate (CRISP-DM: Modeling).

For pre-trained backbones we do the standard two-phase transfer-learning schedule:
  Phase 1  train the new head with the base frozen (LR).
  Phase 2  unfreeze the top of the base and fine-tune (LR / 10).
The from-scratch CNN trains in a single phase (nothing to unfreeze).

Every run logs a per-model history JSON + the saved .keras model into recognition/models
and appends a row to recognition/results/history so evaluate.py can build the bake-off.

Usage:
  python -m recognition.src.train --model all
  python -m recognition.src.train --model cnn_scratch --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

from . import models as model_zoo
from .config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    INPUT_SHAPE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_CLASSES,
    REDUCE_LR_PATIENCE,
    RESULTS_DIR,
    SEED,
)
from .data import make_datasets


def _set_seeds() -> None:
    tf.random.set_seed(SEED)                 # seed TF ops so weight init / shuffles are reproducible
    tf.keras.utils.set_random_seed(SEED)     # seeds python/numpy/tf together


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",          # watch validation accuracy
            patience=EARLY_STOPPING_PATIENCE,# stop after N epochs with no improvement
            restore_best_weights=True,       # roll back to the best epoch (key overfitting guard)
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",              # watch validation loss
            factor=0.3,                      # when it stalls, cut LR to 30%
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,                     # don't go below this
        ),
    ]


def _compile(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),          # Adam optimiser at the given learning rate
        loss="sparse_categorical_crossentropy",          # multi-class loss with integer labels
        metrics=["accuracy"],
    )


def train_one(name: str, epochs: int, train_ds, val_ds) -> dict:
    _set_seeds()                                         # same starting point for every candidate (fairness)
    model = model_zoo.build(name, NUM_CLASSES, INPUT_SHAPE)   # build the requested architecture
    is_pretrained = name in model_zoo.PRETRAINED         # does it have a frozen backbone to fine-tune?

    t0 = time.time()                                     # start timing the whole training

    # Phase 1: train head (frozen base for pretrained; full net for scratch)
    _compile(model, LEARNING_RATE)                       # LR = 1e-3
    hist1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=_callbacks(), verbose=2
    )
    history = {k: [float(v) for v in vs] for k, vs in hist1.history.items()}   # keep the training curves

    # Phase 2: fine-tune the top of a pretrained backbone at a lower LR
    if is_pretrained:
        backbone = _backbone_layers(model)               # resolve the inlined backbone layers by name
        # unfreeze the backbone, then keep only the last ~30 layers trainable to stay
        # stable on a small GPU, and keep BatchNorm layers frozen (standard practice)
        for layer in backbone:
            layer.trainable = True                       # unfreeze everything...
        for layer in backbone[:-30]:
            layer.trainable = False                      # ...then re-freeze all but the last ~30
        for layer in backbone:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False                  # keep BN frozen (moving stats shouldn't shift on small data)
        _compile(model, LEARNING_RATE / 10.0)            # recompile at LR/10 = 1e-4 (gentle fine-tune)
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(5, epochs // 2),                  # fewer epochs for the fine-tune phase
            callbacks=_callbacks(),
            verbose=2,
        )
        for k, vs in hist2.history.items():
            history.setdefault(k, []).extend(float(v) for v in vs)   # append phase-2 curves

    train_seconds = time.time() - t0                     # total wall-clock training time

    model_path = MODELS_DIR / f"{name}.keras"
    model.save(model_path)                               # persist the trained model

    record = {                                           # summary row for the bake-off / reporting
        "model": name,
        "pretrained": is_pretrained,
        "params": int(model.count_params()),             # model size in parameters
        "train_seconds": round(train_seconds, 1),
        "best_val_accuracy": max(history.get("val_accuracy", [0.0])),
        "epochs_run": len(history.get("loss", [])),
        "model_path": str(model_path),
    }
    (RESULTS_DIR / f"history_{name}.json").write_text(   # save curves + record for plots/evaluate
        json.dumps({"record": record, "history": history}, indent=2)
    )
    print(f"[{name}] saved -> {model_path}  best_val_acc={record['best_val_accuracy']:.4f}")
    return record


def _backbone_layers(model: tf.keras.Model) -> list:
    """Return the backbone's layer objects, resolved from the names the builder stored.

    The candidates use `input_tensor=` so the backbone layers are inlined into the
    outer graph (which is what makes Grad-CAM work). The builder records their names on
    `model.base_layer_names` (plain strings — storing the base Model object instead
    would double-track those inlined layers and corrupt .keras save/load). We resolve
    the names back to the shared layer objects here so toggling `trainable` affects the
    outer model after recompilation.
    """
    names = getattr(model, "base_layer_names", None)     # names stashed by the builder
    if not names:
        raise RuntimeError("Model has no .base_layer_names for fine-tuning.")
    return [model.get_layer(n) for n in names]           # name -> actual layer object


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bake-off candidates.")
    parser.add_argument("--model", default="all", help="model name or 'all'")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--data-dir", default=None,
                        help="override the training image root (e.g. a subset for a quick run)")
    args = parser.parse_args()

    names = list(model_zoo.REGISTRY) if args.model == "all" else [args.model]   # which candidates to train
    ds_kwargs = {"data_dir": args.data_dir} if args.data_dir else {}
    train_ds, val_ds, _test_ds, class_names = make_datasets(**ds_kwargs)        # build the shared datasets once
    print("Classes:", class_names)

    summary = [train_one(n, args.epochs, train_ds, val_ds) for n in names]      # train each candidate
    (RESULTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nTraining complete. Run:  python -m recognition.src.evaluate")


if __name__ == "__main__":
    main()
