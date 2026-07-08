"""Export the winning model to deployment targets (CRISP-DM: Deployment).

  --target tflite   int8-quantised .tflite for mobile (Flutter). Aims for <5 MB.
  --target tfjs     TensorFlow.js bundle for the browser demo.
  --target keras    plain .keras copy for the desktop app (highest accuracy).
  --target all      all of the above.

Usage:
  python -m recognition.src.export --model recognition/models/mobilenetv2.keras --target all
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import INPUT_SHAPE, MODELS_DIR
from .data import make_datasets


def export_tflite(model_path: Path) -> Path:
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # int8 needs a representative dataset drawn from real training images
    train_ds, _, _, _ = make_datasets()

    def rep_data():
        count = 0
        for images, _ in train_ds:
            for i in range(images.shape[0]):
                yield [tf.expand_dims(images[i], 0)]
                count += 1
                if count >= 200:
                    return

    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    out = MODELS_DIR / f"{model_path.stem}_int8.tflite"
    out.write_bytes(tflite_model)
    print(f"TFLite int8 -> {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return out


def export_tfjs(model_path: Path) -> Path:
    try:
        import tensorflowjs as tfjs
    except ImportError:
        raise SystemExit("pip install tensorflowjs to export the web bundle.")
    model = tf.keras.models.load_model(model_path)
    out = MODELS_DIR / f"{model_path.stem}_tfjs"
    tfjs.converters.save_keras_model(model, str(out))
    print(f"TF.js -> {out}")
    return out


def export_keras(model_path: Path) -> Path:
    out = MODELS_DIR / f"{model_path.stem}_desktop.keras"
    shutil.copy(model_path, out)
    print(f"Keras (desktop) -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the winning model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", default="all", choices=["tflite", "tfjs", "keras", "all"])
    args = parser.parse_args()

    model_path = Path(args.model)
    if args.target in ("tflite", "all"):
        export_tflite(model_path)
    if args.target in ("tfjs", "all"):
        try:
            export_tfjs(model_path)
        except SystemExit as e:
            print(e)
    if args.target in ("keras", "all"):
        export_keras(model_path)


if __name__ == "__main__":
    main()
