"""EfficientNetB0 transfer-learning candidate.

Strong accuracy-per-FLOP; a good middle ground on the size/latency/accuracy trade-off.

Note: keras EfficientNet models expect raw [0,255] inputs — preprocessing is built into
the model as a normalization layer. So we rescale [0,1] -> [0,255] and pass straight in.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnetb0(num_classes: int, input_shape) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.Rescaling(255.0)(inputs)            # EfficientNet handles its own norm
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False
    y = layers.GlobalAveragePooling2D(name="gap")(base.output)
    y = layers.Dropout(0.3)(y)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)
    model = models.Model(inputs, outputs, name="efficientnetb0")
    # Record backbone layer NAMES for fine-tuning; storing `base` itself would
    # double-track its inlined layers and corrupt .keras save/load. See mobilenetv2.py.
    model.base_layer_names = [layer.name for layer in base.layers]
    return model
