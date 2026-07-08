"""ResNet50 transfer-learning candidate.

Higher-capacity reference — the best-case accuracy ceiling, and the heaviest/largest.
Useful precisely because it tests whether the extra size buys enough accuracy to justify
itself against the scorecard's latency + size penalties. Likely a desktop-only winner.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


def build_resnet50(num_classes: int, input_shape) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.Rescaling(255.0)(inputs)            # [0,1] -> [0,255]
    x = preprocess_input(x)                        # caffe-style mean subtraction
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False
    y = layers.GlobalAveragePooling2D(name="gap")(base.output)
    y = layers.Dropout(0.3)(y)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)
    model = models.Model(inputs, outputs, name="resnet50")
    # Record backbone layer NAMES for fine-tuning; storing `base` itself would
    # double-track its inlined layers and corrupt .keras save/load. See mobilenetv2.py.
    model.base_layer_names = [layer.name for layer in base.layers]
    return model
