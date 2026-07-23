"""MobileNetV2 transfer-learning candidate.

Lightweight, mobile-friendly. The base is frozen initially; train.py unfreezes the top
layers for fine-tuning at a lower LR (same policy for every pre-trained candidate).

Input contract: [0,1] float. We rescale to [-1,1] inside the model via the official
preprocess_input so callers never worry about it.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_mobilenetv2(num_classes: int, input_shape) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="input")   # (224,224,3) in [0,1]
    x = layers.Rescaling(255.0)(inputs)          # [0,1] -> [0,255]
    x = preprocess_input(x)                        # -> [-1,1] as MobileNetV2 expects (its exact training convention)
    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)  # ImageNet backbone, no top
    base.trainable = False                         # frozen now; unfrozen later in the fine-tune phase
    y = layers.GlobalAveragePooling2D(name="gap")(base.output)  # feature maps -> vector
    y = layers.Dropout(0.3)(y)                     # regularise
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)  # 29-class head
    model = models.Model(inputs, outputs, name="mobilenetv2")
    # Record the backbone layer NAMES (plain strings) for train.py's fine-tune phase.
    # We must NOT store `base` itself as an attribute: its layers are already inlined
    # into this model via input_tensor, so tracking it again double-registers those
    # layers and corrupts .keras save/load ("expected 1 variables, received 0").
    model.base_layer_names = [layer.name for layer in base.layers]
    return model
