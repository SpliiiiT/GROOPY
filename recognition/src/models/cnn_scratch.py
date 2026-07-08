"""From-scratch CNN — the required baseline for the bake-off.

No pre-training. A compact VGG-style stack (Conv-BN-ReLU x2 -> MaxPool, repeated),
global average pooling, and a dense head with dropout. Deliberately small so it also
wins on model size, and simple enough that Grad-CAM explanations are clean.

This is the "did we actually understand CNNs" artefact — we design and own every layer.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def _conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return x


def build_cnn_scratch(num_classes: int, input_shape) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="input")
    x = _conv_block(inputs, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 256)          # last conv block — Grad-CAM target
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="cnn_scratch")
