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
    # One VGG-style block = two conv layers then a downsample. Two 3x3 convs stacked see a
    # 5x5 receptive field but with fewer parameters than one 5x5 conv (and more non-linearity).
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)  # 3x3 conv; no bias (BN adds its own)
    x = layers.BatchNormalization()(x)                                # normalise activations -> faster, stabler training
    x = layers.Activation("relu")(x)                                  # non-linearity (keeps positives, zeroes negatives)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)  # second conv at the same width
    x = layers.BatchNormalization()(x)                                # BN again
    x = layers.Activation("relu")(x)                                  # ReLU again
    x = layers.MaxPooling2D(2)(x)                                     # halve H,W -> bigger receptive field, less compute
    return x


def build_cnn_scratch(num_classes: int, input_shape) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="input")  # (224,224,3) image tensor in [0,1]
    x = _conv_block(inputs, 32)      # 224 -> 112, learns low-level edges/textures
    x = _conv_block(x, 64)           # 112 -> 56, mid-level shapes
    x = _conv_block(x, 128)          # 56 -> 28, higher-level handshape parts
    x = _conv_block(x, 256)          # 28 -> 14, most abstract features — last conv block = Grad-CAM target
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # average each 14x14 map -> 256-vector (tiny head, clean Grad-CAM)
    x = layers.Dropout(0.4)(x)                          # drop 40% of units -> regularisation vs overfitting
    x = layers.Dense(256, activation="relu")(x)         # fully-connected classifier layer
    x = layers.Dropout(0.3)(x)                          # more dropout before the output
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)  # 29-way class probabilities
    return models.Model(inputs, outputs, name="cnn_scratch")  # wire input->output into a trainable Model
