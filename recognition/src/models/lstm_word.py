"""Dynamic word-sign classifier — an LSTM over landmark sequences.

Distinct from the static-image CNN bake-off: input is a (timesteps, features) landmark
sequence from MediaPipe Holistic, output is a softmax over the shared word vocabulary.
Masking lets zero-padded short sequences be ignored by the recurrent layers.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_word(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")
    x = layers.Masking(mask_value=0.0)(inputs)          # ignore zero-padded frames
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="lstm_word")
