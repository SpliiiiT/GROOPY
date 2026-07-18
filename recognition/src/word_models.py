"""Word-sign model bake-off registry — the sequence-model counterpart to models/ (CNNs).

Fingerspelling is a static image (CNN bake-off in models/). A word sign is a MOTION, so its
candidates are SEQUENCE models over MediaPipe landmark sequences (timesteps, features):

  lstm         2-layer LSTM (our baseline — same architecture as models/lstm_word.py)
  gru          2-layer GRU (lighter recurrent gate, often trains faster)
  bilstm       bidirectional LSTM (reads the sign forwards and backwards)
  transformer  small self-attention encoder (the modern, non-recurrent candidate)

All share the input contract (Masking so zero-padded frames are ignored) and output a softmax
over the shared word vocabulary, so train_word.py can train any of them under one protocol.
"""
from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="lstm")


def build_gru(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="gru")


def build_bilstm(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(48))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="bilstm")


def _transformer_block(x, num_heads: int, key_dim: int, ff_dim: int, dropout: float):
    # Pre-norm attention + FFN with residuals.
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(h, h)
    x = layers.Add()([x, h])
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.Dense(ff_dim, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(x.shape[-1])(h)
    return layers.Add()([x, h])


def build_transformer(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    d_model = 128
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")
    x = layers.Dense(d_model)(inputs)                       # project landmarks -> d_model
    # learned positional embedding (which frame in the sign)
    positions = tf.range(start=0, limit=timesteps, delta=1)
    pos_emb = layers.Embedding(input_dim=timesteps, output_dim=d_model)(positions)
    x = x + pos_emb
    for _ in range(2):
        x = _transformer_block(x, num_heads=4, key_dim=32, ff_dim=128, dropout=0.3)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="transformer")


REGISTRY: Dict[str, Callable[..., tf.keras.Model]] = {
    "lstm": build_lstm,
    "gru": build_gru,
    "bilstm": build_bilstm,
    "transformer": build_transformer,
}


def build(name: str, num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    if name not in REGISTRY:
        raise KeyError(f"Unknown word model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](num_classes=num_classes, timesteps=timesteps, features=features)
