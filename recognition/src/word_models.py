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
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")  # (30 frames, 258 features)
    x = layers.Masking(mask_value=0.0)(inputs)          # skip zero-padded frames (short signs) in all later layers
    x = layers.LSTM(128, return_sequences=True)(x)      # 1st recurrent layer; keeps per-frame outputs for the next LSTM
    x = layers.Dropout(0.3)(x)                          # regularise
    x = layers.LSTM(64)(x)                              # 2nd recurrent layer; returns only the final summary vector
    x = layers.Dropout(0.3)(x)                          # regularise
    x = layers.Dense(64, activation="relu")(x)          # dense classifier layer
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)  # 20-word probabilities
    return models.Model(inputs, outputs, name="lstm")


def build_gru(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")  # (30, 258)
    x = layers.Masking(mask_value=0.0)(inputs)          # ignore padding frames
    x = layers.GRU(128, return_sequences=True)(x)       # GRU = lighter gated RNN than LSTM (fewer params, often faster)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64)(x)                               # 2nd GRU -> final summary vector
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="gru")


def build_bilstm(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")  # (30, 258)
    x = layers.Masking(mask_value=0.0)(inputs)          # ignore padding frames
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)  # read the sign forwards AND backwards
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(48))(x)        # 2nd bi-directional layer -> summary vector
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="bilstm")


def _transformer_block(x, num_heads: int, key_dim: int, ff_dim: int, dropout: float):
    # Pre-norm attention + FFN with residuals. (Pre-norm = LayerNorm BEFORE each sub-layer -> stabler training.)
    h = layers.LayerNormalization(epsilon=1e-6)(x)      # normalise before attention
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(h, h)  # self-attention (h attends to h)
    x = layers.Add()([x, h])                            # residual connection #1 (keep original + attention output)
    h = layers.LayerNormalization(epsilon=1e-6)(x)      # normalise before the feed-forward net
    h = layers.Dense(ff_dim, activation="relu")(h)      # FFN: expand
    h = layers.Dropout(dropout)(h)                      # regularise
    h = layers.Dense(x.shape[-1])(h)                    # FFN: project back to d_model
    return layers.Add()([x, h])                         # residual connection #2


def build_transformer(num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    d_model = 128                                       # internal width every token is projected to
    inputs = layers.Input(shape=(timesteps, features), name="landmarks")  # (30, 258)
    x = layers.Dense(d_model)(inputs)                       # project 258 raw landmarks -> d_model per frame
    # learned positional embedding (which frame in the sign) — attention alone is order-blind, so we add order info
    positions = tf.range(start=0, limit=timesteps, delta=1)          # [0,1,...,29]
    pos_emb = layers.Embedding(input_dim=timesteps, output_dim=d_model)(positions)  # a learned vector per position
    x = x + pos_emb                                     # inject "which frame" into every token
    for _ in range(2):                                  # stack 2 transformer blocks
        x = _transformer_block(x, num_heads=4, key_dim=32, ff_dim=128, dropout=0.3)
    x = layers.GlobalAveragePooling1D()(x)              # average over the 30 frames -> one clip vector
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)  # 20-word probabilities
    return models.Model(inputs, outputs, name="transformer")


# Registry: name -> builder. train_word.py uses this to train "all" candidates under one protocol.
REGISTRY: Dict[str, Callable[..., tf.keras.Model]] = {
    "lstm": build_lstm,
    "gru": build_gru,
    "bilstm": build_bilstm,
    "transformer": build_transformer,
}


def build(name: str, num_classes: int, timesteps: int, features: int) -> tf.keras.Model:
    if name not in REGISTRY:                            # guard against a typo'd model name
        raise KeyError(f"Unknown word model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](num_classes=num_classes, timesteps=timesteps, features=features)  # dispatch to the builder
