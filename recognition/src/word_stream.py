"""Live dynamic word recognition — rolling-window buffer -> LSTM -> word Token.

Feed it one landmark vector per camera frame (from holistic.landmarks). It keeps the last
SEQ_LEN frames; once full, each step runs the LSTM and, past the confidence gate, emits a
`kind="word"` Token through the shared TokenStream gate/debounce.

Runs alongside the static fingerspelling CNN in the desktop app: the CNN emits `letter`
Tokens, this emits `word` Tokens, both onto the same contract.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from shared.config import FRAME_FEATURES, SEQ_LEN
from shared.contract import KIND_WORD, Token
from shared.vocabulary import INDEX_TO_GLOSS

from .config import CONFIDENCE_GATE
from .holistic import normalize_sequence
from .token_stream import TokenStream


class WordStream:
    """Rolling landmark buffer + LSTM inference -> word Tokens."""

    def __init__(self, model, seq_len: int = SEQ_LEN, gate: float = CONFIDENCE_GATE) -> None:
        self.model = model
        self.seq_len = seq_len
        self.gate = gate
        self._buf: deque = deque(maxlen=seq_len)
        self._stream = TokenStream(gate=gate, kind=KIND_WORD)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.seq_len

    def push(self, landmark_vec: np.ndarray) -> Optional[Token]:
        """Add one frame's landmarks; return a word Token if one is emitted this step."""
        if landmark_vec.shape[0] != FRAME_FEATURES:
            raise ValueError(f"expected {FRAME_FEATURES} features, got {landmark_vec.shape}")
        self._buf.append(landmark_vec.astype(np.float32))
        if not self.ready:
            return None

        # Same normalization as training (sequence_data) — no train/serve skew.
        seq = normalize_sequence(np.stack(self._buf))
        probs = self.model.predict(np.expand_dims(seq, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        gloss = INDEX_TO_GLOSS[idx]
        return self._stream.update(gloss, float(probs[idx]))

    def predict_sequence(self, seq: np.ndarray) -> tuple[str, float]:
        """One-shot classify a full (seq_len, F) sequence -> (gloss, confidence). For tests/offline.

        Pass a RAW landmark sequence; normalization is applied here to match training.
        """
        seq = normalize_sequence(seq)
        probs = self.model.predict(np.expand_dims(seq, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        return INDEX_TO_GLOSS[idx], float(probs[idx])
