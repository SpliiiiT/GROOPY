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
        self.model = model                               # the trained sequence model (gru/transformer/...)
        self.seq_len = seq_len                           # how many frames make one window (30)
        self.gate = gate                                 # confidence threshold to emit
        self._buf: deque = deque(maxlen=seq_len)         # rolling window: appending past maxlen drops the oldest frame
        self._stream = TokenStream(gate=gate, kind=KIND_WORD)  # reuse the same gate/debounce, tagged as "word"
        # Latest raw prediction (for live UI feedback, even below the emit gate).
        self.last_gloss: Optional[str] = None            # last predicted word (shown live even if not emitted)
        self.last_conf: float = 0.0                      # its confidence

    def reset(self) -> None:
        self._buf.clear()                                # empty the window (e.g. after a committed word)

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.seq_len            # True once we have a full 30-frame window

    def push(self, landmark_vec: np.ndarray) -> Optional[Token]:
        """Add one frame's landmarks; return a word Token if one is emitted this step."""
        if landmark_vec.shape[0] != FRAME_FEATURES:      # safety: every frame must be exactly 258 features
            raise ValueError(f"expected {FRAME_FEATURES} features, got {landmark_vec.shape}")
        self._buf.append(landmark_vec.astype(np.float32))  # slide the newest frame into the window
        if not self.ready:                               # not enough frames yet -> wait
            return None

        # Same normalization as training (sequence_data) — no train/serve skew.
        stacked = np.stack(self._buf)                    # window -> array of shape (30, 258)
        # The LSTM has no "not signing" class, so it always argmaxes SOME word. Suppress the
        # guess when the window has essentially no HAND landmarks (indices 132: = LH+RH) —
        # i.e. the person isn't signing — so idle noise doesn't surface a spurious word.
        hand_activity = float(np.mean(np.any(stacked[:, FRAME_FEATURES - 126:] != 0.0, axis=1)))  # fraction of frames with hands
        if hand_activity < 0.3:                          # GATE: hands basically absent -> not signing
            self.last_gloss, self.last_conf = None, 0.0
            return None

        seq = normalize_sequence(stacked)                # shoulder-centre + scale (identical to training)
        probs = self.model.predict(np.expand_dims(seq, 0), verbose=0)[0]  # (1,30,258) -> 20 class probabilities
        idx = int(np.argmax(probs))                      # index of the most likely word
        gloss = INDEX_TO_GLOSS[idx]                      # map index -> word string
        self.last_gloss, self.last_conf = gloss, float(probs[idx])  # store for live UI feedback
        return self._stream.update(gloss, float(probs[idx]))        # apply gate/debounce -> Token or None

    def predict_sequence(self, seq: np.ndarray) -> tuple[str, float]:
        """One-shot classify a full (seq_len, F) sequence -> (gloss, confidence). For tests/offline.

        Pass a RAW landmark sequence; normalization is applied here to match training.
        """
        seq = normalize_sequence(seq)                    # normalise exactly like training
        probs = self.model.predict(np.expand_dims(seq, 0), verbose=0)[0]  # class probabilities
        idx = int(np.argmax(probs))                      # best class
        return INDEX_TO_GLOSS[idx], float(probs[idx])    # (word, confidence)
