"""Token emission — the Recognition side of the data contract.

Wraps raw model predictions with the contract rules: confidence gate, debounce, and
no-duplicate-within-window. Produces Token objects ready to hand to Track B.

The Token/Sentiment shape now lives in shared/contract.py (single source, shared with the
Synthesis track and the Dart mirror). This module keeps the Recognition-specific *emission*
logic — the stateful gate/debounce — and re-exports Token/CONTROL_TOKENS/CONTRACT_VERSION
so existing importers (e.g. desktop/app.py) keep working unchanged.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

# Make the repo root importable so `shared` resolves whether this is imported as a module
# (python -m recognition.src...) or the app inserts the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../groopy  (two levels above this file)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))            # ensure `import shared...` works from anywhere

from shared.contract import (  # noqa: E402  re-exported for backward compatibility
    CONTRACT,
    CONTRACT_VERSION,
    CONTROL_TOKENS,
    KIND_CONTROL,
    KIND_LETTER,
    Sentiment,
    Token,
)

from .config import CONFIDENCE_GATE, DEBOUNCE_MS  # noqa: E402   # the two live-tuning constants

__all__ = [
    "Token",
    "Sentiment",
    "TokenStream",
    "CONTROL_TOKENS",
    "CONTRACT",
    "CONTRACT_VERSION",
]


class TokenStream:
    """Stateful gate/debounce. Feed it (label, confidence); it returns a Token or None."""

    def __init__(
        self,
        gate: float = CONFIDENCE_GATE,      # minimum confidence to emit (default 0.80)
        debounce_ms: int = DEBOUNCE_MS,     # minimum ms between two emits (default 500)
        kind: str = KIND_LETTER,            # what this stream emits: "letter" (CNN) or "word" (WordStream)
    ) -> None:
        self.gate = gate
        self.debounce_ms = debounce_ms
        self.kind = kind
        self._last_token: Optional[str] = None   # remember the last emitted label (for the no-repeat rule)
        self._last_emit_ms: float = 0.0          # remember when we last emitted (for debounce)

    def update(
        self, label: str, confidence: float, sentiment: Optional[Sentiment] = None
    ) -> Optional[Token]:
        now_ms = time.time() * 1000.0                       # current time in milliseconds
        if confidence < self.gate:                          # GATE ①: too unsure -> emit nothing
            return None
        if now_ms - self._last_emit_ms < self.debounce_ms:  # GATE ②: debounce — fired too recently
            return None
        if label == self._last_token and (now_ms - self._last_emit_ms) < self.debounce_ms * 2:
            return None                                     # GATE ③: same label held steady -> one token, not a stream

        self._last_token = label                            # commit: remember this label...
        self._last_emit_ms = now_ms                         # ...and the time we emitted it
        kind = KIND_CONTROL if label in CONTROL_TOKENS else self.kind   # del/space/nothing -> "control"
        return Token(
            token=label if label in CONTROL_TOKENS else label.lower(),  # controls kept as-is; letters/words lowercased
            confidence=round(float(confidence), 3),         # tidy the confidence for the contract
            timestamp=int(now_ms),                          # epoch ms stamp
            kind=kind,
            sentiment=sentiment,                            # optional sentiment metadata (usually None here)
        )
