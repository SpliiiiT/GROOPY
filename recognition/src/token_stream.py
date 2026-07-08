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
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.contract import (  # noqa: E402  re-exported for backward compatibility
    CONTRACT,
    CONTRACT_VERSION,
    CONTROL_TOKENS,
    KIND_CONTROL,
    KIND_LETTER,
    Sentiment,
    Token,
)

from .config import CONFIDENCE_GATE, DEBOUNCE_MS  # noqa: E402

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
        gate: float = CONFIDENCE_GATE,
        debounce_ms: int = DEBOUNCE_MS,
        kind: str = KIND_LETTER,
    ) -> None:
        self.gate = gate
        self.debounce_ms = debounce_ms
        self.kind = kind
        self._last_token: Optional[str] = None
        self._last_emit_ms: float = 0.0

    def update(
        self, label: str, confidence: float, sentiment: Optional[Sentiment] = None
    ) -> Optional[Token]:
        now_ms = time.time() * 1000.0
        if confidence < self.gate:
            return None
        if now_ms - self._last_emit_ms < self.debounce_ms:
            return None
        if label == self._last_token and (now_ms - self._last_emit_ms) < self.debounce_ms * 2:
            return None

        self._last_token = label
        self._last_emit_ms = now_ms
        kind = KIND_CONTROL if label in CONTROL_TOKENS else self.kind
        return Token(
            token=label if label in CONTROL_TOKENS else label.lower(),
            confidence=round(float(confidence), 3),
            timestamp=int(now_ms),
            kind=kind,
            sentiment=sentiment,
        )
