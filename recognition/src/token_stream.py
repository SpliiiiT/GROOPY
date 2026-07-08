"""Token emission — the Recognition side of the data contract.

Wraps raw model predictions with the contract rules: confidence gate, debounce, and
no-duplicate-within-window. Produces Token dicts ready to hand to Track B.

Reused by the desktop app and (as a reference) by the Flutter recognition module.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

from .config import CONFIDENCE_GATE, CONTRACT_VERSION, DEBOUNCE_MS

CONTROL_TOKENS = {"del", "nothing", "space"}


@dataclass
class Token:
    token: str
    confidence: float
    timestamp: int
    kind: str  # "letter" | "word" | "control"

    def to_dict(self) -> dict:
        return asdict(self)


class TokenStream:
    """Stateful gate/debounce. Feed it (label, confidence); it returns a Token or None."""

    def __init__(
        self,
        gate: float = CONFIDENCE_GATE,
        debounce_ms: int = DEBOUNCE_MS,
        kind: str = "letter",
    ) -> None:
        self.gate = gate
        self.debounce_ms = debounce_ms
        self.kind = kind
        self._last_token: Optional[str] = None
        self._last_emit_ms: float = 0.0

    def update(self, label: str, confidence: float) -> Optional[Token]:
        now_ms = time.time() * 1000.0
        if confidence < self.gate:
            return None
        if now_ms - self._last_emit_ms < self.debounce_ms:
            return None
        if label == self._last_token and (now_ms - self._last_emit_ms) < self.debounce_ms * 2:
            return None

        self._last_token = label
        self._last_emit_ms = now_ms
        kind = "control" if label in CONTROL_TOKENS else self.kind
        return Token(
            token=label if label in CONTROL_TOKENS else label.lower(),
            confidence=round(float(confidence), 3),
            timestamp=int(now_ms),
            kind=kind,
        )


CONTRACT = {"version": CONTRACT_VERSION}
