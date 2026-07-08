"""The GROOPY data contract — the single object that flows between the two tracks.

Recognition EMITS Tokens (sign -> token -> text); Synthesis CONSUMES the same shape on the
way in (text -> tokens/gloss -> sign). Defining it here, once, means both directions and
both languages (Python + the Dart mirror in app/lib/contract/token.dart) agree by
construction.

v2 (this file) adds an OPTIONAL `sentiment` field. It is additive and backward compatible:
a v1 consumer that ignores unknown fields still works, and `sentiment` defaults to None.
What sentiment DRIVES (a label, signing emphasis, avatar expression) is deliberately left
open — see sentiment/ and synthesis/src/pipeline.apply_sentiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

CONTRACT_VERSION = "v2"

# Control tokens shared by both tracks. NOTE: "del" (not "delete") — matches the ASL
# Alphabet class names and the desktop app. Keep in sync with docs/data_contract.md.
CONTROL_TOKENS = {"del", "nothing", "space"}

# Token.kind values.
KIND_LETTER = "letter"
KIND_WORD = "word"
KIND_CONTROL = "control"


@dataclass
class Sentiment:
    """Sentiment attached to a Token/utterance. Produced by the sentiment/ module.

    label: e.g. "positive" | "neutral" | "negative".
    score: [0,1] confidence/intensity of that label.
    Its behavioural effect on signing is not decided yet — carried as metadata for now.
    """

    label: str
    score: float

    def to_dict(self) -> dict:
        return {"label": self.label, "score": round(float(self.score), 3)}

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Optional[Sentiment]":
        if not d:
            return None
        return Sentiment(label=str(d["label"]), score=float(d["score"]))


@dataclass
class Token:
    """The single object that flows between tracks.

    token: lowercase/normalised — an ASL letter ("a"), a word gloss ("hello"), or a
           control token ("space" | "del" | "nothing").
    confidence: 0..1, only emitted at/above the confidence gate.
    timestamp: Unix epoch milliseconds at time of prediction.
    kind: "letter" | "word" | "control".
    sentiment: optional Sentiment metadata (v2+).
    """

    token: str
    confidence: float
    timestamp: int
    kind: str
    sentiment: Optional[Sentiment] = field(default=None)

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict recurses into the dataclass; normalise sentiment via its own to_dict
        d["sentiment"] = self.sentiment.to_dict() if self.sentiment else None
        d["confidence"] = round(float(self.confidence), 3)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Token":
        return Token(
            token=str(d["token"]),
            confidence=float(d["confidence"]),
            timestamp=int(d["timestamp"]),
            kind=str(d.get("kind", KIND_LETTER)),
            sentiment=Sentiment.from_dict(d.get("sentiment")),
        )


# Lightweight description of the contract, handy for logging / handshakes.
CONTRACT = {"version": CONTRACT_VERSION, "control_tokens": sorted(CONTROL_TOKENS)}
