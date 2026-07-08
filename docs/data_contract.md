# Data Contract — Recognition ↔ Synthesis

**Status:** v2 — bidirectional. Single source of truth: `shared/contract.py` (Python) +
`app/lib/contract/token.dart` (Dart mirror).
**Owners:** Recognition track (emits) · Synthesis track (consumes) · Sentiment (annotates).

The single object that flows through the system. Both directions pivot on it:

- **Recognition (Sign → Text/Speech)** *produces* Tokens: a fingerspelled `letter`, a
  whole-sign `word`, or a `control`.
- **Synthesis (Text/Speech → Sign)** *consumes* the same shape: text/ASR → glosses → Tokens
  → sign-video clips (with fingerspelling fallback for out-of-vocabulary words).

## The Token object

```json
{
  "token": "hello",
  "confidence": 0.94,
  "timestamp": 1730812345678,
  "kind": "word",
  "sentiment": { "label": "positive", "score": 0.88 }
}
```

| Field | Type | Rules |
|-------|------|-------|
| `token` | string | Lowercase, normalised. A single ASL letter (`"a"`) or a word gloss (`"hello"`). No spaces except the literal `"space"` control token. |
| `confidence` | float | `0.0`–`1.0`. Only emitted when ≥ the confidence gate (default `0.80`). |
| `timestamp` | int | Unix epoch **milliseconds** at time of prediction. |
| `kind` | string | `"letter"`, `"word"`, or `"control"` (`space` / `del` / `nothing`). Lets Synthesis pick fingerspelling vs a whole-sign video. |
| `sentiment` | object \| null | **(v2, optional)** `{ "label": "positive"\|"neutral"\|"negative", "score": 0.0–1.0 }`. Produced by the sentiment module on the underlying text. Additive and backward compatible — omit or `null` when not analysed. **What it drives (a label, signing emphasis, or expression) is not yet decided**, so consumers should treat it as metadata for now. |

## Shared vocabulary

- **Letters:** `a`–`z` plus controls `space`, `del`, `nothing` (matches the ASL Alphabet 29 classes).
- **Words:** the curated 10–30 gloss list for the dynamic module. **This list is the contract** —
  every word Recognition can emit must have a matching entry in Synthesis's video dictionary.
- **Fallback:** if Synthesis receives a `word` token it has no video for, it falls back to
  fingerspelling the letters. If Recognition is unsure, it emits nothing (below the gate).

## Emission rules (Recognition side)

- Debounce: at most **one token per 500 ms**.
- Gate: only emit when `confidence ≥ 0.80`.
- Never emit two identical consecutive tokens within the debounce window.

## Versioning

Bump the version in lockstep across `shared/contract.py` (`CONTRACT_VERSION`) and the
Flutter `app/lib/contract/token.dart` (`kContractVersion`). Current: **v2** (added optional
`sentiment`; additive/backward-compatible with v1).
