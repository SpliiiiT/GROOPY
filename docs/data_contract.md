# Data Contract — Recognition ↔ Synthesis

**Status:** proposed — lock this in Week 3 before parallel work begins.
**Owners:** Oussama (emits) · Mariem (consumes)

The single object that flows between the two tracks. Recognition (Sign → Text) produces it;
Synthesis (Text → Sign) consumes the same shape on the way in.

## The Token object

```json
{
  "token": "hello",
  "confidence": 0.94,
  "timestamp": 1730812345678,
  "kind": "word"
}
```

| Field | Type | Rules |
|-------|------|-------|
| `token` | string | Lowercase, normalised. A single ASL letter (`"a"`) or a word gloss (`"hello"`). No spaces except the literal `"space"` control token. |
| `confidence` | float | `0.0`–`1.0`. Only emitted when ≥ the confidence gate (default `0.80`). |
| `timestamp` | int | Unix epoch **milliseconds** at time of prediction. |
| `kind` | string | `"letter"`, `"word"`, or `"control"` (`space` / `del` / `nothing`). Lets Synthesis pick fingerspelling vs a whole-sign video. |

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

Bump `CONTRACT_VERSION` in both `recognition/src/config.py` and the Flutter
`app/lib/contract/token.dart` together. Current: **v1**.
