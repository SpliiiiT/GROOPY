# Sentiment — options & decision brief

*For the teammate who owns the sentiment module. This lays out what's already built and the
choices to make, with exact code hooks and effort estimates.*

---

## What's already built ✅

- **`sentiment/src/analyze.py`** — `analyze(text) -> Sentiment(label, score)`. A dependency-free
  **lexicon stub** (positive/negative word lists + simple negation) so the system runs today.
  It has a **pluggable backend**: `set_backend(...)` swaps in a real model behind the same
  interface.
- **`shared/contract.py`** — a `Sentiment(label, score)` dataclass, carried as an **optional field
  on the `Token`** (contract v2). So sentiment already travels through the whole system.
- **`synthesis/src/pipeline.py`** — `synthesize()` already calls `analyze()` and attaches the
  result; **`apply_sentiment(plan, sentiment)` is the seam** — currently a **pass-through no-op**.
- **`desktop/synthesis_app.py`** — already **displays the sentiment label** under the gloss.

So sentiment is *computed, carried, and shown* today. The open question is only: **what should it
DO?**

---

## Two independent decisions

### Decision A — what sentiment *drives* (the behavioural role)

| Option | What it does | Effort | Feasible with our clip-based synthesis? |
|--------|--------------|--------|------------------------------------------|
| **A1 · Label only** | Show an emoji/label ("positive 🙂"); no effect on signing. | **~0** (mostly done) | ✅ Yes |
| **A2 · Emphasis / speed** | Strong sentiment → slower playback + hold/repeat key signs (like tone of voice). | **~1 evening** | ✅ Yes |
| **A3 · Facial expression** | Map sentiment to ASL **non-manual markers** (facial expression). Linguistically the "right" answer. | **Large** | ❌ Needs a 3D **avatar** — we play fixed clips, so not feasible without rebuilding the output |

**Implementation sketch for A2** (the one worth doing):
- Add fields to the plan steps in `synthesis/src/gloss_to_signplan.py` (e.g. `hold_ms`, `repeat`).
- Implement `apply_sentiment(plan, sentiment)` in `synthesis/src/pipeline.py`: for
  `abs(score) high & label != neutral`, increase `hold_ms` / set `repeat=2` on `WordClip` steps.
- Honour those fields in `synthesis/src/player.py` (loop the clip / extend the pause).
- No change needed to `analyze.py` or the contract.

### Decision B — how good the sentiment *model* is

| Option | What it is | Effort |
|--------|-----------|--------|
| **B1 · Keep the lexicon stub** | Fine for a PoC; deterministic, no dependencies. | 0 |
| **B2 · Plug a real model** | e.g. a HuggingFace `transformers` sentiment pipeline, wired via `set_backend()`. Better accuracy, adds a dependency + download. | ~½ day |

`analyze.py` is already structured for B2 — implement a class with a `score(text) -> Sentiment`
method and call `set_backend(YourBackend())`. Nothing else changes.

---

## Recommendation (for a PoC / soutenance)

- **Decision A → A1 now, A2 if time allows.** A1 is essentially done and already demoable. A2 is a
  small, visible "wow" (sentiment changes *how* it signs) and stays within our clip pipeline.
  Skip **A3** (avatar) — out of scope.
- **Decision B → B2 if you want a real ML contribution** (a trained/pretrained sentiment model is a
  nice thing to present); otherwise **B1** is a valid PoC choice — just be explicit that it's a
  rule-based baseline.

Whatever you pick for A, it's a change to **one function** (`apply_sentiment`) — the rest of the
system already carries and displays sentiment, so nothing else needs to move.

---

## Where the hooks are (quick reference)

| To change… | Edit |
|------------|------|
| The sentiment model | `sentiment/src/analyze.py` (add a backend, `set_backend`) |
| What sentiment drives | `synthesis/src/pipeline.py` → `apply_sentiment(plan, sentiment)` |
| Plan step timing/repeat (for A2) | `synthesis/src/gloss_to_signplan.py` + `synthesis/src/player.py` |
| Where the label shows | `desktop/synthesis_app.py` (already done) |
