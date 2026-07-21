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
- **`sentiment/src/models.py`, `train_scratch.py`, `evaluate.py`, `eval_realistic.py`** — a real
  scratch-vs-pretrained bake-off (see Decision B below), opt-in via `set_backend()`.

So sentiment is *computed, carried, and shown* today, with a real model available if you want one.
The open question is only: **what should sentiment DO?** (Decision A, below — still yours to make.)

---

## Two independent decisions

### Decision A — what sentiment *drives* (the behavioural role)

| Option | What it does | Effort | Feasible with our clip-based synthesis? |
|--------|--------------|--------|------------------------------------------|
| **A1 · Label only** | Show an emoji/label ("positive 🙂"); no effect on signing. | **~0** (done) | ✅ Yes |
| **A2 · Emphasis / speed** | Strong sentiment → slower playback + hold/repeat key signs (like tone of voice). | **DONE (2026-07-21)** | ✅ Yes |
| **A3 · Facial expression** | Map sentiment to ASL **non-manual markers** (facial expression). Linguistically the "right" answer. | **Large** | ❌ Needs a 3D **avatar** — we play fixed clips, so not feasible without rebuilding the output |

**A2 is built** — `apply_sentiment(plan, sentiment)` in `synthesis/src/pipeline.py` is no longer a
no-op: confident, non-neutral sentiment (`label != "neutral"` and `score >= 0.75`) sets
`hold_ms=400`/`repeat=2` on the plan's `WordClip` steps only (fingerspelled/out-of-vocabulary
words are untouched — emphasis lands on the actual emotional-content signs, not names/rare
words). `synthesis/src/gloss_to_signplan.py`'s `WordClip`/`Fingerspell` both carry `hold_ms`/
`repeat` fields (default 0/1 — no effect on an unmodified plan); `synthesis/src/player.py`
honours them by replaying the clip/fingerspell sequence and holding an extra pause after.
Verified live in `desktop/synthesis_app.py`: "hello friend I am happy" (score 1.0) visibly
replays and holds each clip; a neutral sentence plays exactly as before. No change needed to
`analyze.py` or the contract — this consumes the `Sentiment` that's already being computed
and carried.

### Decision B — how good the sentiment *model* is — DONE (2026-07-21), a real bake-off

Built a 3-way bake-off, mirroring the CNN/word tracks: one from-scratch candidate vs. two
pretrained ones, scored on a fixed protocol (`sentiment/src/evaluate.py`).

| Candidate | What it is |
|---|---|
| **scratch** | TF-IDF + Logistic Regression, trained from scratch on IMDB (`train_scratch.py`) — no GPU/Colab needed, trains in ~9s on CPU. |
| **distilbert** | `distilbert-base-uncased-finetuned-sst-2-english` — pretrained, general-purpose, binary. |
| **twitter_roberta** | `cardiffnlp/twitter-roberta-base-sentiment-latest` — pretrained, natively 3-class, tuned on short informal text. |

**Bake-off #1 — IMDB accuracy/latency/size scorecard** (weights: accuracy 50%, latency 30%,
size 20%):

| Rank | Model | Accuracy | Latency (ms) | Size (MB) | Score |
|------|-------|----------|---------------|-----------|-------|
| 1 | distilbert | 0.860 | 40.5 | 267.8 | **0.744** |
| 2 | scratch | 0.768 | 0.6 | 0.8 | 0.662 |
| 3 | twitter_roberta | 0.724 | 81.2 | 498.6 | 0.0 |

**This pick turned out to be wrong for actual deployment.** IMDB has zero neutral reviews, so
this metric can't reward or punish neutral-detection quality — and `distilbert` was trained only
on movie-review polarity, so it has no real concept of "neutral" at all. A second check on 20
hand-labeled, app-realistic sentences (greetings, everyday statements, clear emotion, negation —
`sentiment/src/eval_realistic.py`, `sentiment/results/realistic_eval.json`) caught it:

| Model | Overall | Positive | Negative | Neutral |
|---|---|---|---|---|
| **twitter_roberta** | **100%** | 6/6 | 8/8 | **6/6** |
| scratch | 90% | 5/6 | 7/8 | 6/6 |
| distilbert | 70% | 6/6 | 8/8 | **0/6** |

`distilbert` gets **every single neutral sentence wrong** ("hello my name is Oussama" → positive
100%, "the box is on the table" → positive 95%) — it was never trained to hedge, so it forces
everything into pos/neg. `twitter_roberta` is perfect. The 500MB/80ms cost that looks bad in
bake-off #1 doesn't actually matter in practice: sentiment runs once per typed/spoken sentence
(a button click), not per-frame like the vision models, so even 80ms is imperceptible.

**Recommendation: `twitter_roberta`** (`sentiment/src/models.py`'s `RECOMMENDED_MODEL`), not the
raw scorecard winner. Report-worthy finding either way: **the benchmark you pick determines the
winner you get** — IMDB accuracy alone would have shipped a model that fails on roughly a third of
realistic input. Activate it with:
```python
from sentiment.src import analyze, models
analyze.set_backend(models.load_recommended_backend())
```
The default stays the dependency-free `LexiconBackend` unless you opt in — `transformers`+`torch`
(~1GB combined, see `sentiment/requirements.txt`) are optional, not required to run the system.

---

## Recommendation (for a PoC / soutenance)

- **Decision A → done.** A1 (label) and A2 (emphasis/replay on confident sentiment) are both built
  and verified live — sentiment now visibly changes *how* the app signs, not just a label under
  the gloss. Skip **A3** (avatar) — out of scope.
- **Decision B → done.** `twitter_roberta` is built, evaluated two ways, and ready to opt into via
  `load_recommended_backend()` — a real ML contribution with a genuinely interesting CRISP-DM
  story (the first evaluation protocol picked the wrong model; a domain-realistic recheck caught
  it). `B1` (the lexicon stub) remains the safe zero-dependency default.

Whatever you pick for A, it's a change to **one function** (`apply_sentiment`) — the rest of the
system already carries and displays sentiment, so nothing else needs to move.

---

## Where the hooks are (quick reference)

| To change… | Edit |
|------------|------|
| The sentiment model | `sentiment/src/analyze.py` (`set_backend`); candidates live in `sentiment/src/models.py` |
| Opt into the real model | `analyze.set_backend(models.load_recommended_backend())` |
| Re-run the bake-off | `python -m sentiment.src.evaluate` (IMDB), `python -m sentiment.src.eval_realistic` (domain check) |
| What sentiment drives | `synthesis/src/pipeline.py` → `apply_sentiment(plan, sentiment)` |
| Plan step timing/repeat (for A2) | `synthesis/src/gloss_to_signplan.py` + `synthesis/src/player.py` |
| Where the label shows | `desktop/synthesis_app.py` (already done) |
