# GROOPY — Project Report

*A bidirectional sign-language communication proof of concept, built following CRISP-DM.
This report explains what was built, how, and why — for review, not as a replacement for
the interpreter it is inspired by.*

---

## 1. What GROOPY is and why

GROOPY is a **two-way** communication aid between a Deaf/Hard-of-Hearing signer and a hearing
non-signer:

- **Track A — Recognition (Sign → Text/Speech).** The signer either **fingerspells** letters
  (a static hand-shape classifier) or performs **whole-word signs** (a motion classifier). The
  result is shown as text, and optionally spoken aloud.
- **Track B — Synthesis (Text/Speech → Sign).** The hearing person types or speaks a sentence;
  the app plays back the matching **sign-video clips**, and fingerspells any word outside its
  vocabulary so it never dead-ends on an unknown word.

A **shared layer** underpins both directions: one data contract (`Token`/`Sentiment`) and one
20-word vocabulary, which is simultaneously the recognition model's output classes and the
synthesis clip dictionary's keys — a single source of truth rather than two systems that happen
to agree.

This is a proof of concept for a school project, not a production interpreter replacement — the
goal was to demonstrate the *methodology* (compare candidates rigorously, verify claims
empirically, be honest about limitations) as much as the end result. It is built with, and
credited to, the Deaf community (AVST/ATILS).

---

## 2. Methodology: CRISP-DM and the "bake-off"

Every modeling decision in this project follows the same pattern, applied **three separate
times** (fingerspelling, word signs, sentiment):

> Don't assume an architecture. Train several candidates under one identical, fixed protocol,
> score them on a **weighted scorecard** (accuracy, latency, size, and — where relevant —
> robustness/stability), and let the scorecard pick the most *shippable* model — not simply the
> most accurate one.

This matters because the "best" model depends on the deployment target. All three bake-offs in
this project ended up telling some version of the same story: **the model that wins on raw
accuracy is not always the model that should ship**, and the reasons why differ each time (see
§3–§5) — which is itself the strongest evidence that the bake-off methodology was worth doing,
rather than picking one model on instinct.

---

## 3. Recognition, Track A: Fingerspelling (static CNN bake-off)

**Task:** classify one hand image → one of 29 classes (A–Z + del/nothing/space).
**Data:** Kaggle ASL Alphabet, 87,000 images, trained on a balanced 600-images/class subset
(the full set is highly redundant; this keeps Colab training time reasonable without losing
accuracy).
**Candidates:** a CNN built from scratch vs. three ImageNet-pretrained backbones (MobileNetV2,
EfficientNetB0, ResNet50), each fine-tuned in two phases under one fixed protocol.

| Rank | Model | Accuracy | Latency | Size | Score |
|---|---|---|---|---|---|
| 1 | **EfficientNetB0** | 99.94% | 297 ms | 29 MB | **0.629** |
| 2 | ResNet50 | 100.00% | 243 ms | 211 MB | 0.566 |
| 3 | cnn_scratch | 99.32% | **26 ms** | **15 MB** | 0.475 |
| 4 | MobileNetV2 | 99.32% | 163 ms | 22 MB | 0.369 |

All four land at ~99–100% — the dataset is easy, so accuracy barely separates them. EfficientNetB0
wins the scorecard, but it is also the **slowest** model; for a **live, real-time** demo,
`cnn_scratch` is arguably the better deployable choice — 11× faster, smallest, at effectively the
same accuracy. This is the central methodological point: *the scorecard's weights encode
deployment priorities, and different priorities legitimately elect a different winner.*

**Trust check (Grad-CAM):** heatmaps confirmed EfficientNetB0 focuses tightly on the handshape,
while `cnn_scratch` scatters to the background on harder cases — explaining *why* it misfires on
those. Both models confuse the *exact* letter pairs (M/A, M/E, Q/G) predicted during the initial
data-exploration phase, closing the CRISP-DM loop: the hypothesis was confirmed by the model's own
errors, not just asserted.

**Exported:** `efficientnetb0_int8.tflite` (4.94 MB, under a 5 MB mobile budget) plus a desktop
`.keras` copy.

---

## 4. Recognition, Track B: Whole-word signs (sequence-model bake-off)

**Task:** classify a ~1-second motion (30 frames of MediaPipe Holistic landmarks, 258
features/frame) → one of 20 words. A word sign is a *movement*, so — unlike fingerspelling — the
candidates are sequence models, not image CNNs: LSTM, GRU, BiLSTM, and a small Transformer (all
built from scratch; see §4.3 for why no pretrained option exists here).

### 4.1 The data-scarcity investigation

This track's real story is a data problem, solved methodically rather than worked around:

| Stage | Data | Test accuracy |
|---|---|---|
| Baseline LSTM, WLASL mirror | 131 clips (~6.5/class) | 5% (chance) |
| + augmentation only | 131 clips | 5% |
| **+ landmark normalization** | 131 clips | **37%** |
| + recovered source-URL clips | 151 clips | 41% |
| + switch to ASL Citizen dataset | 791 clips (~40/class) | 70% |
| + architecture bake-off (Transformer) | 791 clips | ~75–78% |

Two decisive moves, in order:
1. **Landmark normalization** — recentring each frame on the shoulder midpoint and scaling by
   shoulder width — made signs position- and distance-invariant, lifting accuracy from chance to
   37% on the *same* 131 clips. Without it, the model had to memorise absolute positions from a
   handful of examples per class.
2. **A data diagnosis, not just more scraping:** WLASL (the standard academic word-sign dataset)
   turned out to be small and decaying — 62% of its referenced source videos were dead. Rather
   than accept that ceiling, clips were recovered directly from surviving original sources, and
   the project switched to **ASL Citizen** (Microsoft, NeurIPS 2023), a larger, consent-based,
   everyday-signer dataset, roughly tripling the usable data.

### 4.2 The architecture bake-off

On the combined 791-sequence dataset, all four candidates were retrained once under a Keras
environment migration (see §6) and re-scored:

| Model | Accuracy | Latency | Size |
|---|---|---|---|
| BiLSTM | 78.0% (highest raw accuracy) | 237 ms | 4.5 MB |
| GRU | 75.4% (narrow scorecard winner this run) | 142 ms | 2.4 MB |
| **Transformer (shipped)** | 74.6% | **21 ms** | 3.1 MB |
| LSTM (baseline) | 68.6% | 118 ms | 3.1 MB |

The top three sit within a few accuracy points of each other on a small (118-sample) test set —
within normal retrain-to-retrain noise for data this size, so small scorecard-rank swings between
runs are expected rather than a real capability difference. **The Transformer ships anyway**: it
is the fastest candidate by a wide margin — attention parallelises across the sequence where
recurrent models process it step by step — and at this accuracy level the models are
statistically indistinguishable, so for a **live** capture-to-commit interaction, latency is what
a user actually feels. The same "scorecard vs. deployment" lesson as §3, arrived at independently.

### 4.3 Why no pretrained word-sign model

The fingerspelling track contrasts a scratch model against ImageNet-pretrained backbones. The
word-sign track deliberately does not, because there is no equivalent "ImageNet of sign
language":

- Pretrained **video-input** models (e.g. S3D on Kinetics-400) operate on raw RGB frames, not
  landmark coordinates — adopting one would mean abandoning the MediaPipe-landmark pipeline
  entirely for a much heavier video model.
- Pretrained **skeleton-based** action-recognition models (ST-GCN, PoseC3D) are the closer
  conceptual match, since they also take joint coordinates. They were investigated and rejected
  for a specific, non-obvious reason: they are pretrained on datasets using ~25-joint full-body
  skeletons **with no finger landmarks at all**. Sign-language meaning lives almost entirely in
  hand shape and finger configuration — exactly the information those datasets discard. This is a
  genuine domain gap, not just extra engineering effort: closer to "pretrain on audio, hope it
  transfers to images" than the near-transfer case ImageNet provides for fingerspelling.

---

## 5. Synthesis and Sentiment

### 5.1 The synthesis pipeline

Typed or spoken text (via an optional offline ASR backend) is converted to a rule-based ASL
gloss sequence, then to a **sign plan**: each gloss either plays a pre-recorded clip (if in the
20-word vocabulary) or is fingerspelled letter-by-letter (fallback for names, rare words, or
anything else out of vocabulary) — so the system never dead-ends on unknown input.

### 5.2 Sentiment: a third bake-off, and a benchmark that picked the wrong model

Sentiment analysis runs on the input text and is carried through the whole pipeline. Two
independent questions were resolved:

**How good is the sentiment model?** A third bake-off was built: a from-scratch candidate
(TF-IDF + Logistic Regression, trained on IMDB reviews in ~9 seconds on CPU) against two
pretrained transformer models (a general-purpose one trained on movie reviews, and one natively
trained on informal short text with a genuine neutral class). The first evaluation — accuracy on
a held-out IMDB slice — picked the general-purpose pretrained model. **That pick was wrong for
actual deployment.** IMDB has no neutral reviews, so that metric cannot reward or penalise
neutral-detection quality — and the model it picked, having only ever been trained to force text
into positive/negative, gets **every neutral test sentence wrong** on a second, realistic check
(20 hand-labelled, app-style sentences: greetings, everyday statements, clear emotion, negation).
The model trained on informal text *with* a genuine neutral class scores 100% on that same check.
**The model that was actually shipped is the one that failed the first benchmark** — a concrete
demonstration that the benchmark you choose determines the winner you get, and that a scorecard
number should be checked against realistic input before trusting it.

**What should sentiment behaviourally drive?** Beyond just displaying a label, confident
non-neutral sentiment now adds emphasis to how the sentence is signed: the relevant clips replay
with an extra held pause, similar to tone of voice in speech. Facial-expression modulation (the
linguistically "correct" answer for ASL non-manual markers) remains out of scope — it would
require a 3D avatar, and this project plays fixed video clips.

---

## 6. Engineering rigor: the environment migration

Partway through the project, the fingerspelling models (trained on Google Colab, whose runtime
had moved to a newer version of the underlying ML framework) stopped loading in the local
development environment (which was on an older version) — a live demo blocker. Rather than
patch around it, the upgrade was carried out methodically:

1. Verified the fix in an **isolated, throwaway copy** of the environment first, never touching
   the working one blind.
2. Confirmed the fingerspelling models loaded correctly there — **then explicitly checked whether
   the upgrade would break anything that currently worked** (the word-sign models). It did — the
   exact mirror-image failure. Rather than ignore this, the word models were retrained fresh under
   the new environment and re-verified end to end (explainability checks, the full test suite,
   both live desktop demos) *before* promoting the verified environment into the real one.
3. Fixed the root cause (an unpinned dependency version) so the same mismatch cannot silently
   recur on a future retrain.

A second, related environment issue surfaced later when re-running training on Colab: forcing an
exact version match between the two environments turned out to break compatibility with Colab's
own bundled packages. The fix was to recognise that the original constraint (avoiding a specific
version incompatibility) no longer applied, and to let each environment remain internally
consistent rather than force an unnecessary match — a case where the more cautious-looking
approach (pin everything to be identical) was actually the wrong one.

This is included in this report deliberately: it demonstrates that model results are only useful
if the whole pipeline that produced them is verified and reproducible, and that engineering
correctness was treated with the same rigor as the modeling work.

---

## 7. Live demos and the packaged application

Both directions have been demonstrated **live** on a laptop webcam, and a live-only-when-signing
gate was added to both recognition paths (fingerspelling and word signs each independently
suppress a guess when no hand is actually visible, rather than confidently guessing from
background noise — a real usability bug that was found and fixed for both).

The project is also being packaged into a **single installable desktop application** (rather than
requiring a Python environment to run) — one launcher window offering both directions, bundling
only the models and assets actually needed to run (not the full training datasets), so it can be
handed to someone to try without any setup.

---

## 8. Honest limitations

- This is a **proof of concept**, not a production interpreter.
- Word recognition covers a fixed **20-word vocabulary** at ~75% test accuracy — usable for a
  demo, not production; live single-signer accuracy is somewhat below the test figure (different
  signer/camera conditions than the training data).
- Synthesis uses a **rule-based gloss**, not full ASL grammar.
- Sign output is a **fixed clip dictionary** — coverage is bounded by available clips.
- Facial-expression sentiment (the linguistically complete answer) is out of scope without a 3D
  avatar.
- Mobile/web deployment was not built; the current deployment target is a packaged desktop app.

---

## 9. Conclusion

GROOPY delivers a working, bidirectional sign-language proof of concept built on three rigorous,
independently-run model bake-offs, each surfacing a genuine, non-obvious insight rather than
simply picking a winner: an accuracy leader is not always the deployment choice (fingerspelling,
word signs), and a benchmark's blind spot can pick the wrong model outright (sentiment). The
strongest outcomes are methodological — a fair, evidence-based comparison at every modeling
decision, a data-scarcity problem diagnosed and solved rather than worked around, and an
environment migration carried out with the same verify-before-trusting discipline as the modeling
itself.

*Further detail: `docs/results.md` (full findings write-up), `docs/presentation.md` (demo
runbook), `docs/sentiment_options.md` (sentiment decision brief), `docs/changelog.md` (a
session-by-session engineering log).*
