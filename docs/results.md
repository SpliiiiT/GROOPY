# GROOPY — Results & Findings

*Two-way sign-language communication: a proof of concept, not a replacement for human
interpreters. Built following CRISP-DM.*

---

## 1. System overview

GROOPY is a **bidirectional** communication aid between a Deaf/HoH signer and a hearing
non-signer, built as two tracks over one shared contract:

- **Track A — Recognition (Sign → Text/Speech).** The signer either **fingerspells** letters
  (a static image classifier) or performs **whole-word signs** (a temporal sequence
  classifier). Output is text on screen, optionally spoken.
- **Track B — Synthesis (Text/Speech → Sign).** The hearing person types or speaks; the app
  plays the matching **sign-video clips**, fingerspelling any word outside the vocabulary.

A **shared layer** (`shared/`) holds the single source of truth: a `Token` data contract and
a curated 20-word vocabulary that is *simultaneously* the word model's output classes and the
synthesis clip dictionary keys.

Both directions were demonstrated **live** on a laptop (webcam + PyQt desktop apps).

---

## 2. Methodology (CRISP-DM)

| Phase | What we did |
|---|---|
| Data Understanding | EDA on the ASL Alphabet (class balance, sample signs, hard letter pairs) |
| Data Preparation | Hand-region focus, resize to 224×224, augmentation (no horizontal flip), landmark extraction for words |
| Modeling | **Two model bake-offs** — 4 candidates each, one fixed protocol |
| Evaluation | Weighted scorecards + **Grad-CAM** explainability/trust check |
| Deployment | Exported winner (TFLite int8), live desktop demos |

The defining methodological choice is the **bake-off**: rather than assume an architecture,
we train several candidates under one identical protocol and let a **weighted scorecard**
pick the most *shippable* model — balancing accuracy against latency, size, and robustness.

---

## 3. Fingerspelling recognition — CNN bake-off

**Task:** classify one hand image → one of 29 classes (A–Z + del/nothing/space).
**Data:** Kaggle ASL Alphabet (87,000 images). Trained on a balanced 600-images/class subset
(the dataset is highly redundant; this keeps training feasible while preserving accuracy).
**Candidates:** a from-scratch CNN vs. three ImageNet-pretrained backbones (2-phase fine-tune).

| Rank | Model | Accuracy | Macro-F1 | Latency | Size | **Score** |
|---|---|---|---|---|---|---|
| **1** | **efficientnetb0** | 0.9994 | 0.9994 | 297 ms | 29 MB | **0.629** |
| 2 | resnet50 | 1.0000 | 1.0000 | 243 ms | 211 MB | 0.566 |
| 3 | cnn_scratch | 0.9932 | 0.9929 | **26 ms** | **15 MB** | 0.475 |
| 4 | mobilenetv2 | 0.9932 | 0.9931 | 163 ms | 22 MB | 0.369 |

*Scorecard weights: accuracy 40 %, latency 20 %, size 15 %, robustness 15 %, stability 10 %.*

### Key finding: two different "winners"
All four models reach **~99–100 %** — the dataset is easy, so accuracy barely separates them.
- **EfficientNetB0 wins the scorecard** on near-top accuracy, *despite being the slowest model*
  (297 ms). Because the accuracies are nearly tied, min-max normalisation amplifies a trivial
  0.7 % accuracy gap.
- But for a **real-time webcam app**, the from-scratch **cnn_scratch is the better deployable
  model**: **11× faster (26 ms)**, **smallest (15 MB)**, at effectively the same accuracy
  (99.3 %).

This is the central insight: **the scorecard's weights encode deployment priorities.** For a
mobile/real-time target you would re-weight toward latency and size, which legitimately elects
`cnn_scratch`. The "best" model depends on the use case, not accuracy alone.

---

## 4. Word-sign recognition — sequence-model bake-off

**Task:** classify a ~1-second **motion** (30 frames of MediaPipe Holistic landmarks, 258
features/frame) → one of 20 words. A word sign is a *movement*, so — unlike fingerspelling —
the candidates are **sequence models**, not image CNNs.
**Data:** 791 landmark sequences (~40/class) from WLASL + ASL Citizen (118-sample test set).

| Rank | Model | Accuracy | Macro-F1 | Latency | Size | **Score** |
|---|---|---|---|---|---|---|
| **1** | **transformer** | **0.780** | 0.778 | **20 ms** | 3.1 MB | **0.93** |
| 2 | gru | 0.771 | 0.773 | 89 ms | 2.4 MB | 0.68 |
| 3 | bilstm | 0.771 | 0.780 | 178 ms | 4.5 MB | 0.44 |
| 4 | lstm (baseline) | 0.703 | 0.690 | 89 ms | 3.1 MB | 0.30 |

*Scorecard weights: accuracy 60 %, latency 20 %, size 20 %.*

### Key finding: the Transformer wins on both accuracy *and* speed
The **self-attention Transformer** is the highest accuracy (78 %) **and** the fastest
(20 ms — ~4× faster than the recurrent models, because attention parallelises whereas
LSTMs/GRUs process frames sequentially). Every newer architecture beat the LSTM baseline.

---

## 5. The data-scarcity investigation (word track)

Word recognition began at **chance (5 %)** and reached **78 %**. The journey is itself a result:

| Stage | Data | Test accuracy |
|---|---|---|
| Baseline LSTM, WLASL mirror | 131 clips (~6.5/class) | 0.05 *(chance)* |
| + data augmentation only | 131 clips | 0.05 |
| **+ landmark normalization** | 131 clips | **0.37** |
| + recovered source-URL clips | 151 clips | 0.41 |
| + ASL Citizen dataset | 791 clips (~40/class) | 0.70 |
| **+ Transformer (bake-off)** | 791 clips | **0.78** |

Two decisive moves:
1. **Landmark normalization** (recentring on the shoulder midpoint, scaling by shoulder width)
   made signs **position- and distance-invariant** — lifting accuracy from chance to 37 % on
   the *same* data. Without it, the model had to memorise absolute positions from ~5 examples.
2. **A data diagnosis**: WLASL is small and decaying (62 % of the referenced videos were dead).
   We recovered clips directly from their original sources, then switched to the larger,
   consent-based **ASL Citizen** dataset (~30 clips/word) — tripling the data.

---

## 6. Explainability & trust — Grad-CAM

Grad-CAM heatmaps verify the fingerspelling models attend to the **hand**, not the background:

- **EfficientNetB0** — tight, consistent focus on the handshape across signs → high robustness.
- **cnn_scratch** — good focus on clean signs (B, A, K) but **scatters to the background** on
  harder cases, which is *why* it misfired on them → moderate robustness.

Plugging the robustness scores from the heatmaps back into the scorecard (EfficientNet 0.85
and cnn_scratch 0.65 are read directly from the heatmaps; resnet50 0.80 and mobilenetv2 0.70
are estimated) re-scores as:

| Rank | Model | Accuracy | Robustness | Total (was) |
|---|---|---|---|---|
| 1 | efficientnetb0 | 0.9994 | 0.85 | **0.681** (0.629) |
| 2 | resnet50 | 1.0000 | 0.80 | 0.611 (0.566) |
| 3 | cnn_scratch | 0.9932 | 0.65 | 0.498 (0.475) |
| 4 | mobilenetv2 | 0.9932 | 0.70 | 0.399 (0.369) |

The **ranking is unchanged** — robustness *reinforces* the accuracy winner and cannot flip it
(the score gaps exceed what the 15 % robustness weight can bridge). To favour the fast,
deployable model instead, one would re-weight the scorecard toward latency and size, not adjust
robustness — the weights, not the robustness scores, encode the deployment priority.

**Closing the CRISP-DM loop:** both models confuse the *exact* hard pairs predicted during EDA
— **M/A, M/E, Q/G**. Our Data-Understanding hypothesis was confirmed by the model errors.

---

## 7. Synthesis (Text/Speech → Sign)

Typed or spoken input → rule-based ASL gloss → a sequence of pre-recorded sign clips, with
automatic **fingerspelling fallback** for out-of-vocabulary words (so it never dead-ends).
Demonstrated live: e.g. *"hello my name is Oussama"* plays the `hello` / `name` clips and
fingerspells `my` and `O-U-S-S-A-M-A`.

---

## 8. Deployment

- **Winner exported:** `efficientnetb0_int8.tflite` (**4.94 MB — under the 5 MB mobile budget**)
  and a desktop `.keras`.
- **Live demos:** `desktop/app.py` (fingerspelling + words together) and
  `desktop/synthesis_app.py` (text/speech → sign).
- Reproducible: two Colab/local notebooks (EDA + bake-off), a headless test suite, and stub-data
  generators so the whole pipeline runs without the real datasets.

---

## 9. Honest limitations

- It is a **proof of concept, not an interpreter**.
- Word recognition covers a **20-word vocabulary** at **78 %** — usable for a demo, not
  production; live single-signer accuracy is lower than the test figure.
- Text→sign uses a **naive gloss** (word lookup), not full ASL grammar.
- Sign output is a **fixed clip dictionary**; coverage is bounded by available clips.
- **Mobile/web deployment is not built** — only the shared Dart contract stub exists.
- CNN accuracies are reported on a 600/class subset (chosen for feasible training time).

---

## 10. Conclusion

GROOPY delivers a working, bidirectional sign-language PoC with **two rigorous CRISP-DM
bake-offs**, explainability, and live demos in both directions. The strongest outcomes are
methodological: a **fair, weighted comparison** that surfaces the accuracy-vs-deployment
trade-off; a **normalization fix** that took word recognition from chance to 78 %; and an
**evidence-based data investigation** that turned a scarcity wall into a working model.
