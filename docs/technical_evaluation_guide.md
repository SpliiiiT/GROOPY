# GROOPY — Technical Evaluation Guide (ACT phase)

> **Purpose of this document** — get you ready to defend the project in front of the professor.
> The ACT grade covers **4 axes**:
> 1. **How the code works** — the system end to end
> 2. **Model architectures** — the layers + the hyperparameters
> 3. **Evaluation metrics** — how we measured and compared
> 4. **XAI** — explainability (Grad-CAM and interpretability)
>
> This guide covers the **3 in-house model families** (bake-offs): the **CNN** (fingerspelling),
> the **sequence model** (word signs), and the **sentiment** model.

---

## 0. Big picture — what is GROOPY?

GROOPY is a **bidirectional sign-language communication proof-of-concept**, between a Deaf/HoH
person (the signer) and a hearing non-signer.

Two directions + one shared layer, all in a single repo:

| Track | Folder | Direction | What powers it |
|-------|--------|-----------|----------------|
| **A — Recognition** | `recognition/` | Sign → Text/Speech | CNN (letters) + sequence model (words) |
| **B — Synthesis** | `synthesis/` | Text/Speech → Sign | ASR → gloss → sign plan → clip playback |
| **shared/** | `shared/` | source of truth | `contract.py`, `vocabulary.py`, `config.py` |
| **sentiment/** | `sentiment/` | tone analysis | 3-model bake-off |

**The keyword of the project is "bake-off"**: for each task we don't train ONE model, we put
**several in competition** under an **identical protocol**, then a **weighted scorecard** picks the
winner. That's what shows the professor we understood the trade-off (accuracy / speed / size), not
just that we can call `.fit()`.

**Methodology**: CRISP-DM (Business understanding → Data → Modeling → Evaluation → Deployment).
Every bake-off follows these steps.

---

## 1. HOW THE CODE WORKS (axis 1)

### 1.1 Overall software architecture

```
shared/           ← contract + vocabulary + paths (imported by EVERYONE)
  contract.py       Token(text, kind, confidence, sentiment)  — the exchange format (v2)
  vocabulary.py     the 20 curated words = word-model classes = clip keys
  config.py         SEQ_LEN=30, FRAME_FEATURES=258, repo-relative paths

recognition/
  src/config.py     FIXED training protocol (same hyperparameters for all candidates)
  src/models/       CNN candidates: cnn_scratch, efficientnet, mobilenetv2, resnet50
  src/word_models.py sequence candidates: lstm, gru, bilstm, transformer
  src/train.py      unified CNN trainer (identical protocol for each candidate)
  src/train_word.py unified sequence trainer
  src/evaluate*.py  computes the bake-off metrics
  src/scorecard.py  normalizes metrics + weights them → picks the winner
  src/xai_gradcam.py Grad-CAM (explainability)
  src/holistic.py   MediaPipe Holistic + landmark normalization
  src/word_stream.py live (webcam) inference for words

synthesis/
  src/pipeline.py   asr → text_to_gloss → gloss_to_signplan → player
sentiment/
  src/models.py     3 backends (scratch / distilbert / twitter_roberta)
desktop/            demo apps (fingerspelling + words + synthesis) + packaged launcher
```

### 1.2 The Recognition pipeline (Sign → Text), live

**Fingerspelling (letters), `desktop/app.py`:**
1. The webcam sends a frame → cropped to **224×224**, normalized to `[0,1]`.
2. A **hand-presence gate** checks a hand is visible (otherwise we predict nothing).
3. The CNN outputs a **softmax over 29 classes** (A–Z + `del`, `nothing`, `space`).
4. **Confidence gate** (`CONFIDENCE_GATE = 0.80`): a token is emitted only if the probability ≥ 0.80.
5. **Debounce** (`DEBOUNCE_MS = 500`): at most one prediction per 500 ms window.
6. **Capture-to-commit**: the letter is added to the text only on a deliberate action (no automatic
   "spew") → avoids spamming 30 letters/second.

**Word signs, `word_stream.py`:**
1. **MediaPipe Holistic** extracts landmarks per frame: **pose 33 + left hand 21 + right hand 21**.
2. Each frame → a vector of **258 features** = `pose 33×(x,y,z,visibility) + LH 21×(x,y,z) +
   RH 21×(x,y,z)`.
3. We stack **30 frames** (`SEQ_LEN=30`) → a sequence `(30, 258)`.
4. **Landmark normalization** (`holistic.normalize_sequence`): recenter x,y on the shoulder midpoint
   and divide by shoulder width (pose 11/12). ⚠️ **This is THE decisive fix**: accuracy went from
   ~5% (chance) to 37% then 70%. It is applied **identically** in training and live → no
   "train/serve skew".
5. The sequence model outputs a **softmax over 20 classes**.

### 1.3 The shared layer (the "contract")

`shared/vocabulary.py` defines **20 words** (`hello, thanks, please, yes, no, sorry, help, want,
need, name, you, me, good, bad, happy, sad, eat, drink, friend, love`). This list is
**simultaneously**: (a) the output classes of the word model, and (b) the keys of the synthesis
clips. One place to change → both directions update. Any word **outside this list** is handled by
**fingerspelling** (letter by letter) → the system never dead-ends.

### 1.4 Reproducibility

`SEED = 42` everywhere, same splits, same hyperparameters for every candidate of a bake-off. That's
what makes the comparison **fair**.

---

## 2. MODEL ARCHITECTURES (axis 2: layers + hyperparameters)

### 2.A — Fingerspelling CNN

**Task**: classify a hand image into one of **29 classes** (A–Z + del/nothing/space).
**Data**: Kaggle *ASL Alphabet*, 87,000 images, 3000/class. **Input**: `224×224×3`, `[0,1]`.

#### Candidate 1 — `cnn_scratch` (the "built from scratch" baseline)
> File: [cnn_scratch.py](../recognition/src/models/cnn_scratch.py). This is THE "we actually
> understood CNNs" model — we design every layer.

Repeated convolutional block (VGG-style): `[Conv2D → BatchNorm → ReLU] ×2 → MaxPooling2D`.

| Stage | Layers | Filters |
|-------|--------|---------|
| Block 1 | Conv-BN-ReLU ×2 + MaxPool | 32 |
| Block 2 | Conv-BN-ReLU ×2 + MaxPool | 64 |
| Block 3 | Conv-BN-ReLU ×2 + MaxPool | 128 |
| Block 4 | Conv-BN-ReLU ×2 + MaxPool | 256 ← **Grad-CAM target layer** |
| Head | GlobalAveragePooling2D → Dropout(0.4) → Dense(256, ReLU) → Dropout(0.3) → Dense(29, softmax) |

- **Conv2D**: 3×3 kernels, `padding="same"`, `use_bias=False` (the bias is useless because the
  BatchNorm right after cancels it).
- **BatchNormalization**: stabilizes and speeds up training.
- **GlobalAveragePooling** (instead of Flatten): far fewer parameters → small model (~1.25M params,
  ~15 MB) and **clean Grad-CAM**.
- **Dropout 0.4 / 0.3**: regularization against overfitting.
- **~1,248,381 parameters**.

#### Candidates 2–4 — Transfer learning (ImageNet-pretrained backbones)
> `efficientnet.py`, `mobilenetv2.py`, `resnet50.py`. Same **head** for all:
> `GlobalAveragePooling2D → Dropout(0.3) → Dense(29, softmax)`.

| Model | Backbone frozen at start | Built-in preprocessing | Params | Role in the bake-off |
|-------|--------------------------|------------------------|--------|---------------------|
| **EfficientNetB0** | yes | Rescaling `[0,1]→[0,255]` (internal norm) | ~4.09M | best accuracy/FLOP ratio |
| **MobileNetV2** | yes | `preprocess_input` → `[-1,1]` | ~2.30M | lightweight, mobile-oriented |
| **ResNet50** | yes | `preprocess_input` (caffe, mean subtraction) | ~23.6M | accuracy ceiling, heaviest |

**Key point to be able to explain**: preprocessing is **inside the model** (`Rescaling` /
`preprocess_input` layer), so the caller always passes `[0,1]` without worrying about each
backbone's convention.

#### CNN hyperparameters (FIXED protocol, `recognition/src/config.py`)
| Hyperparameter | Value |
|---|---|
| Image | 224×224×3 |
| Batch size | 32 |
| Epochs | 20 (max) |
| Optimizer | **Adam** |
| Learning rate (head) | **1e-3**; backbones fine-tuned at **LR/10 = 1e-4** |
| Loss | `sparse_categorical_crossentropy` |
| Validation split | 15% |
| EarlyStopping | patience 4 on `val_accuracy`, `restore_best_weights` |
| ReduceLROnPlateau | factor 0.3, patience 2, min_lr 1e-6 |
| Seed | 42 |

**Two-phase fine-tuning (pretrained models only)** — textbook transfer learning:
- **Phase 1**: backbone frozen, train only the new head (LR 1e-3).
- **Phase 2**: unfreeze **the last ~30 layers** of the backbone, keep **BatchNorm frozen** (standard
  practice), and retrain at **LR/10**.
- `cnn_scratch` trains in **a single phase** (nothing to unfreeze).

---

### 2.B — Sequence model for word signs

**Task**: classify a **sequence** `(30 frames, 258 features)` into one of **20 words**.
A signed word is a **motion**, so the candidates are **sequence models** (not CNNs). All share:
`Input(30,258) → Masking(0.0)` (ignores padding frames) → … → `Dense(20, softmax)`.

| Candidate | Core architecture | Params | Idea |
|-----------|-------------------|--------|------|
| **lstm** | LSTM(128, seq) → Drop(0.3) → LSTM(64) → Drop(0.3) → Dense(64) | 253,012 | recurrent baseline |
| **gru** | GRU(128, seq) → Drop → GRU(64) → Drop → Dense(64) | 191,700 | lighter gate, trains faster |
| **bilstm** | Bi-LSTM(96, seq) → Drop → Bi-LSTM(48) → Drop → Dense(64) | 372,692 | reads the sign both ways |
| **transformer** | Dense(128)+pos-embedding → 2× attention block → GAP1D → Dense(64) | 241,876 | attention, non-recurrent, parallelizable |

**The Transformer block** (pre-norm, with residuals):
```
LayerNorm → MultiHeadAttention(4 heads, key_dim=32) → + residual
LayerNorm → Dense(128, ReLU) → Dropout → Dense(d_model) → + residual
```
- **Learned positional embedding** (`Embedding(30, 128)`): tells the model *which frame* it is
  (attention alone is order-invariant).
- `d_model = 128`, 2 stacked blocks, `GlobalAveragePooling1D` before the head.

#### Sequence hyperparameters (`train_word.py`)
| Hyperparameter | Value |
|---|---|
| SEQ_LEN | 30 frames |
| Features/frame | 258 |
| Batch size | 16 |
| Epochs | 60 (max, with EarlyStopping) |
| Optimizer / LR | Adam / 1e-3 |
| Loss | `sparse_categorical_crossentropy` |
| **Augmentation** | ×8 copies per sample (noise/scale/shift) — **vital** given the tiny data (~40/class) |
| Masking | value 0.0 (padded frames ignored) |

**To be able to defend**: why Masking? Because sequences are < 30 frames and are **zero-padded**;
Masking stops the model from learning on empty frames. Why ×8 augmentation? Because the dataset is
**tiny** (791 sequences, ~40/class) — this was the project's real wall, solved by combining
**WLASL + ASL Citizen**.

---

### 2.C — Sentiment model (tone analysis)

**Task**: classify text into `positive / negative / neutral`. 3 candidates:

| Candidate | Architecture | Size | Note |
|-----------|--------------|------|------|
| **scratch** | **TF-IDF + Logistic Regression** (from scratch, on IMDB) | 0.84 MB | trains in ~9s CPU |
| **distilbert** | Pretrained transformer (DistilBERT SST-2, **binary**) | 268 MB | PyTorch CPU |
| **twitter_roberta** | Pretrained transformer (RoBERTa Twitter, **natively 3-class**) | 499 MB | PyTorch CPU |

- The **binary** models (scratch, distilbert) derive "neutral" from a **probability band around 0.5**
  (`NEUTRAL_BAND = 0.15`), since IMDB has no neutral label.
- ⚠️ **Important technical note**: HuggingFace `transformers` **dropped TensorFlow support** in recent
  versions → the 2 pretrained models run on **PyTorch (CPU only)**, isolated in
  `sentiment/requirements.txt` (the rest of the project is TensorFlow/Keras).

---

## 3. EVALUATION METRICS (axis 3)

### 3.1 The principle: weighted scorecard (`scorecard.py`)

We do NOT judge a model on accuracy alone. Each metric is **min-max normalized to [0,1]** across the
candidates ("lower is better" metrics like latency are inverted), then **weighted**. The total picks
the winner.

**CNN weights**: accuracy 40% · latency 20% · size 15% · robustness 15% · stability 10%
*(robustness & stability are manual 0–1 scores: the human check "the model looks at the hand and
stays stable live" is part of the decision).*

**Word weights**: accuracy 60% · latency 20% · size 20% *(100% automatic).*

**Sentiment weights**: accuracy 50% · latency 30% · size 20%.

### 3.2 CNN results (fingerspelling) — "easy" dataset, all at ~99–100%

| Model | Test acc | Params | Latency | Size | Verdict |
|-------|----------|--------|---------|------|---------|
| **EfficientNetB0** | **0.9994** | 4.09M | ~297 ms (slowest) | int8 export 4.94 MB | **🏆 scorecard winner (0.629)** |
| resnet50 | 1.0000 | 23.6M | heavy | largest | accuracy ceiling |
| cnn_scratch | 0.9932 | 1.25M | **~26 ms** | 15 MB | **the real real-time/deployable model** |
| mobilenetv2 | 0.9932 | 2.30M | fast | light | — |

**The story to tell**: EfficientNet wins on accuracy **but is the slowest**. `cnn_scratch` is the
one you'd deploy in real time (26 ms, 15 MB, 99.3%). *"The scorecard encodes the priorities"* — if
we gave latency more weight, the in-house baseline would win. This is exactly the kind of nuance the
professor wants to hear.

### 3.3 Sequence results (words) — 20 classes, chance = 5%

| Rank | Model | Acc | Macro-F1 | Latency | Size | Score |
|------|-------|-----|----------|---------|------|-------|
| 1 | **gru** | 0.7542 | 0.7472 | 142 ms | 2.35 MB | **0.724** 🏆 |
| 2 | transformer | 0.7458 | 0.7494 | **21 ms** | 3.13 MB | 0.711 |
| 3 | bilstm | **0.7797** | 0.7914 | 237 ms | 4.54 MB | 0.600 |
| 4 | lstm | 0.6864 | 0.6487 | 118 ms | 3.08 MB | 0.244 |

**Nuance to know**: the scorecard picks **gru**, but we keep the **transformer** as the default live
model — it's within the noise of gru (74.6% vs 75.4%) and **~7× faster** (21 ms vs 142 ms), which
matters more for the capture-to-commit UX. **bilstm** has the best raw accuracy (77.97%) but is
penalized by its latency/size. We measure **Macro-F1** (not just accuracy) because the classes are
small → macro-F1 weights every class equally.

### 3.4 Sentiment results — THE great "metric trap" example

| Model | IMDB acc | Latency | Size | Raw score | **Realistic eval (20 app-style sentences)** |
|-------|----------|---------|------|-----------|--------------------------------------------|
| distilbert | 0.86 | 40 ms | 268 MB | **0.744** (raw winner) | **0.70** — gets **0/6** neutrals wrong! |
| scratch | 0.768 | 0.6 ms | 0.84 MB | 0.662 | 0.90 |
| twitter_roberta | 0.724 | 81 ms | 499 MB | 0.000 | **1.00** ✅ **← the real right choice** |

**The story (highly valued in the defense)**: the IMDB-accuracy scorecard picks **distilbert**. But
**IMDB has NO neutral examples** → the metric cannot test neutral detection. On a realistic test
(`eval_realistic.py`), distilbert misses **all** neutrals (only ever trained to force pos/neg), while
**twitter_roberta** scores 100%. Conclusion: `RECOMMENDED_MODEL = "twitter_roberta"`. **The wrong
benchmark picks the wrong model** — a real data-science lesson, not just a number.

---

## 4. XAI — EXPLAINABILITY (axis 4)

### 4.1 Grad-CAM on the CNN (`xai_gradcam.py`) — the main XAI artifact

**Grad-CAM** (Gradient-weighted Class Activation Mapping) answers: *"which regions of the image does
the CNN rely on for its prediction?"* We produce a **heatmap** overlaid on the image.

**How it works (be able to explain)**:
1. Take the **last convolutional layer** (for `cnn_scratch`, the 256-filter block) — its activation
   maps still keep spatial resolution.
2. Compute the **gradient of the predicted class** with respect to those activation maps
   (`tf.GradientTape`).
3. **Global-average the gradients** (pooling) → one weight per map = "importance".
4. **Weighted combination** of the maps → ReLU (keep what pushes *toward* the class) → normalize to
   [0,1] → resize to 224×224 and overlay in `jet`.

**What we found (results in `recognition/results/gradcam_*.png`)**:
- **EfficientNetB0** focuses its attention **tightly on the hand** → robust.
- **cnn_scratch** sometimes drifts onto the **background** on hard cases → less robust.
- The **hard pairs M/A, M/E, Q/G** (predicted during EDA) are confirmed → **this closes the CRISP-DM
  loop**: what we hypothesized at exploration is verified by explainability.
- These observations **feed the "robustness" score** of the scorecard (manual 0–1). After re-scoring
  with robustness, EfficientNet wins **even wider** (0.681).

**Why it matters here**: in sign language, a model that "cheats" on the background or skin tone would
be **biased and unfair**. Grad-CAM is therefore a **trust and bias check**, not just a pretty
picture.

### 4.2 Interpretability of the other models

- **Sequence model (words)**: no Grad-CAM (no image), but explainability comes from the **landmark
  normalization** (the model sees an invariant hand/body geometry, not raw pixels) and the **Masking**
  (we know exactly which frames count). You can also inspect the **confusion matrix** to see which
  words get mixed up.
- **Sentiment**: the `scratch` model (TF-IDF + LogReg) is **intrinsically interpretable** — you can
  read the **word weights** (which tokens push toward positive/negative). The **neutral-band**
  analysis (§3.4) is itself a form of explaining the model's behavior.

---

## 5. Cheat sheet — likely professor questions

| Question | Short answer |
|----------|--------------|
| Why a CNN for letters and an RNN/Transformer for words? | A letter is a **static image** (CNN); a word is a **motion over time** (sequence). |
| Why `use_bias=False` in the Convs? | The **BatchNorm** right after cancels any bias → useless parameter. |
| Why GlobalAveragePooling instead of Flatten? | Far fewer params (small model) **and** clean Grad-CAM. |
| Role of Dropout? | Regularization, prevents overfitting (crucial with little data). |
| What unlocked the word model? | The **landmark normalization** (5%→37%→70%), + WLASL/ASL Citizen merge (791 sequences). |
| Why two-phase fine-tuning? | Protect the ImageNet weights: train the head first, then unfreeze **gently** (LR/10, BatchNorm frozen). |
| What is the scorecard? | A **multi-criteria** decision (accuracy + latency + size + robustness) normalized and weighted — not just accuracy. |
| Best example of evaluation rigor? | Sentiment: the IMDB benchmark picks distilbert, but it **misses all neutrals**; twitter_roberta wins on a realistic test. |
| What is Grad-CAM and what is it for here? | A heatmap of the regions driving the prediction → verify the model **looks at the hand** (trust + anti-bias). |
| How do you control overfitting? | EarlyStopping (`restore_best_weights`), Dropout, augmentation, ReduceLROnPlateau, 15% validation split. |

---

## 6. How to run the code (demo)

```bash
# Fingerspelling CNN — full bake-off training
python -m recognition.src.train --model all
python -m recognition.src.evaluate            # metrics + scorecard
python -m recognition.src.xai_gradcam --model recognition/models/cnn_scratch.keras --n 12

# Words (sequence)
python -m recognition.src.train_word --model all --epochs 60 --augment 8
python -m recognition.src.evaluate_word

# Sentiment
python -m sentiment.src.train_scratch
python -m sentiment.src.evaluate
python -m sentiment.src.eval_realistic

# Desktop demos (live webcam)
python desktop/app.py             # fingerspelling
python desktop/synthesis_app.py   # text → signs

# Tests
pytest tests/test_smoke.py
```

**Environment**: local `.venv`, **TensorFlow 2.17.1 / Keras 3.15.0** (pinned in `requirements.txt`).
The pretrained sentiment models run on **PyTorch CPU** (`sentiment/requirements.txt`).
