# GROOPY — Two-Way Sign Language Communication

A **bidirectional** communication PoC between a Deaf/HoH signer and a hearing non-signer:

- **Track A · Recognition** (Sign → Text/Speech): the signer **fingerspells** or performs
  **whole-word signs** into the camera → text on screen / spoken aloud.
- **Track B · Synthesis** (Text/Speech → Sign): the hearing person **types or speaks into
  the mic** → the app plays the matching **sign-video clips**, fingerspelling any word
  outside the vocabulary.
- **Sentiment** (shared): analyses the text on either side and attaches a label to the
  contract (its behavioural role is an open design seam).

> **This is a proof of concept, not a replacement for human interpreters.**
> Built with, and credited to, the deaf community (AVST / ATILS).

See [docs/architecture.md](docs/architecture.md) for the full system diagram.

## Status & results

Both recognition tracks are complete — each with a trained, scorecard-picked winner — and both
directions run **live** on the desktop apps.

| Track | Winner (from a 4-model bake-off) | Result |
|-------|----------------------------------|--------|
| **Fingerspelling** (image CNN) | EfficientNetB0 | **≈ 99.9 %** accuracy; exported to a **4.9 MB** TFLite model |
| **Word signs** (sequence model) | Transformer | **~75 %** over 20 words; **21 ms** inference (GRU narrowly tops the scorecard, but Transformer ships — same accuracy within noise, far faster live) |
| **Synthesis** (Text/Speech → Sign) | rule-based + clips | runs live; fingerspells out-of-vocab words |

Highlights: **landmark normalization** took word recognition from chance (5 %) to 75 %; a **data
investigation** (WLASL was 62 % dead → recovered clips + switched to ASL Citizen) tripled the word
data; **Grad-CAM** confirms the models attend to the *hand*. Full write-up in
**[docs/results.md](docs/results.md)**; demo runbook in **[docs/presentation.md](docs/presentation.md)**.

## What this repo does

**Recognition — fingerspelling.** Following **CRISP-DM**, a **model bake-off** for static
ASL fingerspelling: a **CNN built from scratch** competes against pre-trained backbones
(MobileNetV2, EfficientNetB0, ResNet50) under one fixed protocol. The winner — chosen by a
weighted scorecard (accuracy, latency, size, robustness, live stability) — ships in the app.

**Recognition — words.** A dynamic word module — **MediaPipe Holistic landmarks + a
sequence-model bake-off** (LSTM / GRU / BiLSTM / Transformer). GRU narrowly tops the scorecard,
but the **Transformer ships** (~75 % over a 20-word vocabulary, ~7× faster inference — same
accuracy-vs-deployment trade-off as the fingerspelling bake-off). A word sign is a *motion*, so
its candidates are sequence models, not image CNNs.

**Synthesis.** Text (typed or from **ASR**) → rule-based ASL gloss → a sequence of
pre-recorded sign clips, with **fingerspelling fallback** so it never dead-ends.

The **shared vocabulary** (`shared/vocabulary.py`) is the single source of truth: it is both
the LSTM's classes and the Synthesis clip keys. One WLASL download serves both.

Deployment targets: **mobile** (Flutter + TFLite), **web** (TF.js), and **desktop apps**
(PyQt, high-accuracy path for the Project Fair demo).

## Where to run it

| Task | Recommended hardware |
|------|----------------------|
| Full bake-off training | **Google Colab (free T4, 16 GB)** — same batch size for every candidate = fair comparison |
| Dev / debugging | Local RTX 2050 (4 GB) |
| Live desktop demo | Local RTX 2050 (4 GB is plenty for inference) |

The code runs identically on both.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get the data (Kaggle ASL Alphabet — needs a kaggle.json token)
python data/download_asl_alphabet.py

# 3. Explore (CRISP-DM: Data Understanding)
jupyter notebook recognition/notebooks/01_eda.ipynb

# 4. Run the bake-off (trains every candidate, same protocol)
python -m recognition.src.train --model all --epochs 20

# 5. Evaluate + pick the winner
python -m recognition.src.evaluate

# 6. Grad-CAM sanity + bias check
python -m recognition.src.xai_gradcam --model recognition/models/<winner>.keras

# 7. Export the winner (TFLite int8 / TF.js / desktop)
python -m recognition.src.export --model recognition/models/<winner>.keras --target all

# 8. Live desktop demo (fingerspelling; add --word-model for whole-word signs)
python desktop/app.py --model recognition/models/<winner>.keras
```

Run steps 4–5 on Colab (open `recognition/notebooks/02_bakeoff.ipynb`); run steps 6–8 locally.

### Recognition — whole-word signs (dynamic LSTM)

```bash
python data/download_wlasl.py && python data/download_sign_clips.py   # WLASL clips + landmarks
python data/prepare_asl_citizen.py --zip data/ASL_Citizen.zip        # + ASL Citizen (recommended)
python -m recognition.src.train_word --model all --epochs 60 --augment 8   # word bake-off
python -m recognition.src.evaluate_word                              # scorecard -> winner
python desktop/app.py --word-model recognition/models/word_transformer.keras
```

### Track B — Synthesis (Text/Speech → Sign)

```bash
pip install -r synthesis/requirements.txt          # opencv, optional ASR backend
python synthesis/make_stub_clips.py                 # placeholder clips to try it now
python desktop/synthesis_app.py                     # type/speak → sign clips + fingerspell
```

### Smoke-test everything without real data

```bash
python data/make_stub_data.py            # synthetic fingerspelling images (CNN)
python data/make_stub_sequences.py       # synthetic landmark sequences (word LSTM)
python synthesis/make_stub_clips.py      # synthetic sign clips (synthesis)
python tests/test_smoke.py               # headless: contract, gloss, sign-plan, sentiment
```

## Repo layout

```
groopy/
├── shared/                     # single source of truth across both tracks
│   ├── contract.py             #   Token + Sentiment (v2), CONTROL_TOKENS
│   ├── vocabulary.py           #   curated word list = LSTM classes = clip keys
│   └── config.py               #   shared paths + LSTM input geometry
├── data/                       # download + stub-data scripts (data itself gitignored)
├── recognition/                # Track A — Sign → Text/Speech
│   ├── notebooks/              #   01_eda, 02_bakeoff (CRISP-DM)
│   ├── src/                    #   CNN bake-off + dynamic word module (holistic, lstm, word_stream)
│   ├── results/                #   bake-off tables + winner decision
│   └── models/                 #   trained + exported artefacts (gitignored)
├── synthesis/                  # Track B — Text/Speech → Sign
│   ├── src/                    #   asr, text_to_gloss, gloss_to_signplan, player, pipeline
│   └── clips/                  #   sign-video dictionary (gitignored)
├── sentiment/                  # shared, partner-owned — analyze(text) → Sentiment
├── desktop/                    # PyQt apps: app.py (recognition), synthesis_app.py (synthesis)
├── app/                        # Flutter shell (mobile/web) — Dart contract mirror
├── tests/                      # headless smoke tests
└── docs/                       # architecture, data_contract, results, presentation
```

## The data contract (shared with Track B)

Recognition emits, Synthesis consumes, the same object:

```json
{ "token": "hello", "confidence": 0.94, "timestamp": 1730812345678 }
```

See `docs/data_contract.md`. Lock it before parallel work begins.

## Datasets & licences

- **ASL Alphabet** (Kaggle, grassknoted) — 87k images, 29 classes. Core fingerspelling data.
- **ASL Citizen** (Microsoft, NeurIPS 2023) — ~30 videos/sign from everyday, consenting signers.
  Primary word-recognition data.
- **WLASL** — word-level ASL (non-commercial); supplements the word data.
- **TunSL** (warcoder / Mendeley) — 4,423 images, 57 signs, CC BY 4.0. Tunisian pilot only.

Check every licence before shipping. WLASL is non-commercial; ASL Citizen requires its licence for
commercial use.
