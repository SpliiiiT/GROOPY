# GROOPY — Sign Language Recognition (Track A)

Two-way sign language communication app. This repository holds the **Recognition track**
(Sign → Text/Speech), owned by Oussama, plus the shared integration hooks that connect
to the Synthesis track (Text/Speech → Sign, owned by Mariem).

> **This is a proof of concept, not a replacement for human interpreters.**
> Built with, and credited to, the deaf community (AVST / ATILS).

## What this repo does

Following **CRISP-DM**, we run a **model bake-off** for static ASL fingerspelling: a
**CNN built from scratch** competes against pre-trained backbones (MobileNetV2,
EfficientNetB0, ResNet50) under one fixed protocol. The winning model — chosen by a
weighted scorecard (accuracy, latency, size, robustness, live stability) — ships in the app.
A dynamic word module (MediaPipe Holistic + LSTM) handles a curated 10–30 sign vocabulary.

Deployment targets: **mobile** (Flutter + TFLite), **web** (TF.js), and a **desktop app**
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

# 8. Live desktop demo
python desktop/app.py --model recognition/models/<winner>.keras
```

Run steps 4–5 on Colab (open `recognition/notebooks/02_bakeoff.ipynb`); run steps 6–8 locally.

## Repo layout

```
groopy/
├── data/                       # download scripts (data itself is gitignored)
├── recognition/                # Oussama's track
│   ├── notebooks/              # 01_eda, 02_bakeoff (CRISP-DM)
│   ├── src/                    # preprocess, models/, train, evaluate, scorecard, xai, export
│   ├── results/                # bake-off tables + winner decision
│   └── models/                 # trained + exported artefacts
├── desktop/                    # PyQt high-accuracy webcam app
├── app/                        # shared Flutter shell (mobile/web) — contract stub here
└── docs/                       # data_contract.md, etc.
```

## The data contract (shared with Track B)

Recognition emits, Synthesis consumes, the same object:

```json
{ "token": "hello", "confidence": 0.94, "timestamp": 1730812345678 }
```

See `docs/data_contract.md`. Lock it before parallel work begins.

## Datasets & licences

- **ASL Alphabet** (Kaggle, grassknoted) — 87k images, 29 classes. Core fingerspelling data.
- **WLASL100** — word-level ASL subset for the dynamic module (C-UDA, non-commercial).
- **TunSL** (warcoder / Mendeley) — 4,423 images, 57 signs, CC BY 4.0. Tunisian pilot only.

Check every licence before shipping. WLASL is non-commercial.
