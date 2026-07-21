"""Central configuration for the GROOPY Recognition track.

Every module imports from here so the bake-off stays fair: the same image size,
splits, batch size, and training protocol apply to every candidate model.
"""
from __future__ import annotations

from pathlib import Path

from shared.paths import app_root

# ----------------------------------------------------------------------------
# Paths (repo-root relative, so it works on Colab, locally, and packaged without edits)
# ----------------------------------------------------------------------------
REPO_ROOT = app_root()
DATA_DIR = REPO_ROOT / "data"
ASL_TRAIN_DIR = DATA_DIR / "asl_alphabet_train"   # created by download_asl_alphabet.py
ASL_TEST_DIR = DATA_DIR / "asl_alphabet_test"
MODELS_DIR = REPO_ROOT / "recognition" / "models"
RESULTS_DIR = REPO_ROOT / "recognition" / "results"

for _d in (MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Classes — ASL Alphabet: A–Z + 3 controls = 29
# ----------------------------------------------------------------------------
CLASS_NAMES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]
NUM_CLASSES = len(CLASS_NAMES)  # 29

# ----------------------------------------------------------------------------
# Image / input
# ----------------------------------------------------------------------------
IMG_SIZE = 224          # 224x224 — standard for the pre-trained backbones
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# ----------------------------------------------------------------------------
# FIXED TRAINING PROTOCOL — identical for every candidate (CRISP-DM fairness)
# ----------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3         # head; backbones fine-tune at LR/10 (see train.py)
VAL_SPLIT = 0.15             # fraction of the training set held out for validation
SEED = 42                    # reproducibility across the whole bake-off
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2

# ----------------------------------------------------------------------------
# Runtime / inference
# ----------------------------------------------------------------------------
CONFIDENCE_GATE = 0.80       # emit a token only at/above this confidence
DEBOUNCE_MS = 500            # at most one prediction per this window
# NOTE: the contract version now lives in shared/contract.py (CONTRACT_VERSION), the single
# source of truth for both tracks. Import it from there, not here.

# ----------------------------------------------------------------------------
# Bake-off scorecard weights (must sum to 1.0). See scorecard.py.
# ----------------------------------------------------------------------------
SCORECARD_WEIGHTS = {
    "accuracy": 0.40,     # test accuracy / macro-F1
    "latency": 0.20,      # ms per frame (lower better)
    "size": 0.15,         # model size in MB (lower better)
    "robustness": 0.15,   # Grad-CAM hand-focus + low bias (manual 0..1 score)
    "stability": 0.10,    # live-webcam stability (manual 0..1 score)
}

# Mobile deployment constraints (desktop path relaxes these)
MAX_LATENCY_MS = 150
MAX_SIZE_MB = 5.0

# ----------------------------------------------------------------------------
# Word-sign bake-off scorecard (sequence models). All criteria are automatic —
# accuracy dominates, with real-time latency + model size as tie-breakers.
# ----------------------------------------------------------------------------
WORD_SCORECARD_WEIGHTS = {
    "accuracy": 0.60,     # macro-F1 on the held-out sequences
    "latency": 0.20,      # ms per sequence (lower better)
    "size": 0.20,         # model size in MB (lower better)
}
