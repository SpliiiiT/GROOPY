"""Shared paths + constants used by BOTH tracks.

Kept separate from recognition/src/config.py (which holds the CNN bake-off training
protocol) so the synthesis track and tooling can import shared locations without pulling
in training-specific settings. Repo-root relative, so it works on Colab and locally.
"""
from __future__ import annotations

from pathlib import Path

from .paths import app_root

REPO_ROOT = app_root()

DATA_DIR = REPO_ROOT / "data"
# Fingerspelling fallback assets: one curated representative image per letter (A-Z), committed
# to git — synthesis only ever displays one image per letter (see player.py's _first_image),
# so it doesn't need the full 1.3GB/87k-image ASL Alphabet training set Recognition trains on.
# Also makes this bundleable into a packaged app. Regenerate via a one-off copy of the first
# sorted image per letter from data/asl_alphabet_train if this set ever needs refreshing.
LETTERS_DIR = REPO_ROOT / "synthesis" / "assets" / "letters"

# Synthesis word-sign clip dictionary (gitignored; populated by data/download_sign_clips.py
# or synthesis/make_stub_clips.py).
CLIPS_DIR = REPO_ROOT / "synthesis" / "clips"

# WLASL word-level videos (serve LSTM training + synthesis clips).
WLASL_DIR = DATA_DIR / "wlasl"
# Pre-extracted landmark sequences for the dynamic-word LSTM: <gloss>/<sample>.npy,
# each array shaped (SEQ_LEN, FRAME_FEATURES). Built from WLASL by sequence_data.py, or
# synthesised by data/make_stub_sequences.py.
LANDMARKS_DIR = DATA_DIR / "wlasl_landmarks"

# Dynamic-word LSTM input geometry (shared by holistic.py, sequence_data.py, the model).
SEQ_LEN = 30            # frames per sampled sign window (pad/truncate to this)
# MediaPipe Holistic landmark counts -> per-frame feature vector length:
#   pose 33*4 (x,y,z,visibility) + left hand 21*3 + right hand 21*3 = 258
POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3
FRAME_FEATURES = POSE_FEATURES + HAND_FEATURES * 2   # 258

# Reproducibility for the sequence split (mirrors recognition SEED).
SEED = 42
