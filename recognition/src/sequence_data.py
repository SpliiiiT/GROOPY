"""Landmark-sequence dataset for the dynamic word module.

Loads pre-extracted landmark sequences (one .npy per sample, shaped (SEQ_LEN,
FRAME_FEATURES)) laid out by class:

    data/wlasl_landmarks/<gloss>/<sample>.npy

Classes come from shared.vocabulary.GLOSSES (the single source shared with Synthesis), so
the LSTM's outputs line up 1:1 with the clip dictionary. Splits are deterministic given
SEED — the same fair-protocol philosophy as recognition/src/data.py for the CNN bake-off.

Extraction from raw WLASL videos is done by holistic.video_to_sequence (see
data/download_sign_clips.py); this module only consumes the .npy cache so it stays light
and testable on synthetic stubs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from shared.config import FRAME_FEATURES, LANDMARKS_DIR, SEED, SEQ_LEN
from shared.vocabulary import GLOSSES, GLOSS_TO_INDEX

from .holistic import normalize_sequence


def _pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """Force a (T, F) sequence to exactly (seq_len, F) by truncating or zero-padding."""
    t = seq.shape[0]
    if t == seq_len:
        return seq
    if t > seq_len:
        return seq[:seq_len]
    pad = np.zeros((seq_len - t, seq.shape[1]), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a plausibly-varied copy of a landmark sequence, for training-time augmentation.

    Combines: a small global scale + translation (signer distance/position), light additive
    noise on the *present* landmarks only (never on zero-padded/missing frames), and temporal
    resampling (signing speed). Handedness is NOT mirrored — flipping would change some signs.
    """
    seq_len, feat = seq.shape
    out = seq.astype(np.float32).copy()

    # global scale + translation (leave exact zeros as zeros so padding stays padding)
    present = out != 0.0
    scale = 1.0 + rng.uniform(-0.08, 0.08)
    shift = rng.uniform(-0.04, 0.04)
    out = np.where(present, out * scale + shift, 0.0)

    # light landmark noise, present entries only
    out = np.where(present, out + rng.normal(0.0, 0.012, size=out.shape), 0.0)

    # temporal resample: sample seq_len indices from a random speed-warped timeline
    speed = rng.uniform(0.8, 1.2)
    src = np.clip(np.linspace(0, (seq_len - 1) * speed, seq_len), 0, seq_len - 1).astype(int)
    out = out[src]
    return out.astype(np.float32)


def load_dataset(
    landmarks_dir: Path = LANDMARKS_DIR,
    seq_len: int = SEQ_LEN,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = SEED,
    augment_factor: int = 0,
) -> tuple:
    """Return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names.

    X arrays are float32 (N, seq_len, FRAME_FEATURES); y are int class indices aligned to
    shared.vocabulary.GLOSSES.

    augment_factor: if >0, add this many augmented copies of every TRAIN sample (val/test stay
    clean). Essential when the real per-class sample count is tiny (as with the WLASL mirror).
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    for gloss in GLOSSES:
        cls_dir = landmarks_dir / gloss
        if not cls_dir.is_dir():
            continue
        for npy in sorted(cls_dir.glob("*.npy")):
            seq = np.load(npy).astype(np.float32)
            if seq.ndim != 2 or seq.shape[1] != FRAME_FEATURES:
                raise ValueError(
                    f"{npy}: expected (T, {FRAME_FEATURES}), got {seq.shape}"
                )
            seq = normalize_sequence(seq)          # position/scale invariance
            X.append(_pad_or_truncate(seq, seq_len))
            y.append(GLOSS_TO_INDEX[gloss])

    if not X:
        raise FileNotFoundError(
            f"No landmark sequences under {landmarks_dir}. Run data/make_stub_sequences.py "
            "(smoke test) or data/download_sign_clips.py (real WLASL)."
        )

    X_arr = np.stack(X)
    y_arr = np.array(y, dtype=np.int64)

    # Deterministic shuffle + split.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_arr))
    X_arr, y_arr = X_arr[perm], y_arr[perm]

    n = len(X_arr)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    X_test, y_test = X_arr[:n_test], y_arr[:n_test]
    X_val, y_val = X_arr[n_test : n_test + n_val], y_arr[n_test : n_test + n_val]
    X_train, y_train = X_arr[n_test + n_val :], y_arr[n_test + n_val :]

    if augment_factor > 0 and len(X_train):
        aug_rng = np.random.default_rng(seed + 1)
        aug_X = [X_train]
        aug_y = [y_train]
        for _ in range(augment_factor):
            aug_X.append(np.stack([augment_sequence(s, aug_rng) for s in X_train]))
            aug_y.append(y_train)
        X_train = np.concatenate(aug_X)
        y_train = np.concatenate(aug_y)
        # shuffle the expanded training set
        perm2 = aug_rng.permutation(len(X_train))
        X_train, y_train = X_train[perm2], y_train[perm2]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), list(GLOSSES)
