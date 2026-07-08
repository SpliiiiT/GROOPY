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


def _pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """Force a (T, F) sequence to exactly (seq_len, F) by truncating or zero-padding."""
    t = seq.shape[0]
    if t == seq_len:
        return seq
    if t > seq_len:
        return seq[:seq_len]
    pad = np.zeros((seq_len - t, seq.shape[1]), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def load_dataset(
    landmarks_dir: Path = LANDMARKS_DIR,
    seq_len: int = SEQ_LEN,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = SEED,
) -> tuple:
    """Return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names.

    X arrays are float32 (N, seq_len, FRAME_FEATURES); y are int class indices aligned to
    shared.vocabulary.GLOSSES.
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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), list(GLOSSES)
