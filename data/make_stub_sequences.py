"""Generate SYNTHETIC landmark sequences for smoke-testing the dynamic word LSTM.

NOT real motion capture. Writes a handful of .npy sequences per vocabulary gloss into
data/wlasl_landmarks/<gloss>/, each shaped (SEQ_LEN, FRAME_FEATURES). Every gloss gets a
distinct, learnable base pattern plus per-sample noise, so the LSTM pipeline (sequence_data
-> train_word -> word_stream) can be exercised end-to-end in seconds without WLASL.

Usage:
  python data/make_stub_sequences.py                  # 12 sequences/gloss
  python data/make_stub_sequences.py --per-class 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import FRAME_FEATURES, LANDMARKS_DIR, SEQ_LEN  # noqa: E402
from shared.vocabulary import GLOSSES  # noqa: E402


def make_sequence(idx: int, rng: np.random.Generator) -> np.ndarray:
    """A distinct, learnable (SEQ_LEN, FRAME_FEATURES) pattern for class `idx`."""
    t = np.linspace(0, 1, SEQ_LEN).reshape(-1, 1)
    # Class-specific frequency/phase so trajectories differ per gloss.
    freq = 1.0 + (idx % 7)
    phase = (idx * 0.37) % (2 * np.pi)
    base = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t + phase)   # (SEQ_LEN, 1)
    seq = np.tile(base, (1, FRAME_FEATURES)).astype(np.float32)
    # Per-feature offset (deterministic per class) + small per-sample noise.
    feat_offset = (np.arange(FRAME_FEATURES) * (idx + 1) % 17) / 100.0
    seq = seq + feat_offset.astype(np.float32)
    seq = seq + rng.normal(0, 0.02, size=seq.shape).astype(np.float32)
    return np.clip(seq, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic landmark sequences.")
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    total = 0
    for idx, gloss in enumerate(GLOSSES):
        cls_dir = LANDMARKS_DIR / gloss
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            np.save(cls_dir / f"{gloss}_{i:03d}.npy", make_sequence(idx, rng))
            total += 1

    print(f"Wrote {total} stub sequences across {len(GLOSSES)} glosses -> {LANDMARKS_DIR}")
    print("SYNTHETIC landmark sequences for smoke-testing only — not real motion.")


if __name__ == "__main__":
    main()
