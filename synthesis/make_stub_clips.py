"""Generate tiny SYNTHETIC sign-video clips for smoke-testing the Synthesis pipeline.

NOT real sign language. Writes a short, labelled, coloured .mp4 per vocabulary gloss into
synthesis/clips/ so the pipeline (and the player) can be exercised end-to-end without the
WLASL download. Each clip just shows the gloss text on a moving coloured background.

Usage:
  python synthesis/make_stub_clips.py                 # ~1s clips for every vocab gloss
  python synthesis/make_stub_clips.py --seconds 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run as a script (python synthesis/make_stub_clips.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import vocabulary as vocab  # noqa: E402
from shared.config import CLIPS_DIR  # noqa: E402

SIZE = 256
FPS = 20


def _colour(idx: int) -> tuple[int, int, int]:
    # BGR for OpenCV
    return (idx * 29 + 120) % 256, (idx * 97 + 80) % 256, (idx * 53 + 40) % 256


def make_clip(gloss: str, idx: int, seconds: float, out_dir: Path) -> Path:
    import cv2  # lazy

    out = out_dir / vocab.CLIP_MAP[gloss]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, FPS, (SIZE, SIZE))
    n_frames = max(1, int(seconds * FPS))
    base = _colour(idx)
    for f in range(n_frames):
        import numpy as np

        frame = np.full((SIZE, SIZE, 3), base, dtype=np.uint8)
        # a moving marker so the clip visibly animates
        x = int((f / max(1, n_frames - 1)) * (SIZE - 40)) + 20
        cv2.circle(frame, (x, SIZE // 2), 14, (255, 255, 255), -1)
        cv2.putText(
            frame, gloss.upper(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sign clips.")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    for idx, gloss in enumerate(vocab.GLOSSES):
        make_clip(gloss, idx, args.seconds, CLIPS_DIR)
    print(f"Wrote {vocab.NUM_WORDS} stub clips -> {CLIPS_DIR}")
    print("These are SYNTHETIC placeholders for smoke-testing only — not real sign language.")


if __name__ == "__main__":
    main()
