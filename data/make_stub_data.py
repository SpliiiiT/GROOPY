"""Generate a tiny SYNTHETIC ASL-Alphabet-shaped dataset for smoke-testing.

This is NOT real data. It writes a handful of trivially-separable images per class
into the exact folder layout the real download produces:

    data/asl_alphabet_train/<CLASS>/<CLASS>_<i>.jpg

Each class gets a distinct background colour and its label drawn in the centre, plus
per-image random noise so augmentation has something to do and the classes are
learnable. Use it to confirm the pipeline wires together end-to-end (train ->
evaluate -> export) in seconds on CPU, without the 87k-image Kaggle download.

Usage:
  python data/make_stub_data.py                # 20 images/class
  python data/make_stub_data.py --per-class 8  # fewer, faster
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

# Mirror config.CLASS_NAMES without importing tensorflow-heavy config.
CLASS_NAMES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]

DATA_DIR = Path(__file__).resolve().parent
TRAIN_DIR = DATA_DIR / "asl_alphabet_train"
IMG_SIZE = 224


def _class_colour(idx: int) -> tuple[int, int, int]:
    """A distinct, deterministic background colour per class."""
    r = (idx * 53 + 40) % 256
    g = (idx * 97 + 80) % 256
    b = (idx * 29 + 120) % 256
    return r, g, b


def make_image(label: str, idx: int, rng: random.Random) -> Image.Image:
    base = _class_colour(idx)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base)
    draw = ImageDraw.Draw(img)

    # A few random rectangles so images within a class differ (noise for augment).
    for _ in range(6):
        x0, y0 = rng.randint(0, IMG_SIZE - 1), rng.randint(0, IMG_SIZE - 1)
        x1, y1 = x0 + rng.randint(5, 40), y0 + rng.randint(5, 40)
        jitter = tuple(min(255, max(0, c + rng.randint(-40, 40))) for c in base)
        draw.rectangle([x0, y0, x1, y1], fill=jitter)

    # Draw the label big in the centre (the learnable signal).
    text = label.upper()
    # Pillow's default font is tiny; scale up by drawing then resizing a small tile.
    tile = Image.new("RGB", (48, 48), base)
    tdraw = ImageDraw.Draw(tile)
    tdraw.text((6, 14), text[:4], fill=(255, 255, 255))
    tile = tile.resize((150, 150), Image.NEAREST)
    img.paste(tile, ((IMG_SIZE - 150) // 2, (IMG_SIZE - 150) // 2))
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ASL stub data.")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = TRAIN_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            img = make_image(cls, idx, rng)
            img.save(cls_dir / f"{cls}_{i:03d}.jpg", quality=85)
            total += 1

    print(f"Wrote {total} stub images across {len(CLASS_NAMES)} classes -> {TRAIN_DIR}")
    print("This is SYNTHETIC data for smoke-testing only — not real ASL.")


if __name__ == "__main__":
    main()
