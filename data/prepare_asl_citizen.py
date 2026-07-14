"""Prepare ASL Citizen word-sign data for the dynamic word LSTM.

ASL Citizen (Microsoft, NeurIPS 2023) is a community-sourced isolated-sign dataset: ~30
everyday-signer videos per sign, which is far more depth than WLASL for our 20 words. This
reads the unzipped dataset and extracts MediaPipe landmark sequences for our vocabulary into
the SAME format the trainer consumes.

Expected layout (after unzipping ASL_Citizen.zip):
    <asl-dir>/
      splits/{train,val,test}.csv     columns: Participant ID, Video file, Gloss, ASL-LEX Code
      videos/<id>-<GLOSS>.mp4

Our glosses map to ASL Citizen's UPPERCASE labels via shared.vocabulary.ASL_CITIZEN_GLOSSES
(variants like WANT1/WANT2 fold into one class). Landmarks go to shared.config.LANDMARKS_DIR
as `aslc_<i>.npy`, so they COMBINE with any WLASL clips already there (distinct prefix, no
collision). ASL Citizen clips are already trimmed to the isolated sign (whole video used).

Usage:
  python data/prepare_asl_citizen.py --asl-dir data/asl_citizen/ASL_Citizen
  python data/prepare_asl_citizen.py --asl-dir <path> --max-per-gloss 40 --fresh
  python -m recognition.src.train_word --epochs 60 --augment 8      # then train
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import LANDMARKS_DIR, SEQ_LEN  # noqa: E402
from shared.vocabulary import ASL_CITIZEN_GLOSSES, GLOSSES  # noqa: E402


def _read_splits(asl_dir: Path) -> list[tuple[str, str]]:
    """Return (video_filename, GLOSS) across all split CSVs. Falls back to scanning videos/."""
    pairs: list[tuple[str, str]] = []
    splits = asl_dir / "splits"
    csvs = list(splits.glob("*.csv")) if splits.is_dir() else []
    if csvs:
        for c in csvs:
            with open(c, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    pairs.append((row["Video file"].strip(), row["Gloss"].strip().upper()))
    else:
        # No CSVs: derive gloss from filename "<id>-<GLOSS>.mp4"
        for v in (asl_dir / "videos").glob("*.mp4"):
            gloss = v.stem.split("-", 1)[-1].strip().upper()
            pairs.append((v.name, gloss))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ASL Citizen landmarks for our vocab.")
    parser.add_argument("--asl-dir", required=True,
                        help="unzipped ASL_Citizen dir (contains splits/ and videos/)")
    parser.add_argument("--max-per-gloss", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--fresh", action="store_true",
                        help="remove existing aslc_*.npy for our glosses first")
    args = parser.parse_args()

    asl_dir = Path(args.asl_dir)
    videos = asl_dir / "videos"
    if not videos.is_dir():
        sys.exit(f"{videos} not found. Point --asl-dir at the unzipped ASL_Citizen folder.")

    from recognition.src.holistic import video_to_sequence  # needs mediapipe

    # ASL Citizen GLOSS (uppercase) -> our gloss
    label_to_word = {lab: word for word, labs in ASL_CITIZEN_GLOSSES.items() for lab in labs}
    pairs = _read_splits(asl_dir)

    # group our-vocab videos by our gloss
    by_word: dict[str, list[str]] = {g: [] for g in GLOSSES}
    for fname, gloss in pairs:
        word = label_to_word.get(gloss)
        if word:
            by_word[word].append(fname)

    n_seqs = 0
    per_word = {}
    for word in GLOSSES:
        out_dir = LANDMARKS_DIR / word
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.fresh:
            for old in out_dir.glob("aslc_*.npy"):
                old.unlink()
        count = 0
        for fname in by_word[word][: args.max_per_gloss]:
            vpath = videos / fname
            if not vpath.is_file():
                continue
            seq = video_to_sequence(str(vpath), args.seq_len)   # whole clip
            if seq is not None:
                np.save(out_dir / f"aslc_{count:03d}.npy", seq)
                count += 1
                n_seqs += 1
        per_word[word] = count

    print(f"ASL Citizen landmark sequences written: {n_seqs} -> {LANDMARKS_DIR}")
    for w in GLOSSES:
        print(f"  {w:8} {per_word[w]}")
    missing = [w for w in GLOSSES if per_word[w] == 0]
    if missing:
        print(f"WARNING: no videos found for {missing} — check ASL_CITIZEN_GLOSSES labels.")
    print("Next: python -m recognition.src.train_word --epochs 60 --augment 8")


if __name__ == "__main__":
    main()
