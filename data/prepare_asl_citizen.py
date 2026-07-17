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
  # From the zip directly — extracts ONLY our vocab's ~700 videos on the fly (no 46GB unpack):
  python data/prepare_asl_citizen.py --zip data/ASL_Citizen.zip --max-per-gloss 40 --fresh
  # Or from an already-unzipped dataset:
  python data/prepare_asl_citizen.py --asl-dir data/asl_citizen/ASL_Citizen --fresh
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


def _splits_from_csv_text(texts: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for text in texts:
        for row in csv.DictReader(text.splitlines()):
            pairs.append((row["Video file"].strip(), row["Gloss"].strip().upper()))
    return pairs


def _read_splits_dir(asl_dir: Path) -> list[tuple[str, str]]:
    """(video_filename, GLOSS) from splits/*.csv, or from filenames if no CSVs."""
    splits = asl_dir / "splits"
    csvs = list(splits.glob("*.csv")) if splits.is_dir() else []
    if csvs:
        return _splits_from_csv_text([c.read_text(encoding="utf-8") for c in csvs])
    return [(v.name, v.stem.split("-", 1)[-1].strip().upper())
            for v in (asl_dir / "videos").glob("*.mp4")]


def _read_splits_zip(zf) -> list[tuple[str, str]]:
    texts = [zf.read(n).decode("utf-8") for n in zf.namelist() if n.lower().endswith(".csv")]
    return _splits_from_csv_text(texts)


def _group_by_word(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    label_to_word = {lab: word for word, labs in ASL_CITIZEN_GLOSSES.items() for lab in labs}
    by_word: dict[str, list[str]] = {g: [] for g in GLOSSES}
    for fname, gloss in pairs:
        word = label_to_word.get(gloss)
        if word:
            by_word[word].append(fname)
    return by_word


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ASL Citizen landmarks for our vocab.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="ASL_Citizen.zip — extract only our vocab's videos on the fly")
    src.add_argument("--asl-dir", help="already-unzipped ASL_Citizen dir (splits/ + videos/)")
    parser.add_argument("--max-per-gloss", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--fresh", action="store_true",
                        help="remove existing aslc_*.npy for our glosses first")
    args = parser.parse_args()

    from recognition.src.holistic import video_to_sequence  # needs mediapipe

    import tempfile
    import zipfile

    zf = None
    zip_members = None
    if args.zip:
        zf = zipfile.ZipFile(args.zip)
        # member lookup by basename, e.g. "15890366051589533-APPLE.mp4"
        zip_members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".mp4")}
        by_word = _group_by_word(_read_splits_zip(zf))
    else:
        asl_dir = Path(args.asl_dir)
        if not (asl_dir / "videos").is_dir():
            sys.exit(f"{asl_dir}/videos not found. Point --asl-dir at the unzipped ASL_Citizen.")
        by_word = _group_by_word(_read_splits_dir(asl_dir))

    def _seq_for(fname: str):
        """Landmark sequence for a video filename, from the zip (temp) or the unzipped dir."""
        if zf is not None:
            member = zip_members.get(fname)
            if member is None:
                return None
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(zf.read(member))
                tmp_path = tmp.name
            try:
                return video_to_sequence(tmp_path, args.seq_len)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        vpath = Path(args.asl_dir) / "videos" / fname
        return video_to_sequence(str(vpath), args.seq_len) if vpath.is_file() else None

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
            seq = _seq_for(fname)
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
