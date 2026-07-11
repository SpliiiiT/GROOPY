"""Prepare WLASL word-level data — serves BOTH tracks from one source.

Reads the WLASL metadata JSON + the flat videos/ folder (from data/download_wlasl.py) and,
for each gloss in the shared vocabulary, produces:
  1. Synthesis playback clip: one representative (frame-trimmed) video -> synthesis/clips/<gloss>.mp4
  2. Recognition training sequences: MediaPipe Holistic landmarks for every available
     instance -> data/wlasl_landmarks/<gloss>/<i>.npy  (for the word LSTM)

Only our curated glosses are processed. WLASL gloss strings don't always match ours (e.g.
"thanks" vs "thank you"), so a small ALIASES map reconciles them; anything still unmatched
is reported and simply gets fingerspelled by Synthesis / skipped by the LSTM.

Usage:
  python data/download_wlasl.py            # first: fetch the dataset
  python data/download_sign_clips.py       # then: build clips + sequences
  python data/download_sign_clips.py --clips-only --max-per-gloss 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import CLIPS_DIR, LANDMARKS_DIR, SEQ_LEN, WLASL_DIR  # noqa: E402
from shared.vocabulary import CLIP_MAP, GLOSSES, wlasl_gloss_candidates  # noqa: E402


def _find_json(wlasl_dir: Path) -> Path:
    for name in ("WLASL_v0.3.json", "WLASL_v0.3(1).json", "wlasl.json"):
        p = wlasl_dir / name
        if p.is_file():
            return p
    hits = list(wlasl_dir.glob("*.json"))
    if hits:
        return hits[0]
    sys.exit(f"No WLASL metadata JSON under {wlasl_dir}. Run data/download_wlasl.py first.")


def _videos_dir(wlasl_dir: Path) -> Path:
    for cand in (wlasl_dir / "videos", wlasl_dir):
        if cand.is_dir() and any(cand.glob("*.mp4")):
            return cand
    sys.exit(f"No videos/*.mp4 found under {wlasl_dir}. Run data/download_wlasl.py first.")


def _write_trimmed_clip(src: Path, dst: Path, frame_start: int, frame_end: int) -> bool:
    """Copy src -> dst keeping only the WLASL frame range (1-indexed, inclusive)."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        cap.release()
        return False
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    lo = max(0, frame_start - 1)
    hi = float("inf") if frame_end is None or frame_end < 0 else frame_end
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if lo <= i < hi:
            writer.write(frame)
        i += 1
    writer.release()
    cap.release()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WLASL clips + landmark sequences.")
    parser.add_argument("--wlasl-dir", default=str(WLASL_DIR))
    parser.add_argument("--clips-only", action="store_true", help="skip landmark extraction")
    parser.add_argument("--max-per-gloss", type=int, default=25,
                        help="cap instances processed per gloss (extraction is CPU-heavy)")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()

    wlasl_dir = Path(args.wlasl_dir)
    meta = json.loads(_find_json(wlasl_dir).read_text())
    videos_dir = _videos_dir(wlasl_dir)

    # gloss (lowercased, as WLASL stores it) -> list of instance dicts
    by_gloss = {entry["gloss"].lower(): entry["instances"] for entry in meta}

    if not args.clips_only:
        from recognition.src.holistic import video_to_sequence  # needs mediapipe

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    found, missing = [], []
    n_clips = n_seqs = 0

    for gloss in GLOSSES:
        instances = None
        for cand in wlasl_gloss_candidates(gloss):
            if cand.lower() in by_gloss:
                instances = by_gloss[cand.lower()]
                break
        if not instances:
            missing.append(gloss)
            continue

        # Instances whose video actually exists on disk.
        avail = [
            (inst, videos_dir / f"{inst['video_id']}.mp4")
            for inst in instances
            if (videos_dir / f"{inst['video_id']}.mp4").is_file()
        ]
        if not avail:
            missing.append(gloss)
            continue
        found.append(gloss)

        # 1) representative playback clip (trimmed to its frame range)
        inst0, path0 = avail[0]
        if _write_trimmed_clip(path0, CLIPS_DIR / CLIP_MAP[gloss],
                               inst0.get("frame_start", 1), inst0.get("frame_end", -1)):
            n_clips += 1

        # 2) landmark sequences for the LSTM
        if not args.clips_only:
            out_dir = LANDMARKS_DIR / gloss
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, (inst, path) in enumerate(avail[: args.max_per_gloss]):
                seq = video_to_sequence(
                    str(path), args.seq_len,
                    frame_start=inst.get("frame_start", 1),
                    frame_end=inst.get("frame_end", -1),
                )
                if seq is not None:
                    np.save(out_dir / f"{gloss}_{i:03d}.npy", seq)
                    n_seqs += 1

    print(f"\nGlosses found in WLASL: {len(found)}/{len(GLOSSES)} -> {found}")
    if missing:
        print(f"NOT found ({len(missing)}): {missing}")
        print("  -> Synthesis will fingerspell these; the LSTM won't learn them.")
        print("  -> Inspect data/wlasl/wlasl_class_list.txt and extend ALIASES to recover more.")
    print(f"Clips written: {n_clips} -> {CLIPS_DIR}")
    if not args.clips_only:
        print(f"Landmark sequences: {n_seqs} -> {LANDMARKS_DIR}")
        print("Next: python -m recognition.src.train_word --epochs 40")


if __name__ == "__main__":
    main()
