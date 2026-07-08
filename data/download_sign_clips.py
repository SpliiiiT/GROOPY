"""Prepare WLASL word-level data — serves BOTH tracks from one source.

For each gloss in the shared vocabulary this script produces:
  1. Synthesis playback clip: one representative video copied to synthesis/clips/<gloss>.mp4
     (the CLIP_MAP filename), so Text/Speech -> Sign can play it.
  2. Recognition training sequences: MediaPipe Holistic landmarks extracted from every
     sample video into data/wlasl_landmarks/<gloss>/<sample>.npy for the word LSTM.

WLASL acquisition is manual and non-commercial (videos are YouTube-sourced via the WLASL
metadata release). This script therefore consumes videos you have ALREADY downloaded,
laid out one folder per gloss:

    <wlasl-videos>/<gloss>/<anything>.mp4

Usage:
  python data/download_sign_clips.py --videos /path/to/wlasl_videos
  python data/download_sign_clips.py --videos /path/to/wlasl_videos --clips-only

See: https://github.com/dxli94/WLASL  (licence: computer-vision research, non-commercial).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import CLIPS_DIR, LANDMARKS_DIR, SEQ_LEN  # noqa: E402
from shared.vocabulary import CLIP_MAP, GLOSSES  # noqa: E402

_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".avi", ".mkv")


def _videos_for(gloss_dir: Path) -> list[Path]:
    return [p for p in sorted(gloss_dir.iterdir()) if p.suffix.lower() in _VIDEO_EXTS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WLASL clips + landmark sequences.")
    parser.add_argument(
        "--videos", required=True,
        help="folder of pre-downloaded WLASL videos, one subfolder per gloss",
    )
    parser.add_argument("--clips-only", action="store_true", help="skip landmark extraction")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()

    videos_root = Path(args.videos)
    if not videos_root.is_dir():
        sys.exit(f"--videos path not found: {videos_root}")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.clips_only:
        LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
        # Import here so the clips-only path needs no mediapipe.
        from recognition.src.holistic import video_to_sequence

    missing_glosses = []
    n_clips = n_seqs = 0
    for gloss in GLOSSES:
        gloss_dir = videos_root / gloss
        if not gloss_dir.is_dir():
            missing_glosses.append(gloss)
            continue
        vids = _videos_for(gloss_dir)
        if not vids:
            missing_glosses.append(gloss)
            continue

        # 1) representative clip for playback
        shutil.copy(vids[0], CLIPS_DIR / CLIP_MAP[gloss])
        n_clips += 1

        # 2) landmark sequences for LSTM training
        if not args.clips_only:
            out_dir = LANDMARKS_DIR / gloss
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, vid in enumerate(vids):
                seq = video_to_sequence(str(vid), args.seq_len)
                if seq is not None:
                    np.save(out_dir / f"{gloss}_{i:03d}.npy", seq)
                    n_seqs += 1

    print(f"Clips written: {n_clips} -> {CLIPS_DIR}")
    if not args.clips_only:
        print(f"Landmark sequences written: {n_seqs} -> {LANDMARKS_DIR}")
    if missing_glosses:
        print(f"WARNING: no videos found for {len(missing_glosses)} glosses: {missing_glosses}")
        print("Those words will be fingerspelled by Synthesis and unavailable to the LSTM.")


if __name__ == "__main__":
    main()
