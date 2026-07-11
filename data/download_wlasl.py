"""Targeted WLASL download — fetch ONLY the videos our vocabulary needs.

The `wlasl-processed` Kaggle mirror is the full WLASL2000 corpus (~thousands of videos,
several GB). We only need ~20 glosses, so instead of pulling the whole dataset this script:
  1. downloads the metadata JSON (~12 MB),
  2. resolves our vocabulary's glosses (via WLASL aliases) to their video_ids,
  3. downloads ONLY those individual video files.

Reuses the same kaggle.json / access token as the ASL Alphabet flow.

Usage:
  python data/download_wlasl.py                 # our vocabulary, up to 25 videos/gloss
  python data/download_wlasl.py --max-per-gloss 40
  python data/download_wlasl.py --all           # the whole dataset (several GB) instead
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import WLASL_DIR  # noqa: E402
from shared.vocabulary import GLOSSES, wlasl_gloss_candidates  # noqa: E402

DATASET = "risangbaskoro/wlasl-processed"
DATA_DIR = Path(__file__).resolve().parent
META_NAME = "WLASL_v0.3.json"

_API = None


def _api():
    """Authenticated Kaggle API client (works regardless of PATH / venv activation)."""
    global _API
    if _API is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        _API = KaggleApi()
        _API.authenticate()
    return _API


def _download_file(remote_name: str, out_dir: Path) -> Path:
    """Download a single dataset file into out_dir, unzipping if Kaggle wrapped it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # quiet=True; the API places the file (or <file>.zip) directly under out_dir.
    _api().dataset_download_file(DATASET, remote_name, path=str(out_dir), quiet=True)
    base = Path(remote_name).name
    zipped = out_dir / f"{base}.zip"
    if zipped.is_file():
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(out_dir)
        zipped.unlink()
    return out_dir / base


def _download_all() -> None:
    WLASL_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the FULL wlasl-processed dataset (several GB) ...")
    _api().dataset_download_files(DATASET, path=str(WLASL_DIR), unzip=True, quiet=False)
    print("Done ->", WLASL_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted WLASL download for our vocabulary.")
    parser.add_argument("--max-per-gloss", type=int, default=25)
    parser.add_argument("--all", action="store_true", help="download the entire dataset instead")
    args = parser.parse_args()

    if args.all:
        _download_all()
        return

    # 1) metadata JSON
    print(f"Fetching {META_NAME} ...")
    meta_path = _download_file(META_NAME, WLASL_DIR)
    meta = json.loads(meta_path.read_text())
    by_gloss = {e["gloss"].lower(): e["instances"] for e in meta}

    # 2) resolve our glosses -> video_ids
    videos_dir = WLASL_DIR / "videos"
    wanted: list[tuple[str, str]] = []   # (gloss, video_id)
    found, missing = [], []
    for gloss in GLOSSES:
        insts = None
        for cand in wlasl_gloss_candidates(gloss):
            if cand.lower() in by_gloss:
                insts = by_gloss[cand.lower()]
                break
        if not insts:
            missing.append(gloss)
            continue
        found.append(gloss)
        for inst in insts[: args.max_per_gloss]:
            wanted.append((gloss, str(inst["video_id"])))

    print(f"Glosses matched in WLASL: {len(found)}/{len(GLOSSES)} -> {found}")
    if missing:
        print(f"NOT matched ({len(missing)}): {missing}")
        print("  -> extend WLASL_ALIASES in shared/vocabulary.py to recover more.")

    # 3) download only the needed videos (skip any already present)
    print(f"\nDownloading up to {len(wanted)} videos -> {videos_dir}")
    got = skipped = failed = 0
    for _gloss, vid in wanted:
        if (videos_dir / f"{vid}.mp4").is_file():
            skipped += 1
            continue
        try:
            _download_file(f"videos/{vid}.mp4", videos_dir)
            got += 1
        except Exception:
            failed += 1   # some instances are absent from the mirror (404) — skip and continue
    print(f"videos: downloaded={got} skipped(existing)={skipped} unavailable={failed}")
    print("Next: python data/download_sign_clips.py")


if __name__ == "__main__":
    main()
