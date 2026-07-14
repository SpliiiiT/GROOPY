"""Recover extra WLASL videos from their original direct-URL sources (free supplement).

Some WLASL instances aren't in the Kaggle mirror yet AREN'T flagged in missing.txt — their
source clips (signingsavvy, spreadthesign, aslsignbank, asldeafined, signschool, startasl,
aslsearch, ...) are direct MP4 downloads that may still be live. This grabs the ones for our
vocabulary that we don't already have locally, straight from those URLs. No Kaggle, no forms.

In practice ~half the sources are still alive (dead ones are skipped). Idempotent: existing
videos are left untouched.

Usage:
  python data/recover_wlasl_extra.py
  python data/download_sign_clips.py     # then re-extract landmarks (incl. the recovered clips)
"""
from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import WLASL_DIR  # noqa: E402
from shared.vocabulary import GLOSSES, wlasl_gloss_candidates  # noqa: E402

_HDR = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 15


def _looks_like_video(data: bytes, url: str) -> bool:
    # crude but effective: real MP4/MOV have 'ftyp' in the first bytes; else trust .mp4 URLs
    return len(data) > 5000 and (b"ftyp" in data[:64] or url.lower().endswith(".mp4"))


def main() -> None:
    meta_path = WLASL_DIR / "WLASL_v0.3.json"
    if not meta_path.is_file():
        sys.exit(f"{meta_path} not found. Run data/download_wlasl.py first.")
    meta = json.loads(meta_path.read_text())
    by = {e["gloss"].lower(): e["instances"] for e in meta}
    missing = set((WLASL_DIR / "missing.txt").read_text().split()) if (WLASL_DIR / "missing.txt").is_file() else set()
    videos = WLASL_DIR / "videos"
    videos.mkdir(parents=True, exist_ok=True)

    # our-vocab instances: not on disk, not officially dead, and have a source URL
    targets = []
    for g in GLOSSES:
        for cand in wlasl_gloss_candidates(g):
            if cand.lower() in by:
                for i in by[cand.lower()]:
                    vid = str(i["video_id"])
                    if (not (videos / f"{vid}.mp4").is_file()
                            and vid not in missing and i.get("url")):
                        targets.append((g, vid, i.get("source", "?"), i["url"]))
                break

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    got = Counter()
    failed = Counter()
    for _g, vid, src, url in targets:
        try:
            req = urllib.request.Request(url, headers=_HDR)
            with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ctx) as r:
                data = r.read()
            if _looks_like_video(data, url):
                (videos / f"{vid}.mp4").write_bytes(data)
                got[src] += 1
            else:
                failed[src] += 1
        except Exception:
            failed[src] += 1

    print(f"tried {len(targets)} | recovered {sum(got.values())} | failed {sum(failed.values())}")
    if got:
        print("recovered by source:", dict(got))
    if failed:
        print("dead/blocked by source:", dict(failed))
    print(f"videos on disk now: {len(list(videos.glob('*.mp4')))}")
    if got:
        print("Next: python data/download_sign_clips.py  (re-extract landmarks)")


if __name__ == "__main__":
    main()
