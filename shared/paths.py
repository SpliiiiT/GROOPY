"""App-root path resolution — works both from source and from a PyInstaller-frozen build.

recognition/src/config.py, sentiment/src/config.py, and shared/config.py all locate their
data/models/results relative to "the repo root". That's simply `Path(__file__).resolve().
parents[N]` when running from source, but PyInstaller loads modules from inside a bundle, so
`__file__`-relative paths don't point at the bundled data. Centralising the frozen-vs-source
check here means the packaged build (see packaging/groopy.spec) doesn't need each config file
to duplicate it.
"""
from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """Repo root when running from source; the PyInstaller bundle's data root when frozen."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]  # shared/ is one level under the repo root
