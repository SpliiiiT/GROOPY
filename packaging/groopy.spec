# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the GROOPY desktop launcher (Sign<->Text, one packaged app).

Build (from the repo root, venv active):
  pyinstaller packaging/groopy.spec --distpath dist --workpath build --noconfirm

Bundles the two default models (cnn_scratch + word_transformer), the sign-clip library, and
the curated per-letter fingerspelling images -- everything the launcher needs to run without
the dev environment. Does NOT bundle training data, the CNN bake-off's other 3 models, or
sentiment's optional transformers/torch backends (LexiconBackend has zero dependencies and is
what ships; the ML sentiment backends stay a dev/opt-in feature, not part of the packaged app).
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

REPO_ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(REPO_ROOT / "recognition" / "models" / "cnn_scratch.keras"), "recognition/models"),
    (str(REPO_ROOT / "recognition" / "models" / "word_transformer.keras"), "recognition/models"),
    (str(REPO_ROOT / "synthesis" / "clips"), "synthesis/clips"),
    (str(REPO_ROOT / "synthesis" / "assets" / "letters"), "synthesis/assets/letters"),
    # MediaPipe ships its own graph/model assets (.tflite/.binarypb) as package data that
    # PyInstaller's default analysis doesn't auto-collect (it only traces Python imports, not
    # runtime-loaded resource files) -- without this, mediapipe.solutions.hands.Hands() raises
    # FileNotFoundError the moment it's actually instantiated (i.e. the app launches fine, then
    # crashes the first time recognition tries to process a frame).
] + collect_data_files("mediapipe")

a = Analysis(
    [str(REPO_ROOT / "desktop" / "launcher.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # torch/transformers are pulled in transitively (reachable from sentiment/src/models.py's
    # pretrained backends) but never actually used by the packaged app's default code path --
    # LexiconBackend ships, the ML sentiment models are a dev-only opt-in (see
    # sentiment/requirements.txt). Excluding them cuts ~1GB+ of unused bloat.
    excludes=["torch", "transformers"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GROOPY",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # crash was diagnosed (mediapipe data files) and fixed; ship windowed
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GROOPY",
)
