"""Import-order + OpenMP shims — MUST be imported before PyQt5 in every desktop entry point.

Two native-library landmines this defuses:

1. torch-after-Qt DLL crash. faster-whisper -> ctranslate2 does `import torch`, and torch's
   c10.dll fails to initialise (WinError 1114) if Qt was loaded first. Importing faster-whisper
   HERE — before any PyQt5 import — loads torch's DLLs first, after which Qt loads cleanly.
   (In the packaged .exe torch is excluded, so this import is a harmless no-op there.)

2. Duplicate OpenMP abort. TensorFlow and ctranslate2 each ship their own OpenMP runtime; on
   Windows loading both aborts the process ("OMP: Error #15 ... libiomp5md.dll already
   initialized"), which looks like the app just closing. KMP_DUPLICATE_LIB_OK lets them coexist.

Keep this import at the very top of launcher.py / synthesis_app.py (before `from PyQt5 ...`).
"""
from __future__ import annotations

import os

# Allow TensorFlow's and ctranslate2's OpenMP runtimes to coexist instead of aborting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Quieter, and avoids some thread-oversubscription stalls when both libs are loaded.
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))

try:
    # Load ctranslate2/torch DLLs before Qt. Best-effort: if faster-whisper isn't installed
    # (or torch is excluded, as in the .exe), just skip — the mic path degrades gracefully.
    import faster_whisper  # noqa: F401
except Exception:
    pass
