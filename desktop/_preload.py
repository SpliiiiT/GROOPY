"""Process-wide shims — imported before PyQt5 in the desktop entry points.

The GUI process deliberately does NOT load faster-whisper / ctranslate2 / torch: those native
libraries crash when loaded alongside Qt in the packaged .exe (torch c10.dll WinError 1114, and
a duplicate-OpenMP abort with TensorFlow). Speech-to-text therefore runs in a SEPARATE worker
process (see synthesis/src/asr_worker.py and launcher.py's --asr-worker branch), where there is
no Qt, so Whisper loads cleanly and a crash there can never take down the GUI.

All that remains here is the OpenMP guard (harmless, and covers TensorFlow + any stray load).
"""
from __future__ import annotations

import os

# Let multiple OpenMP runtimes coexist instead of aborting the process on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
