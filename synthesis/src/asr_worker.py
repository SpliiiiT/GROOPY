"""ASR worker — records from the mic and transcribes, in a clean Qt-free process.

Invoked as a subprocess by asr.record_and_transcribe_subprocess():
    python -m synthesis.src.asr_worker <out_txt_path> [seconds]

Loads sounddevice + faster-whisper HERE, isolated from the GUI's Qt/TensorFlow process (where
those native libs crash). Writes the transcript to <out_txt_path> (never stdout — a windowed
parent may have no usable stdout), then exits. Any failure just writes an empty file.

The frozen .exe uses launcher.py's `--asr-worker` branch instead (same logic), because a
PyInstaller onedir build can't run `python -m`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable when run as `python -m` from the repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run(out_path: str, seconds: float) -> None:
    text = ""
    try:
        from synthesis.src.asr import _record_wav_sounddevice, transcribe

        wav = _record_wav_sounddevice(seconds)      # capture mic audio (no PyAudio)
        text = transcribe(wav) or ""                # faster-whisper (bundled model, offline)
    except Exception:
        text = ""                                    # never propagate — parent reads the file
    try:
        Path(out_path).write_text(text, encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    if len(sys.argv) < 2:
        return
    out_path = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    run(out_path, seconds)


if __name__ == "__main__":
    main()
