"""GROOPY launcher — one entry point for both directions.

Wraps the existing apps (app.py's MainWindow, synthesis_app.py's SynthWindow) behind a single
window instead of two separate `python ...` invocations — this is what a packaged build (see
packaging/) actually launches. Reuses both window classes unmodified; only one QApplication
may exist per process, so the launcher owns it and both windows attach to it. Opening one
doesn't close the launcher, so both directions can run side by side.

Usage:
  python desktop/launcher.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Make the repo importable when run as a script (python desktop/launcher.py) AND when frozen
# into a packaged .exe (PyInstaller sets sys.frozen; sys.path already includes the bundle root
# in that case, so this insert is a harmless no-op there).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5 import QtCore, QtWidgets  # noqa: E402


def _default_model_paths() -> tuple[Optional[str], Optional[str]]:
    """(cnn_path, word_path) for whichever default models are actually present on disk —
    same optional-model philosophy as app.py: run with whatever's available."""
    from recognition.src.config import MODELS_DIR

    cnn = MODELS_DIR / "cnn_scratch.keras"
    word = MODELS_DIR / "word_transformer.keras"
    return (
        str(cnn) if cnn.is_file() else None,
        str(word) if word.is_file() else None,
    )


class LauncherWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GROOPY")
        self._children: list[QtWidgets.QMainWindow] = []  # keep refs so Qt doesn't GC them

        title = QtWidgets.QLabel("GROOPY")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        subtitle = QtWidgets.QLabel("Two-way sign language communication — pick a direction:")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setWordWrap(True)

        recognize_btn = QtWidgets.QPushButton("🖐  Sign → Text  (webcam)")
        recognize_btn.setMinimumHeight(48)
        recognize_btn.clicked.connect(self._open_recognition)

        synth_btn = QtWidgets.QPushButton("\U0001F4AC  Text → Sign")
        synth_btn.setMinimumHeight(48)
        synth_btn.clicked.connect(self._open_synthesis)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color: #888;")
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.status.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(recognize_btn)
        layout.addWidget(synth_btn)
        layout.addWidget(self.status)
        layout.setSpacing(12)
        layout.setContentsMargins(32, 32, 32, 32)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(440, 280)

    def _open_recognition(self) -> None:
        from desktop.app import MainWindow

        cnn_path, word_path = _default_model_paths()
        if cnn_path is None and word_path is None:
            self.status.setText(
                "No recognition models found in recognition/models/ — nothing to open."
            )
            return
        win = MainWindow(cnn_path, speak=False, word_model_path=word_path)
        win.show()
        self._children.append(win)
        self.status.setText("Sign → Text opened.")

    def _open_synthesis(self) -> None:
        from desktop.synthesis_app import SynthWindow

        win = SynthWindow(with_sentiment=True)
        win.show()
        self._children.append(win)
        self.status.setText("Text → Sign opened.")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
