"""GROOPY synthesis demo — Text/Speech -> Sign (PyQt5).

The Track B counterpart to app.py: type a sentence (or click Speak to use the mic), and the
app plays the matching sign-video clips in sequence, fingerspelling any word outside the
curated vocabulary. Shows the gloss breakdown and the (optional) sentiment label.

Usage:
  python desktop/synthesis_app.py
  python desktop/synthesis_app.py --no-sentiment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from synthesis.src.pipeline import synthesize  # noqa: E402


class SynthWindow(QtWidgets.QMainWindow):
    def __init__(self, with_sentiment: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("GROOPY — Text/Speech -> Sign")
        self.with_sentiment = with_sentiment

        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Type a sentence, e.g. 'hello how are you'")
        self.input.returnPressed.connect(self._on_submit)

        sign_btn = QtWidgets.QPushButton("Sign it")
        sign_btn.clicked.connect(self._on_submit)
        speak_btn = QtWidgets.QPushButton("🎤 Speak")
        speak_btn.clicked.connect(self._on_speak)

        self.gloss_label = QtWidgets.QLabel("")
        self.gloss_label.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.gloss_label.setWordWrap(True)
        self.sentiment_label = QtWidgets.QLabel("")
        self.sentiment_label.setStyleSheet("font-size: 16px; color: #555;")
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color: #888;")

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.input)
        row.addWidget(sign_btn)
        row.addWidget(speak_btn)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(row)
        layout.addWidget(QtWidgets.QLabel("Gloss / plan:"))
        layout.addWidget(self.gloss_label)
        layout.addWidget(self.sentiment_label)
        layout.addWidget(self.status)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(640, 240)

    def _render(self, text: str) -> None:
        if not text.strip():
            return
        result = synthesize(text=text, with_sentiment=self.with_sentiment)
        self.gloss_label.setText(result.plan.summary())
        if result.sentiment:
            s = result.sentiment
            self.sentiment_label.setText(f"sentiment: {s.label} ({s.score:.0%})")
        else:
            self.sentiment_label.setText("")
        # Play the clips (blocks until done / 'q'). Missing assets are reported to status.
        missing: list[str] = []
        from synthesis.src.player import play_sign_plan

        self.status.setText("Playing… (press q/Esc in the video window to stop)")
        QtWidgets.QApplication.processEvents()
        play_sign_plan(result.plan, on_missing=missing.append)
        self.status.setText(
            "Done." if not missing else f"Done — {len(missing)} asset(s) missing (run stub/clip download)."
        )

    def _on_submit(self) -> None:
        self._render(self.input.text())

    def _on_speak(self) -> None:
        self.status.setText("Listening…")
        QtWidgets.QApplication.processEvents()
        try:
            from synthesis.src.asr import listen_mic

            text = listen_mic()
        except Exception as e:  # missing backend / mic
            self.status.setText(str(e))
            return
        self.input.setText(text)
        self._render(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="GROOPY text/speech -> sign demo.")
    parser.add_argument("--no-sentiment", action="store_true", help="skip sentiment analysis")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = SynthWindow(with_sentiment=not args.no_sentiment)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
