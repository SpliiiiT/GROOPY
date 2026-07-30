"""GROOPY synthesis demo — Text/Speech -> Sign (PyQt5).

The Track B counterpart to app.py: type a sentence (or click Speak to use the mic), and the
app plays the matching sign-video clips in sequence, fingerspelling any word outside the
curated vocabulary. Shows the gloss breakdown and the sentiment (which also drives signing
emphasis — a held pause + replay on strong, non-neutral sentiment).

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

import desktop._preload  # noqa: E402,F401  MUST precede PyQt5 (torch/OpenMP DLL shims)

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from desktop import theme  # noqa: E402
from synthesis.src.pipeline import EMPHASIS_SCORE_THRESHOLD, synthesize  # noqa: E402


class SynthWindow(QtWidgets.QMainWindow):
    def __init__(self, with_sentiment: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("GROOPY — Text / Speech → Sign")
        self.with_sentiment = with_sentiment

        # Header
        title = QtWidgets.QLabel("Text / Speech → Sign", objectName="title")
        subtitle = QtWidgets.QLabel(
            "Type or speak a sentence · known words play sign clips, the rest are fingerspelled",
            objectName="subtitle",
        )
        subtitle.setWordWrap(True)

        # Input row
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("e.g.  hello how are you   ·   thank you friend")
        self.input.returnPressed.connect(self._on_submit)
        self.input.setMinimumHeight(44)

        sign_btn = QtWidgets.QPushButton("Sign it")
        sign_btn.clicked.connect(self._on_submit)
        self.speak_btn = QtWidgets.QPushButton("🎤  Speak", objectName="ghost")
        self.speak_btn.clicked.connect(self._on_speak)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.input, stretch=1)
        row.addWidget(sign_btn)
        row.addWidget(self.speak_btn)

        # Result card
        sec = QtWidgets.QLabel("GLOSS / SIGN PLAN", objectName="section")
        self.badge = QtWidgets.QLabel("")
        self.badge.setVisible(False)
        head_row = QtWidgets.QHBoxLayout()
        head_row.addWidget(sec)
        head_row.addStretch(1)
        head_row.addWidget(self.badge)

        self.gloss_label = QtWidgets.QLabel("—", objectName="sentence")
        self.gloss_label.setWordWrap(True)
        self.gloss_label.setMinimumHeight(48)
        self.emphasis_label = QtWidgets.QLabel("", objectName="subtitle")

        card_l = QtWidgets.QVBoxLayout()
        card_l.setContentsMargins(16, 12, 16, 14)
        card_l.addLayout(head_row)
        card_l.addWidget(self.gloss_label)
        card_l.addWidget(self.emphasis_label)
        card = QtWidgets.QFrame(objectName="card")
        card.setLayout(card_l)

        self.status = QtWidgets.QLabel("Ready.", objectName="status")
        self.status.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(row)
        layout.addWidget(card)
        layout.addWidget(self.status)
        layout.addStretch(1)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(720, 340)

    # ---------------------------------------------------------------- core
    def _render(self, text: str) -> None:
        if not text.strip():
            return
        result = synthesize(text=text, with_sentiment=self.with_sentiment)
        self.gloss_label.setText(result.plan.summary())

        # Sentiment badge (coloured) + note on whether it changed the signing.
        s = result.sentiment
        theme.set_sentiment_badge(self.badge, s)
        if s and s.label != "neutral" and s.score >= EMPHASIS_SCORE_THRESHOLD:
            self.emphasis_label.setText(
                f"↳ strong {s.label} sentiment — signs are emphasised (held + replayed)"
            )
        else:
            self.emphasis_label.setText("")

        # Play the clips (blocks until done / 'q'). Missing assets are reported to status.
        missing: list[str] = []
        from synthesis.src.player import play_sign_plan

        self.status.setText("Playing…  (press q / Esc in the video window to stop)")
        QtWidgets.QApplication.processEvents()
        play_sign_plan(result.plan, on_missing=missing.append)
        self.status.setText(
            "Done." if not missing
            else f"Done — {len(missing)} asset(s) missing (run the stub/clip download)."
        )

    def _on_submit(self) -> None:
        self._render(self.input.text())

    def _on_speak(self) -> None:
        self.speak_btn.setEnabled(False)
        self.status.setText("🎙  Listening… speak now (~4s)")
        QtWidgets.QApplication.processEvents()
        try:
            from synthesis.src.asr import record_and_transcribe

            text = record_and_transcribe(seconds=4.0)
        except Exception as e:  # missing backend / mic
            self.status.setText(f"Speech failed: {e}")
            self.speak_btn.setEnabled(True)
            return
        self.speak_btn.setEnabled(True)
        if not text.strip():
            self.status.setText("Didn't catch that — try again.")
            return
        self.input.setText(text)
        self.status.setText(f'Heard: "{text}"')
        QtWidgets.QApplication.processEvents()
        self._render(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="GROOPY text/speech -> sign demo.")
    parser.add_argument("--no-sentiment", action="store_true", help="skip sentiment analysis")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    theme.apply_theme(app)
    win = SynthWindow(with_sentiment=not args.no_sentiment)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
