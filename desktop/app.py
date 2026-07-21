"""GROOPY desktop app — the high-accuracy live demo (PyQt5 + OpenCV).

Runs the un-quantised winning model against your webcam. Because the desktop isn't
constrained by phone RAM/latency, this is the best-accuracy path and the ideal Project
Fair booth demo on a laptop with an RTX 2050.

Pipeline (identical logic to the mobile app):
  webcam frame -> MediaPipe hand crop -> preprocess -> model -> confidence gate + debounce
              -> Token -> on-screen text (+ optional TTS)

Usage:
  python desktop/app.py --model recognition/models/mobilenetv2_desktop.keras
  python desktop/app.py --model recognition/models/cnn_scratch.keras --speak
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the repo importable when run as a script (python desktop/app.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from recognition.src.config import CLASS_NAMES, IMG_SIZE  # noqa: E402
from recognition.src.preprocess import crop_hand, preprocess_for_model  # noqa: E402
from recognition.src.token_stream import TokenStream  # noqa: E402

NO_HAND = "no_hand"  # sentinel label: distinct from any real CLASS_NAMES prediction


class Recognizer:
    """Loads the winning keras model and turns a frame into (label, confidence).

    Only runs the model when MediaPipe actually finds a hand in frame — otherwise the CNN
    has no choice but to argmax some letter out of background noise (same problem WordStream
    solves for whole-word signs). crop_and_preprocess's "fall back to the full frame" behaviour
    is right for offline cropping, wrong for live inference, so this bypasses it and gates on
    crop_hand directly.
    """

    def __init__(self, model_path: str) -> None:
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES

    def predict(self, bgr_frame) -> tuple[str, float]:
        crop = crop_hand(bgr_frame, static=False)
        if crop is None:
            return NO_HAND, 0.0
        x = preprocess_for_model(crop)
        probs = self.model.predict(np.expand_dims(x, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])


class WordSigner:
    """Optional dynamic-word path: MediaPipe Holistic landmarks -> LSTM -> word Token.

    Runs alongside the fingerspelling Recognizer. Feed it each frame; it returns a
    kind="word" Token when a whole sign is recognised past the confidence gate, else None.
    """

    def __init__(self, model_path: str) -> None:
        import tensorflow as tf

        from recognition.src.holistic import landmarks
        from recognition.src.word_stream import WordStream

        self._landmarks = landmarks
        self.word_stream = WordStream(tf.keras.models.load_model(model_path))

    def push_frame(self, bgr_frame):
        return self.word_stream.push(self._landmarks(bgr_frame, static=False))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self, model_path: str | None = None, speak: bool = False,
        word_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("GROOPY — Sign → Text (desktop)")
        # Fingerspelling CNN is optional: run word-only if no --model is given.
        self.recognizer = Recognizer(model_path) if model_path else None
        self.stream = TokenStream(kind="letter")
        # Optional whole-word signing (Holistic + LSTM).
        self.word_signer = WordSigner(word_model_path) if word_model_path else None
        self.speak = speak
        self._tts = self._init_tts() if speak else None
        self.sentence: list[str] = []

        # UI
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.pred_label = QtWidgets.QLabel("…", alignment=QtCore.Qt.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 28px; font-weight: 600;")
        self.text_label = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignLeft)
        self.text_label.setStyleSheet("font-size: 20px; color: #333;")
        self.text_label.setWordWrap(True)

        capture_btn = QtWidgets.QPushButton("Capture word  (Space)")
        capture_btn.clicked.connect(self._capture_word)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(capture_btn)
        btn_row.addWidget(clear_btn)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.pred_label)
        layout.addWidget(QtWidgets.QLabel("Sentence:"))
        layout.addWidget(self.text_label)
        layout.addLayout(btn_row)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Camera loop
        self.cap = self._open_camera()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)  # ~33 fps capture; model runs each tick

    def _open_camera(self):
        """Open a working webcam. On Windows the default MSMF backend often returns no
        frames — DirectShow (CAP_DSHOW) is far more reliable — so try it first, and try a
        couple of device indices."""
        for idx in (0, 1, 2):
            for backend in (getattr(cv2, "CAP_DSHOW", 0), cv2.CAP_ANY):
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        print(f"camera opened (index {idx})")
                        return cap
                cap.release()
        print("WARNING: no working webcam found — check Windows camera privacy settings "
              "(Settings > Privacy & security > Camera > let desktop apps access) and that "
              "no other app is using the camera.")
        return cv2.VideoCapture(0)

    def _init_tts(self):
        try:
            import pyttsx3

            return pyttsx3.init()
        except Exception:
            print("pyttsx3 not available — running without speech.")
            return None

    def _clear(self) -> None:
        self.sentence = []
        self.text_label.setText("")

    def _capture_word(self) -> None:
        """Commit the current live word guess to the sentence, then reset the buffer."""
        if self.word_signer is None:
            return
        ws = self.word_signer.word_stream
        if not ws.ready or ws.last_gloss is None:
            self.pred_label.setText("… hold a sign steady ~1s, then Space")
            return
        import time as _t

        from shared.contract import KIND_WORD, Token

        self._apply_token(Token(
            token=ws.last_gloss, confidence=round(ws.last_conf, 3),
            timestamp=int(_t.time() * 1000), kind=KIND_WORD,
        ))
        ws.reset()  # fresh window for the next sign

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Space:
            self._capture_word()
        else:
            super().keyPressEvent(event)

    def _tick(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.pred_label.setText("no camera feed — check camera privacy / other apps")
            return
        frame = cv2.flip(frame, 1)  # mirror for natural signing

        # NOTE: predict() runs synchronously here, blocking the UI thread for its full
        # duration each tick. Fine for lightweight models; a heavy one (e.g. ResNet50)
        # will make the GUI stutter. Proper fix: run Recognizer.predict on a QThread
        # worker and deliver (label, conf) back via a signal.
        if self.recognizer is not None:
            label, conf = self.recognizer.predict(frame)
            if label == NO_HAND:
                self.pred_label.setText("no hand detected — show a letter")
            else:
                self.pred_label.setText(f"{label}   ({conf:.0%})")
            token = self.stream.update(label, conf)
            if token is not None:
                self._apply_token(token)

        # Optional whole-word path: feed the same frame to the word signer for a LIVE guess.
        # Words are committed only on Capture (Space) — a sliding window would otherwise spew
        # a word every debounce tick from transitional frames.
        if self.word_signer is not None:
            self.word_signer.push_frame(frame)
            ws = self.word_signer.word_stream
            if ws.ready and ws.last_gloss is not None:
                self.pred_label.setText(
                    f"{ws.last_gloss}   ({ws.last_conf:.0%})   — Space to add"
                )
            elif self.recognizer is None:
                self.pred_label.setText(
                    "… buffering (hold a sign ~1s)" if not ws.ready
                    else "no hands detected — sign a word"
                )

        # render frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio
        ))

    def _apply_token(self, token) -> None:
        # Contract-aware handling of controls, letters, and whole words.
        if token.token == "space":
            self.sentence.append(" ")
        elif token.token == "del":
            if self.sentence:
                self.sentence.pop()
        elif token.token == "nothing":
            return
        elif token.kind == "word":
            # Whole-word signs read as words, so add spacing around them.
            self.sentence.append((" " if self.sentence else "") + token.token + " ")
            if self._tts:
                self._tts.say(token.token)
                self._tts.runAndWait()
        else:
            self.sentence.append(token.token)
            if self._tts:
                self._tts.say(token.token)
                self._tts.runAndWait()
        self.text_label.setText("".join(self.sentence))
        # token.to_dict() is exactly what Track B would receive.

    def closeEvent(self, event) -> None:
        self.cap.release()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="GROOPY desktop live demo.")
    parser.add_argument("--model", default=None, help="fingerspelling .keras model (optional)")
    parser.add_argument("--speak", action="store_true", help="enable text-to-speech")
    parser.add_argument(
        "--word-model",
        default=None,
        help="lstm_word.keras to recognise whole-word signs (optional)",
    )
    args = parser.parse_args()
    if not args.model and not args.word_model:
        parser.error("give at least one of --model (fingerspelling) or --word-model (words).")

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args.model, speak=args.speak, word_model_path=args.word_model)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
