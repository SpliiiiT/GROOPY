"""Speech-to-text for the Synthesis track.

ASR only produces TEXT — that text then flows through the exact same text_to_gloss ->
sign-plan path as typed input, so speech adds no downstream complexity.

Two optional backends, imported lazily with a clear install hint (same pattern as
preprocess._get_hands / app._init_tts):
  - faster-whisper  (offline, recommended)   -> transcribe() and listen_mic()
  - SpeechRecognition (online Google, mic)   -> fallback

Nothing here is imported unless you actually call it, so the rest of the pipeline runs
without any ASR dependency installed.
"""
from __future__ import annotations

from typing import Optional

_MISSING = (                                     # shown if the user calls ASR with no backend installed
    "No ASR backend installed. Install one:\n"
    "  pip install faster-whisper        # offline, recommended\n"
    "  pip install SpeechRecognition      # online fallback (mic needs PyAudio)"
)

_WHISPER = None                                  # cached whisper model (built once)


def _whisper_source(model_size: str = "base") -> str:
    """Where to load the Whisper model from.

    Packaged app: a bundled model dir (models/whisper-<size>) ships inside the build (see
    packaging/groopy.spec), so mic transcription works OFFLINE. Dev/source: fall back to the
    size NAME, which faster-whisper downloads/caches from HuggingFace on first use.
    """
    try:
        from shared.paths import app_root

        bundled = app_root() / "models" / f"whisper-{model_size}"   # _MEIPASS/... when frozen
        if (bundled / "model.bin").is_file():
            return str(bundled)
    except Exception:
        pass
    return model_size


def _get_whisper(model_size: str = "base"):
    """Lazily construct a cached faster-whisper model, or return None if unavailable."""
    global _WHISPER
    if _WHISPER is not None:                      # already built -> reuse
        return _WHISPER
    try:
        from faster_whisper import WhisperModel   # optional dep
    except Exception:
        return None                               # not installed -> signal "no whisper"
    # int8 = quantised, fast on CPU. Prefer a bundled model dir (offline) over a name (downloads).
    _WHISPER = WhisperModel(_whisper_source(model_size), device="cpu", compute_type="int8")
    return _WHISPER


def transcribe(wav_path: str, model_size: str = "base") -> str:
    """Transcribe an audio file to text. Tries faster-whisper, else SpeechRecognition."""
    model = _get_whisper(model_size)
    if model is not None:                         # preferred path: offline whisper
        segments, _ = model.transcribe(wav_path)  # -> iterable of text segments
        return " ".join(seg.text for seg in segments).strip()   # join segments into one string

    # Fallback: SpeechRecognition reading the wav file.
    try:
        import speech_recognition as sr
    except Exception:
        raise RuntimeError(_MISSING)              # neither backend available
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:        # read the wav
        audio = r.record(source)
    return r.recognize_google(audio)              # online Google transcription


def _record_wav_sounddevice(seconds: float, fs: int = 16000) -> str:
    """Record `seconds` of mono 16 kHz audio from the default mic to a temp WAV file.

    Uses sounddevice (ships its own PortAudio wheel — no PyAudio needed, which is the usual
    pain point on Windows). Written with the stdlib `wave` module so there's no soundfile dep.
    """
    import tempfile
    import wave

    import numpy as np  # noqa: F401  (sounddevice returns a numpy array)
    import sounddevice as sd

    rec = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")  # start recording
    sd.wait()                                     # block until the recording finishes
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as w:              # write a standard 16-bit PCM mono WAV
        w.setnchannels(1)
        w.setsampwidth(2)                         # 2 bytes = int16
        w.setframerate(fs)
        w.writeframes(rec.tobytes())
    return path


def record_and_transcribe(seconds: float = 4.0, model_size: str = "base") -> str:
    """Capture `seconds` from the mic and return the transcription.

    Preferred path: sounddevice capture -> faster-whisper (offline, no PyAudio). Falls back to
    the SpeechRecognition mic path if sounddevice isn't available.
    """
    try:
        import sounddevice  # noqa: F401  (probe availability before recording)
        path = _record_wav_sounddevice(seconds)
        return transcribe(path, model_size)       # whisper (or SpeechRecognition file fallback)
    except Exception:
        return listen_mic(seconds)                # last resort: SpeechRecognition mic path


def listen_mic(seconds: float = 4.0) -> str:
    """Record `seconds` from the default microphone and transcribe it to text.

    Legacy path via SpeechRecognition + PyAudio. Prefer record_and_transcribe() (sounddevice),
    which doesn't need PyAudio.
    """
    try:
        import speech_recognition as sr
    except Exception:
        # faster-whisper can transcribe a file but doesn't capture the mic itself.
        raise RuntimeError(
            "Microphone capture needs sounddevice (recommended) or SpeechRecognition + PyAudio:\n"
            "  pip install sounddevice        # recommended, no PyAudio needed\n"
            "  pip install SpeechRecognition PyAudio"
        )
    r = sr.Recognizer()
    with sr.Microphone() as source:               # open the default mic
        audio = r.listen(source, timeout=seconds, phrase_time_limit=seconds)   # capture audio
    # Prefer offline whisper for the actual transcription if present.
    model = _get_whisper()
    if model is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:   # dump mic audio to a temp wav
            f.write(audio.get_wav_data())
            path = f.name
        return transcribe(path)                   # transcribe that file with whisper
    return r.recognize_google(audio)              # else fall back to online Google


def available_backend() -> Optional[str]:
    """Return the name of an installed backend, or None. Handy for UI/status."""
    if _get_whisper() is not None:
        return "faster-whisper"
    try:
        import speech_recognition  # noqa: F401
        return "SpeechRecognition"
    except Exception:
        return None
