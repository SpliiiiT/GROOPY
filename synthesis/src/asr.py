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


def _get_whisper(model_size: str = "base"):
    """Lazily construct a cached faster-whisper model, or return None if unavailable."""
    global _WHISPER
    if _WHISPER is not None:                      # already built -> reuse
        return _WHISPER
    try:
        from faster_whisper import WhisperModel   # optional dep
    except Exception:
        return None                               # not installed -> signal "no whisper"
    _WHISPER = WhisperModel(model_size, device="cpu", compute_type="int8")   # int8 = quantised, fast on CPU
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


def listen_mic(seconds: float = 4.0) -> str:
    """Record `seconds` from the default microphone and transcribe it to text."""
    try:
        import speech_recognition as sr
    except Exception:
        # faster-whisper can transcribe a file but doesn't capture the mic itself.
        raise RuntimeError(
            "Microphone capture needs SpeechRecognition + PyAudio:\n"
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
