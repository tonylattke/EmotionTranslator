from __future__ import annotations

from dataclasses import dataclass

import librosa
from faster_whisper import WhisperModel

from app.config import DEVICE, WHISPER_MODEL

WHISPER_SAMPLE_RATE = 16000

_model: WhisperModel | None = None


@dataclass
class TranscriptionResult:
    text: str
    language: str


def _whisper_device() -> tuple[str, str]:
    if "cuda" in DEVICE:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "float16"
        except Exception:
            pass
    return "cpu", "int8"


def get_whisper() -> WhisperModel:
    global _model
    if _model is None:
        device, compute = _whisper_device()
        _model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    return _model


def transcribe(audio_path: str, language: str | None = None) -> TranscriptionResult:
    # faster-whisper's own decoder (PyAV) breaks with av>=17, so feed it a 16 kHz array.
    audio, _ = librosa.load(audio_path, sr=WHISPER_SAMPLE_RATE, mono=True)
    model = get_whisper()
    segments, info = model.transcribe(audio, language=language, beam_size=5, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    return TranscriptionResult(text=text, language=(info.language or language or "en").lower())
