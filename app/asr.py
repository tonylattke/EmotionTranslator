from __future__ import annotations

from dataclasses import dataclass

from faster_whisper import WhisperModel

from app.config import DEVICE, WHISPER_MODEL
from app.styles import tts_language_label

_model: WhisperModel | None = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    tts_language: str


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


def transcribe(audio_path: str) -> TranscriptionResult:
    model = get_whisper()
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    parts = [s.text.strip() for s in segments if s.text.strip()]
    text = " ".join(parts).strip()
    lang = (info.language or "en").lower()
    tts_lang = tts_language_label("zh" if lang.startswith("zh") else "en")
    return TranscriptionResult(text=text, language=lang, tts_language=tts_lang)
