from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import librosa
from faster_whisper import WhisperModel

from app.config import DEVICE, WHISPER_MODEL

WHISPER_SAMPLE_RATE = 16000

_model: WhisperModel | None = None


@dataclass
class TranscriptionResult:
    text: str
    language: str


def _cuda_available() -> bool:
    if "cuda" not in DEVICE:
        return False
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if os.name == "nt":
        # ctranslate2 needs cuBLAS/cuDNN DLLs on Windows; the CUDA build of torch ships them.
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    return True


def _cpu_model() -> WhisperModel:
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def get_whisper() -> WhisperModel:
    global _model
    if _model is None:
        if _cuda_available():
            try:
                _model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            except Exception as exc:
                print(f"Whisper could not start on CUDA ({exc}); falling back to CPU.")
        if _model is None:
            _model = _cpu_model()
    return _model


def _run(model: WhisperModel, audio, language: str | None):
    segments, info = model.transcribe(audio, language=language, beam_size=5, vad_filter=True)
    return list(segments), info


def transcribe(audio_path: str, language: str | None = None) -> TranscriptionResult:
    global _model
    # faster-whisper's own decoder (PyAV) breaks with av>=17, so feed it a 16 kHz array.
    audio, _ = librosa.load(audio_path, sr=WHISPER_SAMPLE_RATE, mono=True)
    model = get_whisper()
    try:
        segments, info = _run(model, audio, language)
    except RuntimeError as exc:
        if model.model.device != "cuda":
            raise
        # Missing CUDA libraries only surface on the first decode (e.g. cublas64_12.dll on Windows).
        print(f"Whisper failed on CUDA ({exc}); falling back to CPU.")
        _model = model = _cpu_model()
        segments, info = _run(model, audio, language)
    text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    return TranscriptionResult(text=text, language=(info.language or language or "en").lower())
