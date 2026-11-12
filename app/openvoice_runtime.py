from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from app.config import CHECKPOINTS_DIR, DEVICE, assert_checkpoints
from app.styles import tts_language_label

_lock = threading.Lock()
_runtime: "OpenVoiceRuntime | None" = None


@dataclass
class OpenVoiceRuntime:
    device: str
    base_en: object
    base_zh: object
    converter: object

    def base_tts(self, language: str):
        label = tts_language_label(language)
        if label == "Chinese":
            return self.base_zh
        return self.base_en


def _resolve_device(requested: str) -> str:
    if "cuda" in requested and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        return "cpu"
    return requested


def get_runtime() -> OpenVoiceRuntime:
    global _runtime
    with _lock:
        if _runtime is not None:
            return _runtime
        assert_checkpoints()
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        device = _resolve_device(DEVICE)
        en_base = CHECKPOINTS_DIR / "base_speakers" / "EN"
        zh_base = CHECKPOINTS_DIR / "base_speakers" / "ZH"
        conv = CHECKPOINTS_DIR / "converter"

        base_en = BaseSpeakerTTS(str(en_base / "config.json"), device=device)
        base_en.load_ckpt(str(en_base / "checkpoint.pth"))

        base_zh = BaseSpeakerTTS(str(zh_base / "config.json"), device=device)
        base_zh.load_ckpt(str(zh_base / "checkpoint.pth"))

        converter = ToneColorConverter(str(conv / "config.json"), device=device)
        converter.load_ckpt(str(conv / "checkpoint.pth"))

        _runtime = OpenVoiceRuntime(
            device=device,
            base_en=base_en,
            base_zh=base_zh,
            converter=converter,
        )
        return _runtime
