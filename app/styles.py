from __future__ import annotations

from pathlib import Path

from app.config import CHECKPOINTS_DIR

EN_EMOTION_STYLES = (
    "default",
    "friendly",
    "cheerful",
    "excited",
    "sad",
    "angry",
    "terrified",
    "shouting",
    "whispering",
)

DEFAULT_TARGET_EMOTION = "friendly"


def en_checkpoint_base() -> Path:
    return CHECKPOINTS_DIR / "base_speakers" / "EN"


def zh_checkpoint_base() -> Path:
    return CHECKPOINTS_DIR / "base_speakers" / "ZH"


def source_se_path(language: str, speaker: str) -> Path:
    lang = language.lower()
    if lang in ("chinese", "zh"):
        base = zh_checkpoint_base()
        if speaker == "default":
            return base / "zh_default_se.pth"
        return base / "zh_style_se.pth"
    base = en_checkpoint_base()
    if speaker == "default":
        return base / "en_default_se.pth"
    return base / "en_style_se.pth"


def tts_language_label(language: str) -> str:
    lang = language.lower()
    if lang in ("chinese", "zh", "mandarin"):
        return "Chinese"
    return "English"
