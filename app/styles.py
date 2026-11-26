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


def openvoice_base_dir(language: str) -> Path:
    return CHECKPOINTS_DIR / "base_speakers" / ("ZH" if language == "zh" else "EN")


def source_se_path(language: str, speaker: str) -> Path:
    """Precomputed tone embedding of the OpenVoice base speaker that produced the TTS audio."""
    base = openvoice_base_dir(language)
    prefix = "zh" if language == "zh" else "en"
    if speaker == "default":
        return base / f"{prefix}_default_se.pth"
    return base / f"{prefix}_style_se.pth"


def openvoice_language_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"
