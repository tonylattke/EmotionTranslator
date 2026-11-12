from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

OPENVOICE_ROOT = Path(
    os.environ.get("OPENVOICE_ROOT", _ROOT / "vendor" / "OpenVoice")
).resolve()
CHECKPOINTS_DIR = Path(
    os.environ.get("CHECKPOINTS_DIR", OPENVOICE_ROOT / "checkpoints")
).resolve()
DATA_TMP = _ROOT / "data" / "tmp"
DATA_OUT = _ROOT / "data" / "out"
PROCESSED_DIR = OPENVOICE_ROOT / "processed"

DEVICE = os.environ.get("DEVICE", "cuda:0")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))


def ensure_dirs() -> None:
    DATA_TMP.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def assert_checkpoints() -> None:
    en_ckpt = CHECKPOINTS_DIR / "base_speakers" / "EN" / "checkpoint.pth"
    converter_ckpt = CHECKPOINTS_DIR / "converter" / "checkpoint.pth"
    if not en_ckpt.is_file():
        raise FileNotFoundError(
            f"Missing OpenVoice EN checkpoint at {en_ckpt}. Run ./build.sh to download models."
        )
    if not converter_ckpt.is_file():
        raise FileNotFoundError(
            f"Missing OpenVoice converter at {converter_ckpt}. Run ./build.sh to download models."
        )
