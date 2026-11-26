from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

# Portable ffmpeg downloaded by the Windows setup; pydub and whisper_timestamped call it by name.
_LOCAL_FFMPEG = _ROOT / "tools" / "ffmpeg" / "bin"
if _LOCAL_FFMPEG.is_dir():
    os.environ["PATH"] = str(_LOCAL_FFMPEG) + os.pathsep + os.environ.get("PATH", "")

# Gradio checks http://127.0.0.1 at startup; a corporate proxy must not intercept it.
for _var in ("NO_PROXY", "no_proxy"):
    _hosts = [h for h in os.environ.get(_var, "").split(",") if h]
    os.environ[_var] = ",".join(dict.fromkeys(["127.0.0.1", "localhost", *_hosts]))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

OPENVOICE_ROOT = Path(
    os.environ.get("OPENVOICE_ROOT", _ROOT / "vendor" / "OpenVoice")
).resolve()
CHECKPOINTS_DIR = Path(
    os.environ.get("CHECKPOINTS_DIR", OPENVOICE_ROOT / "checkpoints")
).resolve()
DATA_TMP = _ROOT / "data" / "tmp"
DATA_OUT = _ROOT / "data" / "out"
PROCESSED_DIR = OPENVOICE_ROOT / "processed"
PIPER_VOICES_DIR = Path(os.environ.get("PIPER_VOICES_DIR", _ROOT / "models" / "piper")).resolve()

DEVICE = os.environ.get("DEVICE", "cuda:0")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

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
            f"Missing OpenVoice EN checkpoint at {en_ckpt}. Run build.sh (Linux) or build.bat (Windows) to download models."
        )
    if not converter_ckpt.is_file():
        raise FileNotFoundError(
            f"Missing OpenVoice converter at {converter_ckpt}. Run build.sh (Linux) or build.bat (Windows) to download models."
        )
