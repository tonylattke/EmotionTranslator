from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import soundfile as sf

from app.config import DATA_TMP, ensure_dirs

AudioInput = Union[str, Path, Tuple[int, np.ndarray], None]


def save_gradio_audio(audio: AudioInput, suffix: str = ".wav") -> Path:
    ensure_dirs()
    out = DATA_TMP / f"input_{uuid.uuid4().hex}{suffix}"

    if audio is None:
        raise ValueError("No audio provided")

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        sf.write(out, data, sr)
        return out

    if isinstance(audio, tuple) and len(audio) == 2:
        sr, data = audio
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        peak = np.max(np.abs(arr)) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak
        sf.write(out, arr, int(sr))
        return out

    raise TypeError(f"Unsupported audio type: {type(audio)!r}")
