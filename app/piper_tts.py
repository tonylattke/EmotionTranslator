from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.config import PIPER_VOICES_DIR

PIPER_VOICES = {
    "es": "es_ES-davefx-medium",
    "de": "de_DE-thorsten_emotional-medium",
}

# Speakers of the multi-speaker thorsten_emotional voice.
DE_EMOTION_SPEAKERS = {
    "default": "neutral",
    "friendly": "amused",
    "cheerful": "amused",
    "excited": "surprised",
    "sad": "sleepy",
    "angry": "angry",
    "terrified": "surprised",
    "shouting": "angry",
    "whispering": "whisper",
}


@dataclass(frozen=True)
class Prosody:
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    pitch_semitones: float = 0.0
    gain: float = 1.0


# Piper has no emotion styles for Spanish, so emotions are approximated with prosody.
ES_EMOTION_PROSODY = {
    "default": Prosody(),
    "friendly": Prosody(length_scale=1.05, noise_scale=0.6, pitch_semitones=1.0),
    "cheerful": Prosody(length_scale=0.95, noise_scale=0.75, noise_w_scale=0.9, pitch_semitones=2.0),
    "excited": Prosody(length_scale=0.88, noise_scale=0.8, noise_w_scale=1.0, pitch_semitones=3.0),
    "sad": Prosody(length_scale=1.2, noise_scale=0.4, noise_w_scale=0.5, pitch_semitones=-2.0, gain=0.8),
    "angry": Prosody(length_scale=0.92, noise_scale=0.8, noise_w_scale=1.0, pitch_semitones=-1.0, gain=1.2),
    "terrified": Prosody(length_scale=0.85, noise_scale=0.9, noise_w_scale=1.1, pitch_semitones=3.0),
    "shouting": Prosody(length_scale=0.95, noise_scale=0.8, noise_w_scale=1.0, pitch_semitones=1.5, gain=1.4),
    "whispering": Prosody(length_scale=1.1, noise_scale=0.3, noise_w_scale=0.4, gain=0.5),
}

_lock = threading.Lock()
_voices: dict[str, object] = {}


def voice_path(language: str) -> Path:
    return PIPER_VOICES_DIR / f"{PIPER_VOICES[language]}.onnx"


def assert_voice(language: str) -> None:
    path = voice_path(language)
    if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
        raise FileNotFoundError(f"Missing Piper voice {path.name}. Run build.sh (Linux) or build.bat (Windows) to download it.")


def _get_voice(language: str):
    with _lock:
        if language not in _voices:
            from piper import PiperVoice

            assert_voice(language)
            _voices[language] = PiperVoice.load(voice_path(language))
        return _voices[language]


def synthesize(text: str, language: str, emotion: str, output_path: Path, speed: float = 1.0) -> str:
    """Write speech to output_path and return a short description of the style used."""
    from piper import SynthesisConfig

    voice = _get_voice(language)
    if language == "de":
        speaker = DE_EMOTION_SPEAKERS.get(emotion, "neutral")
        prosody = Prosody()
        config = SynthesisConfig(
            speaker_id=voice.config.speaker_id_map[speaker],
            length_scale=prosody.length_scale / speed,
        )
        style = f"piper speaker={speaker}"
    else:
        prosody = ES_EMOTION_PROSODY.get(emotion, Prosody())
        config = SynthesisConfig(
            length_scale=prosody.length_scale / speed,
            noise_scale=prosody.noise_scale,
            noise_w_scale=prosody.noise_w_scale,
        )
        style = f"piper prosody={emotion}"

    with wave.open(str(output_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=config)

    if prosody.pitch_semitones or prosody.gain != 1.0:
        audio, sr = sf.read(output_path, dtype="float32")
        if prosody.pitch_semitones:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=prosody.pitch_semitones)
        audio = np.clip(audio * prosody.gain, -1.0, 1.0)
        sf.write(output_path, audio, sr)
    return style
