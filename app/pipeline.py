from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

from app.asr import transcribe
from app.audio_io import save_gradio_audio, AudioInput
from app.config import DATA_OUT, PROCESSED_DIR, ensure_dirs
from app.openvoice_runtime import get_runtime
from app.styles import DEFAULT_TARGET_EMOTION, source_se_path


@dataclass
class RestyleResult:
    output_path: str
    transcript: str
    language: str
    tts_language: str
    target_emotion: str
    log: str


def restyle_audio(
    audio: AudioInput,
    target_emotion: str = DEFAULT_TARGET_EMOTION,
    keep_timbre: bool = True,
    language_override: str | None = None,
    speed: float = 1.0,
    tau: float = 0.3,
    transcript_override: str | None = None,
) -> RestyleResult:
    ensure_dirs()
    t0 = time.time()
    logs: list[str] = []

    input_path = save_gradio_audio(audio)
    logs.append(f"Saved input: {input_path.name}")

    runtime = get_runtime()
    logs.append(f"OpenVoice device: {runtime.device}")

    if transcript_override and transcript_override.strip():
        text = transcript_override.strip()
        tts_language = language_override or "English"
        lang = "en"
        logs.append("Using manual transcript override.")
    else:
        tr = transcribe(str(input_path))
        text = tr.text
        lang = tr.language
        tts_language = language_override or tr.tts_language
        logs.append(f"ASR language: {lang}")

    if not text:
        raise ValueError(
            "Empty transcript. Speak louder/longer, or type the text in the transcript override field."
        )

    if len(text) > 500:
        text = text[:500]
        logs.append("Transcript truncated to 500 characters.")

    if tts_language == "Chinese" and target_emotion != "default":
        logs.append(
            "Note: OpenVoice V1 only supports emotion styles for English; using default style for Chinese."
        )
        speaker = "default"
    else:
        speaker = target_emotion

    out_id = uuid.uuid4().hex
    styled_path = DATA_OUT / f"styled_{out_id}.wav"
    final_path = DATA_OUT / f"restyled_{out_id}.wav"

    base_tts = runtime.base_tts(tts_language)
    base_tts.tts(text, str(styled_path), speaker=speaker, language=tts_language, speed=speed)
    logs.append(f"TTS style={speaker}, speed={speed}")

    if keep_timbre:
        from openvoice import se_extractor

        target_se, _ = se_extractor.get_se(
            str(input_path),
            runtime.converter,
            target_dir=str(PROCESSED_DIR),
            vad=True,
        )
        se_path = source_se_path(tts_language, speaker)
        source_se = torch.load(str(se_path), map_location=runtime.device).to(
            runtime.device
        )
        runtime.converter.convert(
            audio_src_path=str(styled_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(final_path),
            tau=tau,
            message="@MyShell",
        )
        logs.append(f"Tone conversion tau={tau}")
        output = final_path
    else:
        output = styled_path
        logs.append("Skipped timbre conversion (base TTS only).")

    elapsed = time.time() - t0
    logs.append(f"Done in {elapsed:.1f}s")

    return RestyleResult(
        output_path=str(output),
        transcript=text,
        language=lang,
        tts_language=tts_language,
        target_emotion=speaker,
        log="\n".join(logs),
    )
