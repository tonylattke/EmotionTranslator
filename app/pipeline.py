from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import librosa
import torch

from app import piper_tts
from app.asr import transcribe
from app.audio_io import AudioInput, save_gradio_audio
from app.config import DATA_OUT, PROCESSED_DIR, ensure_dirs
from app.languages import AUTO, SAME_AS_INPUT, require_supported
from app.openvoice_runtime import get_runtime
from app.styles import DEFAULT_TARGET_EMOTION, openvoice_language_name, source_se_path
from app.translate import translate

MAX_TEXT_CHARS = 500
MIN_REFERENCE_SECONDS = 1.0


def _speaker_embedding(runtime, input_path, logs: list[str]):
    from openvoice import se_extractor

    try:
        target_se, _ = se_extractor.get_se(
            str(input_path), runtime.converter, target_dir=str(PROCESSED_DIR), vad=True,
        )
        return target_se
    except AssertionError:
        # get_se needs ~5 s of voiced audio; short mic clips use the whole recording instead.
        if librosa.get_duration(path=str(input_path)) < MIN_REFERENCE_SECONDS:
            raise ValueError(
                "Recording is too short to clone the voice. Record at least a few seconds, "
                "or untick 'Keep speaker timbre'."
            )
        logs.append("Short reference: using the whole clip for the voice timbre.")
        return runtime.converter.extract_se([str(input_path)])


@dataclass
class RestyleResult:
    output_path: str
    transcript: str
    output_text: str
    input_language: str
    output_language: str
    target_emotion: str
    log: str


def restyle_audio(
    audio: AudioInput,
    target_emotion: str = DEFAULT_TARGET_EMOTION,
    keep_timbre: bool = True,
    input_language: str = AUTO,
    output_language: str = SAME_AS_INPUT,
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

    forced_input = None if input_language in (None, "", AUTO) else input_language
    if transcript_override and transcript_override.strip():
        text = transcript_override.strip()
        src_lang = forced_input or "en"
        logs.append(f"Using manual transcript ({src_lang}).")
    else:
        tr = transcribe(str(input_path), language=forced_input)
        text = tr.text
        src_lang = tr.language
        logs.append(f"ASR language: {src_lang}" + (" (forced)" if forced_input else " (detected)"))

    if not text:
        raise ValueError(
            "Empty transcript. Speak louder/longer, or type the text in the transcript override field."
        )
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        logs.append(f"Transcript truncated to {MAX_TEXT_CHARS} characters.")

    require_supported(src_lang)
    tgt_lang = src_lang if output_language in (None, "", SAME_AS_INPUT) else output_language
    tgt = require_supported(tgt_lang)

    output_text = translate(text, src_lang, tgt_lang)
    if tgt_lang != src_lang:
        logs.append(f"Translated {src_lang} -> {tgt_lang}: {output_text}")

    out_id = uuid.uuid4().hex
    styled_path = DATA_OUT / f"styled_{out_id}.wav"
    final_path = DATA_OUT / f"restyled_{out_id}.wav"

    emotion = target_emotion
    if tgt.tts_engine == "piper":
        style = piper_tts.synthesize(output_text, tgt_lang, emotion, styled_path, speed=speed)
        source_se = None
    else:
        if tgt_lang == "zh" and emotion != "default":
            logs.append("OpenVoice V1 has no emotion styles for Chinese; using default style.")
            emotion = "default"
        runtime.base_tts(tgt_lang).tts(
            output_text, str(styled_path), speaker=emotion,
            language=openvoice_language_name(tgt_lang), speed=speed,
        )
        style = f"openvoice speaker={emotion}"
        source_se = torch.load(str(source_se_path(tgt_lang, emotion)), map_location=runtime.device)
    logs.append(f"TTS {tgt.label}: {style}, speed={speed}")

    if keep_timbre:
        target_se = _speaker_embedding(runtime, input_path, logs)
        if source_se is None:
            source_se = runtime.converter.extract_se([str(styled_path)])
        runtime.converter.convert(
            audio_src_path=str(styled_path),
            src_se=source_se.to(runtime.device),
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

    logs.append(f"Done in {time.time() - t0:.1f}s")
    return RestyleResult(
        output_path=str(output),
        transcript=text,
        output_text=output_text,
        input_language=src_lang,
        output_language=tgt_lang,
        target_emotion=emotion,
        log="\n".join(logs),
    )
