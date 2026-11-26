from __future__ import annotations

import argparse
import traceback

# app.config must load before gradio: it puts the local ffmpeg on PATH, which pydub reads at import.
from app.config import (
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    assert_checkpoints,
    ensure_dirs,
)

import gradio as gr

from app.languages import INPUT_LANGUAGE_CHOICES, OUTPUT_LANGUAGE_CHOICES, to_code
from app.openvoice_runtime import get_runtime
from app.pipeline import restyle_audio
from app.styles import DEFAULT_TARGET_EMOTION, EN_EMOTION_STYLES

INPUT_LABELS = [label for label, _ in INPUT_LANGUAGE_CHOICES]
OUTPUT_LABELS = [label for label, _ in OUTPUT_LANGUAGE_CHOICES]


def _on_submit(
    active_source,
    audio_upload,
    audio_mic,
    target_emotion,
    keep_timbre,
    input_language,
    output_language,
    speed,
    tau,
    transcript_override,
):
    if active_source == "microphone":
        audio = audio_mic or audio_upload
    else:
        audio = audio_upload or audio_mic
    if audio is None:
        raise gr.Error("Please record or upload audio first.")
    try:
        result = restyle_audio(
            audio,
            target_emotion=target_emotion,
            keep_timbre=keep_timbre,
            input_language=to_code(input_language),
            output_language=to_code(output_language),
            speed=speed,
            tau=tau,
            transcript_override=transcript_override,
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    return result.output_path, result.transcript, result.output_text, result.log


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EmotionTranslator") as demo:
        gr.Markdown(
            """
            # EmotionTranslator
            Upload or record speech, then re-synthesize it in a **target emotion** and
            **output language** (English, Español, Deutsch, 中文). Optionally keep the
            original speaker timbre.
            """
        )
        active_source = gr.State("upload")
        with gr.Tabs():
            with gr.Tab("Upload") as upload_tab:
                audio_upload = gr.Audio(label="Audio file", source="upload", type="filepath")
            with gr.Tab("Microphone") as mic_tab:
                audio_mic = gr.Audio(label="Microphone", source="microphone", type="filepath")
                gr.Markdown(
                    "Click **Record from microphone**, speak, then **Stop recording** and wait "
                    "until the player appears before translating."
                )
        upload_tab.select(lambda: "upload", None, active_source)
        mic_tab.select(lambda: "microphone", None, active_source)

        with gr.Row():
            input_language = gr.Dropdown(choices=INPUT_LABELS, value="Auto", label="Input language")
            output_language = gr.Dropdown(
                choices=OUTPUT_LABELS, value="Same as input", label="Output language"
            )
            target_emotion = gr.Dropdown(
                choices=list(EN_EMOTION_STYLES),
                value=DEFAULT_TARGET_EMOTION,
                label="Target emotion",
            )
        with gr.Row():
            keep_timbre = gr.Checkbox(value=True, label="Keep speaker timbre from input")
            speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Speech speed")
            tau = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Timbre strength (tau)")
        transcript_override = gr.Textbox(
            label="Transcript override (optional)",
            placeholder="Leave empty to auto-transcribe with Whisper",
            lines=2,
        )
        run_btn = gr.Button("Translate emotion", variant="primary")
        audio_out = gr.Audio(label="Output", type="filepath")
        with gr.Row():
            transcript_out = gr.Textbox(label="Recognized text", lines=3)
            output_text = gr.Textbox(label="Spoken text (output language)", lines=3)
        log_out = gr.Textbox(label="Log", lines=8)

        run_btn.click(
            fn=_on_submit,
            inputs=[
                active_source,
                audio_upload,
                audio_mic,
                target_emotion,
                keep_timbre,
                input_language,
                output_language,
                speed,
                tau,
                transcript_override,
            ],
            outputs=[audio_out, transcript_out, output_text, log_out],
        )
    return demo


def _ensure_gradio_web_stack() -> None:
    """Fail fast with a clear message if incompatible FastAPI/Starlette are installed."""
    import starlette

    major, minor = (int(x) for x in starlette.__version__.split(".")[:2])
    if major > 0 or minor >= 30:
        raise RuntimeError(
            f"Incompatible starlette {starlette.__version__} for Gradio 3.48. "
            "Run: pip install -r requirements-gradio-stack.txt"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()
    _ensure_gradio_web_stack()
    ensure_dirs()
    assert_checkpoints()
    get_runtime()
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
