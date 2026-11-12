from __future__ import annotations

import argparse

import gradio as gr

from app.config import (
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    assert_checkpoints,
    ensure_dirs,
)
from app.openvoice_runtime import get_runtime
from app.pipeline import restyle_audio
from app.styles import DEFAULT_TARGET_EMOTION, EN_EMOTION_STYLES


def _on_submit(
    audio_upload,
    audio_mic,
    target_emotion,
    keep_timbre,
    language_override,
    speed,
    tau,
    transcript_override,
):
    audio = audio_upload or audio_mic
    if audio is None:
        raise gr.Error("Please record or upload audio first.")
    lo = (language_override or "Auto").strip()
    lang = None if lo in ("", "Auto") else lo
    result = restyle_audio(
        audio,
        target_emotion=target_emotion,
        keep_timbre=keep_timbre,
        language_override=lang,
        speed=speed,
        tau=tau,
        transcript_override=transcript_override,
    )
    return result.output_path, result.transcript, result.log


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EmotionTranslator") as demo:
        gr.Markdown(
            """
            # EmotionTranslator
            Upload or record speech, then re-synthesize it in a **target emotion**
            (OpenVoice V1). Optionally keep the original speaker timbre.
            """
        )
        with gr.Tabs():
            with gr.TabItem("Upload"):
                audio_upload = gr.Audio(label="Audio file", source="upload", type="filepath")
            with gr.TabItem("Microphone"):
                audio_mic = gr.Audio(label="Microphone", source="microphone", type="filepath")
        with gr.Row():
            target_emotion = gr.Dropdown(
                choices=list(EN_EMOTION_STYLES),
                value=DEFAULT_TARGET_EMOTION,
                label="Target emotion",
            )
            language_override = gr.Dropdown(
                choices=["Auto", "English", "Chinese"],
                value="Auto",
                label="TTS language",
            )
            keep_timbre = gr.Checkbox(value=True, label="Keep speaker timbre from input")
        with gr.Row():
            speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Speech speed")
            tau = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Timbre strength (tau)")
        transcript_override = gr.Textbox(
            label="Transcript override (optional)",
            placeholder="Leave empty to auto-transcribe with Whisper",
            lines=2,
        )
        run_btn = gr.Button("Translate emotion", variant="primary")
        audio_out = gr.Audio(label="Output", type="filepath")
        transcript_out = gr.Textbox(label="Transcript used", lines=3)
        log_out = gr.Textbox(label="Log", lines=6)

        run_btn.click(
            fn=_on_submit,
            inputs=[
                audio_upload,
                audio_mic,
                target_emotion,
                keep_timbre,
                language_override,
                speed,
                tau,
                transcript_override,
            ],
            outputs=[audio_out, transcript_out, log_out],
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
