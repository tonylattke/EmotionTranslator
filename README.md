# EmotionTranslator

Local Gradio app that **transcribes** your speech (file or microphone), **re-synthesizes** it in a target emotion using [OpenVoice V1](https://github.com/myshell-ai/OpenVoice), and optionally **preserves your voice timbre**.

Typical use: angry delivery → transcript → friendly/cheerful TTS → tone converter matched to your voice.

## Requirements

- Linux with **NVIDIA GPU** (CUDA) recommended
- **ffmpeg** (`sudo apt install ffmpeg`)
- **git**, **curl** or **wget**, **unzip**
- ~2 GB disk for Python packages + checkpoints

## Quick start

```bash
chmod +x build.sh run.sh "run:.sh"
./build.sh
./run.sh
```

If `pip install` times out on `files.pythonhosted.org`, `build.sh` auto-uses the Sportradar dependency proxy when reachable. You can also set:

```bash
export PIP_INDEX_URL="https://cdproxy.sportradar.online/pypi/simple/"
export PIP_TRUSTED_HOST="cdproxy.sportradar.online"
./build.sh
```

Open the URL printed in the terminal (default http://127.0.0.1:7860).

### UI won’t load (`TypeError: unhashable type: 'dict'`)

Newer `fastapi` / `starlette` (pulled in by other packages) break Gradio 3.48. Re-pin the web stack:

```bash
source .venv/bin/activate
pip install -r requirements-gradio-stack.txt
./run.sh
```

## How it works

1. Whisper (`faster-whisper`) transcribes input audio.
2. OpenVoice `BaseSpeakerTTS` generates speech in the chosen emotion style.
3. If “Keep speaker timbre” is enabled, `ToneColorConverter` clones timbre from your input clip.

Emotion is controlled by the TTS **style**, not by editing the original waveform.

## Configuration

Copy `.env.example` to `.env` (done automatically by `build.sh`):

| Variable | Default |
|----------|---------|
| `DEVICE` | `cuda:0` |
| `WHISPER_MODEL` | `base` (use `small` or `medium` for better ASR) |
| `GRADIO_SERVER_NAME` | `127.0.0.1` |
| `GRADIO_SERVER_PORT` | `7860` |

## Manual test checklist

1. Record a short phrase in an intense tone → target **friendly** → output should sound calmer with similar timbre.
2. Upload a WAV file with the same settings.
3. Disable timbre → hear base TTS emotion only.
4. Use transcript override if ASR fails on noisy audio.

## License & attribution

OpenVoice is MIT-licensed. V1 checkpoints are downloaded by `build.sh` from the [camenduru Hugging Face mirror](https://huggingface.co/camenduru/OpenVoice) when the official MyShell S3 URL is unavailable. See upstream README for watermark details (`@MyShell`).

This project is not affiliated with MyShell.
