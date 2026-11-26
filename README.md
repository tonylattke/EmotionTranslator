# EmotionTranslator

Local Gradio app that **transcribes** your speech (file or microphone), **re-synthesizes** it in a target emotion using [OpenVoice V1](https://github.com/myshell-ai/OpenVoice), and optionally **preserves your voice timbre**.

Typical use: angry delivery → transcript → friendly/cheerful TTS → tone converter matched to your voice.

## Requirements

- Linux or Windows 10/11 (64-bit)
- **Python 3.9–3.12** (3.11 recommended; 3.13+ is not supported by PyTorch 2.5)
- **NVIDIA GPU** with a recent driver recommended; without one everything runs on CPU (slower)
- ~6 GB disk for Python packages and models
- Linux only: **ffmpeg** (`sudo apt install ffmpeg`). On Windows the build downloads a portable ffmpeg into `tools/ffmpeg` if none is installed.
- `git` is optional: without it, OpenVoice is downloaded as a zip.

## Quick start

Linux:

```bash
./build.sh
./run.sh
```

Windows (double-click or from `cmd` / PowerShell in the project folder):

```bat
build.bat
run.bat
```

If Python is missing on Windows: `winget install Python.Python.3.11`, then open a new terminal.

Both build scripts run the same installer, [`scripts/setup_env.py`](scripts/setup_env.py). It is safe to re-run. It creates `.venv`, installs PyTorch (CUDA 12.1 wheels when `nvidia-smi` is found, CPU wheels otherwise), the app dependencies, the OpenVoice checkpoints and the Piper voices. Options (environment variables):

| Variable | Effect |
|----------|--------|
| `TORCH_VARIANT=cpu` or `cu121` | Force the PyTorch build |
| `PRELOAD_TRANSLATION_MODELS=1` | Download all translation models now, for fully offline use |
| `PIP_INDEX_URL` | Custom pip index |

If `pip install` times out on `files.pythonhosted.org`, the installer automatically uses the Sportradar dependency proxy when it is reachable.

Open the URL printed in the terminal (default http://127.0.0.1:7860).

### UI won’t load (`TypeError: unhashable type: 'dict'`)

Newer `fastapi` / `starlette` (pulled in by other packages) break Gradio 3.48. Run the build again, which re-pins the web stack (`requirements-gradio-stack.txt`).

### Windows notes

- `run.bat` starts Python in UTF-8 mode (`-X utf8`); OpenVoice prints phonetic symbols that crash the default Windows console encoding.
- Whisper uses the CUDA libraries bundled with PyTorch. If they cannot be loaded, it falls back to CPU and prints a warning.

### Microphone

Click **Record from microphone**, speak, click **Stop recording**, and wait for the player to appear before pressing **Translate emotion**. The browser only allows the microphone on `http://127.0.0.1` / `localhost` or HTTPS, so open the app through one of those (not through the machine's LAN IP).

## How it works

1. Whisper (`faster-whisper`) transcribes the input and detects the language (or uses the one you pick).
2. If the output language differs, the text is translated locally with [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP) models (downloaded on first use, ~300 MB per language pair).
3. Speech is generated in the target emotion and language:

| Output | TTS | Emotion |
|--------|-----|---------|
| English | OpenVoice V1 | Native styles (friendly, cheerful, angry, ...) |
| Deutsch | Piper `thorsten_emotional` | Native speakers (amused, angry, sleepy, whisper, ...) mapped from the emotion list |
| Español | Piper `davefx` | Approximated with speed, pitch and expressiveness |
| 中文 | OpenVoice V1 | Default style only |

4. If “Keep speaker timbre” is enabled, OpenVoice's `ToneColorConverter` gives the output your voice. Clips shorter than ~5 s use the whole recording as reference.

Emotion is controlled by the TTS **style**, not by editing the original waveform. To have every translation model available offline, run `PRELOAD_TRANSLATION_MODELS=1 ./build.sh`.

## Configuration

Copy `.env.example` to `.env` (done automatically by `build.sh` / `build.bat`):

| Variable | Default |
|----------|---------|
| `DEVICE` | `cuda:0` |
| `WHISPER_MODEL` | `small` (`medium` recognizes Spanish/German better but is slower) |
| `PIPER_VOICES_DIR` | `models/piper` |
| `GRADIO_SERVER_NAME` | `127.0.0.1` |
| `GRADIO_SERVER_PORT` | `7860` |

## Manual test checklist

1. Record a short phrase in an intense tone → target **friendly** → output should sound calmer with similar timbre.
2. Upload a WAV file with the same settings.
3. Disable timbre → hear base TTS emotion only.
4. Use transcript override if ASR fails on noisy audio.

## License & attribution

OpenVoice is MIT-licensed. V1 checkpoints are downloaded by the build scripts from the [camenduru Hugging Face mirror](https://huggingface.co/camenduru/OpenVoice) when the official MyShell S3 URL is unavailable. See upstream README for watermark details (`@MyShell`).

This project is not affiliated with MyShell.
