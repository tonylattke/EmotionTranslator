"""Cross-platform installer for EmotionTranslator (Linux and Windows).

Uses only the standard library so it runs on a bare Python before the venv exists.
Safe to re-run: every step skips work that is already done.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENDOR = ROOT / "vendor"
OPENVOICE = VENDOR / "OpenVoice"
TOOLS = ROOT / "tools"
PIPER_DIR = ROOT / "models" / "piper"

IS_WINDOWS = os.name == "nt"
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 12)  # torch 2.5.1 has no wheels for 3.13+

TORCH_VERSION = "2.5.1"
TORCH_INDEXES = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
CDPROXY_INDEX = "https://cdproxy.sportradar.online/pypi/simple/"

OPENVOICE_GIT = "https://github.com/myshell-ai/OpenVoice.git"
OPENVOICE_ZIP = "https://codeload.github.com/myshell-ai/OpenVoice/zip/refs/heads/main"
CHECKPOINT_URLS = [
    "https://huggingface.co/camenduru/OpenVoice/resolve/main/checkpoints_1226.zip",
    "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip",
]
CHECKPOINT_MARKER = OPENVOICE / "checkpoints" / "base_speakers" / "EN" / "checkpoint.pth"

PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_VOICES = [
    "es/es_ES/davefx/medium/es_ES-davefx-medium",
    "de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium",
]

FFMPEG_WINDOWS_ZIP = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

TRANSLATION_PAIRS = ["en-es", "es-en", "en-de", "de-en", "es-de", "de-es", "en-zh", "zh-en"]


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd: list, env: dict | None = None, retries: int = 1) -> None:
    for attempt in range(1, retries + 1):
        result = subprocess.run([str(c) for c in cmd], env=env, cwd=ROOT)
        if result.returncode == 0:
            return
        if attempt < retries:
            print(f"Command failed (attempt {attempt}/{retries}); retrying in 15s...", flush=True)
            time.sleep(15)
    fail(f"command failed: {' '.join(str(c) for c in cmd)}")


def download(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "EmotionTranslator-setup"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                last = 0.0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if total and time.time() - last > 2:
                        print(f"  {done * 100 // total}% of {total >> 20} MB", flush=True)
                        last = time.time()
            tmp.replace(dest)
            return True
        except Exception as exc:
            print(f"  download failed ({exc}); attempt {attempt}/{retries}", flush=True)
            tmp.unlink(missing_ok=True)
            time.sleep(5)
    return False


def extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def check_python() -> None:
    step(f"Python {platform.python_version()} ({sys.executable})")
    if not (MIN_PYTHON <= sys.version_info[:2] <= MAX_PYTHON):
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]} is required "
            f"(found {platform.python_version()}). Python 3.11 is recommended."
        )


def ensure_venv() -> None:
    if venv_python().is_file():
        return
    step("Creating virtualenv .venv")
    run([sys.executable, "-m", "venv", VENV])


def pip_env() -> dict:
    env = os.environ.copy()
    env.setdefault("PIP_DEFAULT_TIMEOUT", "300")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if "PIP_INDEX_URL" not in env:
        try:
            urllib.request.urlopen(CDPROXY_INDEX + "pip/", timeout=5).close()
            env["PIP_INDEX_URL"] = CDPROXY_INDEX
            env["PIP_TRUSTED_HOST"] = "cdproxy.sportradar.online"
            print(f"Using pip index {CDPROXY_INDEX}")
        except Exception:
            print("Using default PyPI index (set PIP_INDEX_URL if downloads fail).")
    return env


def pip(env: dict, *args) -> None:
    run([venv_python(), "-m", "pip", "install", *args], env=env, retries=5)


def ensure_ffmpeg() -> None:
    step("ffmpeg")
    local_bin = TOOLS / "ffmpeg" / "bin"
    if shutil.which("ffmpeg") or (local_bin / "ffmpeg.exe").is_file():
        print("ffmpeg found.")
        return
    if not IS_WINDOWS:
        fail("ffmpeg is required. Install it with: sudo apt install ffmpeg")
    install_ffmpeg_windows(TOOLS)


def install_ffmpeg_windows(tools_dir: Path) -> None:
    """Download a portable ffmpeg build into tools_dir/ffmpeg (bin/ffmpeg.exe, bin/ffprobe.exe)."""
    archive = tools_dir / "ffmpeg.zip"
    if not download(FFMPEG_WINDOWS_ZIP, archive):
        fail("could not download ffmpeg. Install it manually (winget install Gyan.FFmpeg).")
    with tempfile.TemporaryDirectory(dir=tools_dir) as tmp:
        extract_zip(archive, Path(tmp))
        (build_dir,) = [p for p in Path(tmp).iterdir() if p.is_dir()]
        target = tools_dir / "ffmpeg"
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(build_dir), target)
    archive.unlink(missing_ok=True)
    print(f"ffmpeg installed in {tools_dir / 'ffmpeg' / 'bin'}")


def ensure_openvoice() -> None:
    if (OPENVOICE / "setup.py").is_file():
        return
    step("OpenVoice source")
    if shutil.which("git"):
        run(["git", "clone", "--depth", "1", OPENVOICE_GIT, OPENVOICE])
    else:
        fetch_openvoice_zip(OPENVOICE)


def fetch_openvoice_zip(dest: Path) -> None:
    archive = dest.parent / "OpenVoice.zip"
    if not download(OPENVOICE_ZIP, archive):
        fail("could not download OpenVoice from GitHub.")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        extract_zip(archive, Path(tmp))
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(Path(tmp) / "OpenVoice-main"), dest)
    archive.unlink(missing_ok=True)


def torch_variant() -> str:
    forced = os.environ.get("TORCH_VARIANT", "").lower()
    if forced:
        if forced not in TORCH_INDEXES:
            fail(f"TORCH_VARIANT must be one of {', '.join(TORCH_INDEXES)}")
        return forced
    return "cu121" if shutil.which("nvidia-smi") else "cpu"


def install_python_deps(env: dict) -> None:
    variant = torch_variant()
    step(f"PyTorch {TORCH_VERSION} ({variant})")
    specs = [f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"]
    # Large wheels come straight from the PyTorch CDN; the second call is already satisfied
    # for torch itself and only resolves its dependencies through the normal index.
    pip(env, *specs, "--no-deps", "--index-url", TORCH_INDEXES[variant])
    pip(env, *specs)

    step("OpenVoice dependencies")
    pip(env, "-r", ROOT / "requirements-openvoice.txt")
    pip(env, "-e", OPENVOICE, "--no-deps")

    step("App dependencies")
    pip(env, "-r", ROOT / "requirements-app.txt")

    step("Gradio 3.48-compatible web stack")
    pip(env, "-r", ROOT / "requirements-gradio-stack.txt")


def ensure_checkpoints() -> None:
    if CHECKPOINT_MARKER.is_file():
        return
    step("OpenVoice V1 checkpoints (~400 MB)")
    archive = VENDOR / "checkpoints_1226.zip"
    for url in CHECKPOINT_URLS:
        if download(url, archive) and zipfile.is_zipfile(archive):
            extract_zip(archive, OPENVOICE)
            archive.unlink(missing_ok=True)
            return
        archive.unlink(missing_ok=True)
    fail("checkpoint download failed. Place checkpoints under vendor/OpenVoice/checkpoints manually.")


def ensure_piper_voices() -> None:
    step("Piper voices (Español, Deutsch)")
    for voice in PIPER_VOICES:
        name = voice.rsplit("/", 1)[-1]
        for ext in (".onnx", ".onnx.json"):
            target = PIPER_DIR / f"{name}{ext}"
            if target.is_file() and target.stat().st_size > 0:
                continue
            if not download(f"{PIPER_BASE}/{voice}{ext}?download=true", target):
                fail(f"could not download Piper voice {name}{ext}")
    print("Piper voices ready.")


def preload_translation_models() -> None:
    if os.environ.get("PRELOAD_TRANSLATION_MODELS") != "1":
        return
    step("Translation models (~300 MB each)")
    code = (
        "from transformers import MarianMTModel, MarianTokenizer\n"
        f"for pair in {TRANSLATION_PAIRS!r}:\n"
        "    name = f'Helsinki-NLP/opus-mt-{pair}'\n"
        "    MarianTokenizer.from_pretrained(name); MarianMTModel.from_pretrained(name)\n"
        "    print('cached', name)\n"
    )
    run([venv_python(), "-c", code])


def ensure_env_file() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        shutil.copyfile(ROOT / ".env.example", env_file)


def verify() -> None:
    step("Verifying installation")
    run([
        venv_python(), "-c",
        "import torch, openvoice, gradio, piper, transformers, faster_whisper; "
        "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())",
    ])
    missing = [CHECKPOINT_MARKER] + [
        PIPER_DIR / f"{v.rsplit('/', 1)[-1]}.onnx" for v in PIPER_VOICES
    ]
    missing = [p for p in missing if not p.is_file()]
    if missing:
        fail("missing model files: " + ", ".join(str(p) for p in missing))


def main() -> None:
    print("EmotionTranslator setup", f"({platform.system()} {platform.machine()})")
    check_python()
    ensure_ffmpeg()
    ensure_venv()
    env = pip_env()
    ensure_openvoice()
    install_python_deps(env)
    ensure_checkpoints()
    ensure_piper_voices()
    preload_translation_models()
    ensure_env_file()
    verify()
    launcher = "run.bat" if IS_WINDOWS else "./run.sh"
    print(f"\n==> Build complete. Start the app with: {launcher}")


if __name__ == "__main__":
    main()
