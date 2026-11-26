from __future__ import annotations

import threading

import torch

# Helsinki-NLP MarianMT pairs that exist on the Hugging Face Hub; others pivot through English.
_DIRECT_PAIRS = {
    ("en", "es"), ("es", "en"),
    ("en", "de"), ("de", "en"),
    ("es", "de"), ("de", "es"),
    ("en", "zh"), ("zh", "en"),
}

_lock = threading.Lock()
_models: dict[tuple[str, str], tuple[object, object]] = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(src: str, tgt: str):
    key = (src, tgt)
    with _lock:
        if key not in _models:
            from transformers import MarianMTModel, MarianTokenizer

            name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            tokenizer = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name).to(_device()).eval()
            _models[key] = (tokenizer, model)
        return _models[key]


def _translate_direct(text: str, src: str, tgt: str) -> str:
    tokenizer, model = _load(src, tgt)
    batch = tokenizer([text], return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=512)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt or not text:
        return text
    if (src, tgt) in _DIRECT_PAIRS:
        return _translate_direct(text, src, tgt)
    return _translate_direct(_translate_direct(text, src, "en"), "en", tgt)
