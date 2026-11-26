from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    label: str
    tts_engine: str


LANGUAGES: dict[str, Language] = {
    "en": Language("en", "English", "openvoice"),
    "es": Language("es", "Español", "piper"),
    "de": Language("de", "Deutsch", "piper"),
    "zh": Language("zh", "中文", "openvoice"),
}

AUTO = "auto"
SAME_AS_INPUT = "same"

INPUT_LANGUAGE_CHOICES = [("Auto", AUTO)] + [(lang.label, code) for code, lang in LANGUAGES.items()]
OUTPUT_LANGUAGE_CHOICES = [("Same as input", SAME_AS_INPUT)] + [
    (lang.label, code) for code, lang in LANGUAGES.items()
]

_LABEL_TO_CODE = {label: code for label, code in INPUT_LANGUAGE_CHOICES + OUTPUT_LANGUAGE_CHOICES}


def to_code(value: str | None) -> str | None:
    """Accept a code ("es") or a UI label ("Español"); None/'' stays None."""
    if not value:
        return None
    value = value.strip()
    if value in _LABEL_TO_CODE:
        return _LABEL_TO_CODE[value]
    return value.lower()


def require_supported(code: str) -> Language:
    if code not in LANGUAGES:
        supported = ", ".join(f"{lang.label} ({c})" for c, lang in LANGUAGES.items())
        raise ValueError(
            f"Language '{code}' is not supported. Pick the input language manually. Supported: {supported}."
        )
    return LANGUAGES[code]
