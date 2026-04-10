"""
input_processor/screening.py
L1 Screening Layer — input validation and language detection.

GLOBAL CONSTRAINTS:
- No external API calls. Language detection is local (langdetect).
- Fail-closed: unknown language returns "unknown"; any langdetect exception returns "unknown".
- Non-discrimination: language codes are technical identifiers only.
"""
from __future__ import annotations

from langdetect import detect, LangDetectException

# ---------------------------------------------------------------------------
# NLLB BCP-47 language code mapping
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, str] = {
    "en": "eng_Latn",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "hi": "hin_Deva",
    "pt": "por_Latn",
    "sw": "swh_Latn",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_input(text: str) -> None:
    """
    Validate that text is a non-empty string within the 50000-character limit
    and contains only valid UTF-8 content.

    Raises ValueError on any violation.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    if not text:
        raise ValueError("Input text must not be empty.")
    if len(text) > 50000:
        raise ValueError(
            f"Input text exceeds maximum length of 50000 characters "
            f"(got {len(text)})."
        )
    # Verify valid UTF-8: encode then decode round-trip
    try:
        text.encode("utf-8").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Input text contains invalid UTF-8 content: {exc}") from exc


def detect_language(text: str) -> str:
    """
    Detect the language of text and return its NLLB BCP-47 code.

    Returns "unknown" if langdetect raises an exception or the detected
    language code is not in the mapping table.
    """
    try:
        lang_code = detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"

    return _LANG_MAP.get(lang_code, "unknown")
