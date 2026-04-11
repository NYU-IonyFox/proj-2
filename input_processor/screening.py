"""
input_processor/screening.py
L1 Screening Layer — input validation and language detection.

GLOBAL CONSTRAINTS:
- No external API calls. Language detection is local (langdetect).
- Fail-closed: unknown language returns "unknown"; any langdetect exception returns "unknown".
- Non-discrimination: language codes are technical identifiers only.
"""
from __future__ import annotations

import re
import sys

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
# Multilingual tag → NLLB code mapping
# ---------------------------------------------------------------------------

LANG_TAG_TO_NLLB: dict[str, str] = {
    "EN": "eng_Latn",
    "FR": "fra_Latn",
    "ZH": "zho_Hans",
    "AR": "arb_Arab",
    "RU": "rus_Cyrl",
    "ES": "spa_Latn",
    "HI": "hin_Deva",
    "PT": "por_Latn",
    "SW": "swh_Latn",
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


# Tag detection: matches [XX] on its own line (exactly 2 uppercase letters)
_TAG_RE = re.compile(r"^\[([A-Z]{2})\]\s*$", re.MULTILINE)


def parse_multilingual_input(text: str) -> dict[str, str] | None:
    """
    Detect language tags in the format [XX] where XX is a 2-letter uppercase
    language code (e.g. [EN], [FR], [ZH]).

    A valid tag appears on its own line, contains only 2 uppercase letters,
    and is followed by non-empty text before the next tag or end of string.

    Returns a dict mapping each known 2-letter code to its text segment
    (stripped), or None if no valid known tags with non-empty content found.
    Tags not in LANG_TAG_TO_NLLB are logged as warnings and skipped.
    """
    matches = list(_TAG_RE.finditer(text))
    if not matches:
        return None

    result: dict[str, str] = {}
    for i, match in enumerate(matches):
        tag = match.group(1)
        seg_start = match.end()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[seg_start:seg_end].strip()

        if not segment:
            continue

        if tag not in LANG_TAG_TO_NLLB:
            print(
                f"[parse_multilingual_input] Unknown tag '[{tag}]' — skipping segment.",
                file=sys.stderr,
            )
            continue

        result[tag] = segment

    return result if result else None
