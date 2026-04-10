"""
input_processor/multilingual.py
L2 Multilingual Support Layer — NLLB translation, confidence scoring,
bundle construction, and bundle status assessment.

GLOBAL CONSTRAINTS:
- No external API calls. All model inference is local (HuggingFace cache only).
- LOCAL_DEV=true: use cpu/float32. Production: cuda/float16.
- Fail-closed: any translation failure returns (original_text, 0.0).
- Non-discrimination: language codes are technical identifiers only.
"""
from __future__ import annotations

import math
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.60"))
_BUNDLE_HIGH_CONFIDENCE_THRESHOLD = 0.80

_LOCAL_DEV: bool = os.getenv("LOCAL_DEV", "true").lower() in ("true", "1", "yes")

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# ---------------------------------------------------------------------------
# Lazy model loading — loaded once on first translation call
# ---------------------------------------------------------------------------

_nllb_tokenizer = None
_nllb_model = None


def _load_nllb() -> None:
    """Load NLLB tokenizer and model from local HuggingFace cache (no re-download)."""
    global _nllb_tokenizer, _nllb_model  # noqa: PLW0603

    if _nllb_tokenizer is not None and _nllb_model is not None:
        return

    from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

    _nllb_tokenizer = NllbTokenizer.from_pretrained(NLLB_MODEL_ID)

    if _LOCAL_DEV:
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL_ID,
            torch_dtype=torch.float32,
        )
        _nllb_model = _nllb_model.to("cpu")
    else:
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_to_english(text: str, source_lang: str) -> tuple[str, float]:
    """
    Translate text to English using NLLB-200-distilled-600M.

    Returns (translated_text, confidence_score).
    - If source_lang == "eng_Latn": returns (text, 1.0) immediately.
    - Confidence = mean of token-level scores from generation output;
      if not available, returns 0.75 as default.
    - On any failure: returns (text, 0.0).
    """
    if source_lang == "eng_Latn":
        return text, 1.0

    try:
        _load_nllb()

        inputs = _nllb_tokenizer(
            text,
            return_tensors="pt",
            src_lang=source_lang,
            truncation=True,
            max_length=512,
        ).to(_nllb_model.device)

        with torch.no_grad():
            output = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=_nllb_tokenizer.lang_code_to_id["eng_Latn"],
                max_length=512,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Compute confidence from per-token log-probabilities
        token_log_probs: list[float] = []
        if output.scores:
            for step_idx, step_scores in enumerate(output.scores):
                chosen_id = output.sequences[0][step_idx + 1]
                log_prob = torch.log_softmax(step_scores[0], dim=-1)[chosen_id].item()
                token_log_probs.append(log_prob)

        if token_log_probs:
            mean_log_prob = sum(token_log_probs) / len(token_log_probs)
            confidence = round(min(max(math.exp(mean_log_prob), 0.0), 1.0), 4)
        else:
            confidence = 0.75

        translated = _nllb_tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )[0]
        return translated, confidence

    except Exception:  # noqa: BLE001
        return text, 0.0


def compute_uncertainty_flag(confidence: float, source_lang: str) -> bool:
    """
    Return True if confidence < UNCERTAINTY_THRESHOLD or source_lang == "unknown".
    Uses the UNCERTAINTY_THRESHOLD constant (never hardcoded 0.60 here).
    """
    return confidence < UNCERTAINTY_THRESHOLD or source_lang == "unknown"


def build_multilingual_bundle(texts: dict[str, str]) -> list[dict]:
    """
    Build a filtered multilingual evaluation bundle from a dict of
    NLLB BCP-47 language code → original text in that language.

    Per-item filtering rules:
      confidence >= 0.80          → include, warning=False
      UNCERTAINTY_THRESHOLD ≤ conf < 0.80 → include, warning=True
      confidence < UNCERTAINTY_THRESHOLD  → exclude (log to stderr)
      source_lang == "unknown"    → exclude (log to stderr)

    Returns list of included items, each with fields:
      source_language, raw_text, translated_text,
      translation_confidence, warning
    """
    bundle: list[dict] = []

    for source_lang, raw_text in texts.items():
        if source_lang == "unknown":
            print(
                f"[bundle_filter] '{source_lang}' excluded: unknown language. "
                "multilingual coverage incomplete.",
                file=sys.stderr,
            )
            continue

        translated_text, confidence = translate_to_english(raw_text, source_lang)

        if confidence < UNCERTAINTY_THRESHOLD:
            print(
                f"[bundle_filter] '{source_lang}' excluded: "
                f"translation_confidence={confidence:.4f} < {UNCERTAINTY_THRESHOLD}. "
                "multilingual coverage incomplete.",
                file=sys.stderr,
            )
            continue

        warning = confidence < _BUNDLE_HIGH_CONFIDENCE_THRESHOLD

        bundle.append(
            {
                "source_language": source_lang,
                "raw_text": raw_text,
                "translated_text": translated_text,
                "translation_confidence": confidence,
                "warning": warning,
            }
        )

    return bundle


def assess_bundle_status(bundle: list[dict], english_excluded: bool) -> dict:
    """
    Determine pipeline action from the filtered bundle.

    Returns:
      {
        "status": "normal" | "single_language" | "hold_no_english",
        "all_non_english_low_confidence": bool,
        "multilingual_jailbreak_forced_low": bool,
      }

    Logic:
    - If english_excluded → "hold_no_english"
    - Elif bundle is empty or only eng_Latn items remain → "single_language",
        multilingual_jailbreak_forced_low=True
    - Else → "normal"
    - all_non_english_low_confidence=True when ALL non-English items have warning=True
    """
    non_english_items = [
        item for item in bundle if item["source_language"] != "eng_Latn"
    ]

    if english_excluded:
        return {
            "status": "hold_no_english",
            "all_non_english_low_confidence": False,
            "multilingual_jailbreak_forced_low": False,
        }

    if not non_english_items:
        # Bundle is empty or contains only English
        all_non_english_low = False  # no non-English items to assess
        return {
            "status": "single_language",
            "all_non_english_low_confidence": all_non_english_low,
            "multilingual_jailbreak_forced_low": True,
        }

    all_non_english_low = all(item["warning"] for item in non_english_items)

    return {
        "status": "normal",
        "all_non_english_low_confidence": all_non_english_low,
        "multilingual_jailbreak_forced_low": False,
    }
