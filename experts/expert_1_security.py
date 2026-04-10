"""
experts/expert_1_security.py
Expert 1: Security & Adversarial Robustness
"""
from __future__ import annotations

from experts.expert_base import (
    _ANCHORS,
    _EXPERT_ANCHOR_KEY,
    apply_multilingual_escalation,
    recompute_expert_risk_level,
    run_expert,
    validate_anchors,
    validate_high_has_evidence,
    validate_output_neutrality,
)


def run_expert_1(input_data: dict) -> dict:
    """
    Run Expert 1 (Security & Adversarial Robustness) and apply all five
    post-processing steps in mandatory order.
    """
    result = run_expert("expert_1", input_data)

    source_language = input_data.get("detected_language", "eng_Latn")
    translated_text = input_data.get("translated_text", "")

    # Step 1 — must run first: escalates severities that downstream steps read
    apply_multilingual_escalation(result, source_language, translated_text)
    # Step 2 — anchor integrity (auto-corrects wrong anchors in-place)
    validate_anchors(result, _ANCHORS, _EXPERT_ANCHOR_KEY["expert_1"])
    # Step 3 — hallucination guard (downgrades unsupported HIGH → MEDIUM)
    validate_high_has_evidence(result)
    # Step 4 — re-aggregate after all severity changes are final
    recompute_expert_risk_level(result)
    # Step 5 — non-discrimination check (flags only, no auto-correct)
    validate_output_neutrality(result)

    return result
