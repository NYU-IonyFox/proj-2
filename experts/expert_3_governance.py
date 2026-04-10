"""
experts/expert_3_governance.py
Expert 3: Governance, Compliance & Societal Risk
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


def run_expert_3(input_data: dict) -> dict:
    """
    Run Expert 3 (Governance, Compliance & Societal Risk) and apply all five
    post-processing steps in mandatory order.
    """
    result = run_expert("expert_3", input_data)

    source_language = input_data.get("detected_language", "eng_Latn")
    translated_text = input_data.get("translated_text", "")

    # Step 1
    apply_multilingual_escalation(result, source_language, translated_text)
    # Step 2
    validate_anchors(result, _ANCHORS, _EXPERT_ANCHOR_KEY["expert_3"])
    # Step 3
    validate_high_has_evidence(result)
    # Step 4
    recompute_expert_risk_level(result)
    # Step 5
    validate_output_neutrality(result)

    return result
