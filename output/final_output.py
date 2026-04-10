"""
output/final_output.py
L6 Final Output Layer — council_output assembly, audit log writing,
and run_council() orchestrator for the full L1–L6 pipeline.

GLOBAL CONSTRAINTS:
- No external API calls. All inference is local.
- Fail-closed: any unhandled exception returns HOLD(uncertainty) / pipeline_error.
- Non-discrimination: output fields describe risk via framework violations only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (match pipeline_technical_reference_v3_1.md)
# ---------------------------------------------------------------------------

PROMPT_VERSION = "v3.0"
RUBRICS_VERSION = "v4.3"
ANCHORS_VERSION = "1.2"

AUDIT_LOG_DIR = Path(__file__).parent.parent / "audit_logs"
AUDIT_LOG_DIR.mkdir(exist_ok=True)

# Simple in-process counter for submission IDs within one run.
# Resets per process — sufficient for dev/test; Phase 6 can persist to disk.
_submission_counter = 0


# ---------------------------------------------------------------------------
# Task 1 — assemble_council_output()
# ---------------------------------------------------------------------------

def assemble_council_output(
    input_data: dict,
    expert_outputs: list[dict],
    resolution_result: dict,
) -> dict:
    """
    Assemble the final council_output dict (JSON Interface Spec §5.2).
    """
    submission_id = input_data.get("submission_id", "unknown")
    agent_name = input_data.get("agent_name", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_ref = f"audit-{submission_id}-{timestamp}"

    # Map expert list → {"expert_1": ..., "expert_2": ..., "expert_3": ...}
    expert_outputs_map: dict = {}
    for exp in expert_outputs:
        eid = exp.get("expert_id", "")
        if eid:
            expert_outputs_map[eid] = exp

    all_non_english_low = input_data.get("all_non_english_low_confidence", False)

    council_output = {
        "submission_id": submission_id,
        "agent_name": agent_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "final_decision": resolution_result.get("final_decision"),
        "decision_tier": resolution_result.get("decision_tier"),
        "decision_rule_triggered": resolution_result.get("decision_rule_triggered"),
        "expert_summary": resolution_result.get("expert_summary", {}),
        "expert_outputs": expert_outputs_map,
        "multilingual_metadata": {
            "source_language": input_data.get("detected_language", "unknown"),
            "translation_confidence": input_data.get("translation_confidence", 0.0),
            "uncertainty_flag": input_data.get("uncertainty_flag", False),
            "all_non_english_low_confidence": all_non_english_low,
        },
        "council_reasoning": resolution_result.get("council_reasoning", ""),
        "governance_action": resolution_result.get("governance_action", {}),
        "audit_log_reference": audit_ref,
    }

    # Propagate hold_reason and convergent_risk_note if present
    if resolution_result.get("hold_reason"):
        council_output["hold_reason"] = resolution_result["hold_reason"]
    if resolution_result.get("convergent_risk_note"):
        council_output["convergent_risk_note"] = resolution_result["convergent_risk_note"]

    return council_output


# ---------------------------------------------------------------------------
# Task 2 — write_audit_log()
# ---------------------------------------------------------------------------

def write_audit_log(council_output: dict, input_data: dict) -> str:
    """
    Write a JSON audit log file to audit_logs/.
    Returns the filename (string).
    """
    submission_id = council_output.get("submission_id", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"audit-{submission_id}-{timestamp}.json"
    log_path = AUDIT_LOG_DIR / filename

    # Collect post-processing violations from all expert outputs
    anchor_violations: list[str] = []
    neutrality_warnings: list[str] = []
    evidence_warnings: list[str] = []

    expert_outputs_map = council_output.get("expert_outputs", {})
    for _eid, exp in expert_outputs_map.items():
        anchor_violations.extend(exp.get("anchor_violations", []))
        neutrality_warnings.extend(exp.get("neutrality_warnings", []))
        evidence_warnings.extend(exp.get("evidence_warnings", []))

    notes: list[str] = []
    all_non_english_low = (
        council_output.get("multilingual_metadata", {})
        .get("all_non_english_low_confidence", False)
    )
    if all_non_english_low:
        notes.append(
            "All non-English bundle items excluded — multilingual coverage incomplete. "
            "Human review of original non-English outputs recommended."
        )

    audit_entry = {
        "audit_log_reference": council_output.get("audit_log_reference", filename),
        "submission_id": submission_id,
        "logged_at": datetime.now(timezone.utc).isoformat(),

        "input": {
            "agent_name": input_data.get("agent_name", "unknown"),
            "submission_id": submission_id,
            "raw_text": input_data.get("raw_text", ""),
            "detected_language": input_data.get("detected_language", "unknown"),
            "translation_confidence": input_data.get("translation_confidence", 0.0),
            "uncertainty_flag": input_data.get("uncertainty_flag", False),
        },

        "model_config": {
            "expert_model_id": os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
            "prompt_version": {
                "expert_1": PROMPT_VERSION,
                "expert_2": PROMPT_VERSION,
                "expert_3": PROMPT_VERSION,
            },
            "rubrics_version": RUBRICS_VERSION,
            "anchors_schema_version": ANCHORS_VERSION,
        },

        "council_output": council_output,

        "integrity_checks": {
            "anchor_violations": anchor_violations,
            "evidence_warnings": evidence_warnings,
            "neutrality_warnings": neutrality_warnings,
            "notes": notes,
        },
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(audit_entry, f, indent=2, ensure_ascii=False)

    return filename


# ---------------------------------------------------------------------------
# Task 3 — run_council()
# ---------------------------------------------------------------------------

def run_council(agent_name: str, text: str) -> dict:
    """
    Full L1–L6 pipeline orchestrator.

    Order: L1 → L2 → L3 (sequential) → L4 → L5 → L6

    Fail-closed: any unhandled exception returns
      {"final_decision": "HOLD", "hold_reason": "uncertainty",
       "decision_rule_triggered": "pipeline_error"}
    """
    global _submission_counter  # noqa: PLW0603
    try:
        # ------------------------------------------------------------------ L1
        from input_processor.screening import validate_input, detect_language

        try:
            validate_input(text)
        except ValueError:
            return {
                "final_decision": "HOLD",
                "hold_reason": "uncertainty",
                "decision_rule_triggered": "pipeline_error",
            }

        _submission_counter += 1
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        submission_id = f"eval-{date_str}-{_submission_counter:03d}"

        detected_language = detect_language(text)

        input_data: dict = {
            "submission_id": submission_id,
            "agent_name": agent_name,
            "raw_text": text,
        }

        # ------------------------------------------------------------------ L2
        from input_processor.multilingual import (
            translate_to_english,
            compute_uncertainty_flag,
        )

        translated_text, confidence = translate_to_english(text, detected_language)
        uncertainty_flag = compute_uncertainty_flag(confidence, detected_language)

        # Single-language submission path (no multilingual_bundle provided)
        multilingual_jailbreak_forced_low = False
        if detected_language == "unknown":
            uncertainty_flag = True

        input_data.update({
            "detected_language": detected_language,
            "translation_confidence": confidence,
            "uncertainty_flag": uncertainty_flag,
            "translated_text": translated_text,
            "multilingual_bundle": None,
            "all_non_english_low_confidence": False,
            "multilingual_jailbreak_forced_low": multilingual_jailbreak_forced_low,
        })

        # ------------------------------------------------------------------ L3
        from experts.expert_1_security import run_expert_1
        from experts.expert_2_content import run_expert_2
        from experts.expert_3_governance import run_expert_3

        expert_outputs: list[dict] = [
            run_expert_1(input_data),
            run_expert_2(input_data),
            run_expert_3(input_data),
        ]

        # L4 -----------------------------------------------------------------
        from council.arbitration import run_arbitration
        arbitration_result = run_arbitration(
            expert_outputs,
            input_data["uncertainty_flag"],
        )

        # L5 -----------------------------------------------------------------
        from council.resolution import run_resolution
        resolution_result = run_resolution(
            arbitration_result,
            expert_outputs,
            input_data,
        )

        # L6 -----------------------------------------------------------------
        council_output = assemble_council_output(
            input_data,
            expert_outputs,
            resolution_result,
        )
        write_audit_log(council_output, input_data)

        return council_output

    except Exception:  # noqa: BLE001
        return {
            "final_decision": "HOLD",
            "hold_reason": "uncertainty",
            "decision_rule_triggered": "pipeline_error",
        }
