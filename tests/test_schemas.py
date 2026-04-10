"""Unit tests for Phase 1: Pydantic schemas and AnchorLoader."""
from __future__ import annotations

import os
import pytest
from pydantic import ValidationError

from schemas.models import (
    CouncilOutput,
    DimensionScore,
    EvidenceAnchor,
    ExpertOutput,
    GovernanceAction,
    InputSchema,
    MultilangBundleItem,
    MultilingualMetadata,
)
from schemas.anchor_loader import build_anchor_table, load_anchors, validate_anchors

# Path to the shared anchor file (copied into schemas/ as required)
ANCHORS_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas", "framework_anchors.json")


@pytest.fixture(scope="module")
def anchors():
    return load_anchors(ANCHORS_PATH)


# ---------------------------------------------------------------------------
# InputSchema tests
# ---------------------------------------------------------------------------

def _base_input(**overrides) -> dict:
    data = {
        "submission_id": "eval-20260101-001",
        "submitted_at": "2026-01-01T00:00:00Z",
        "agent_name": "TestAgent",
        "raw_text": "hello",
        "detected_language": "eng_Latn",
        "translated_text": "hello",
        "translation_confidence": 1.0,
        "uncertainty_flag": False,
    }
    data.update(overrides)
    return data


def test_input_rejects_confidence_above_1():
    with pytest.raises(ValidationError):
        InputSchema(**_base_input(translation_confidence=1.1))


def test_input_rejects_confidence_below_0():
    with pytest.raises(ValidationError):
        InputSchema(**_base_input(translation_confidence=-0.1))


def test_input_accepts_confidence_boundaries():
    InputSchema(**_base_input(translation_confidence=0.0))
    InputSchema(**_base_input(translation_confidence=1.0))


def test_input_multilingual_bundle_optional_none_by_default():
    obj = InputSchema(**_base_input())
    assert obj.multilingual_bundle is None


def test_input_accepts_multilingual_bundle():
    bundle = [
        {
            "source_language": "fra_Latn",
            "raw_text": "bonjour",
            "translated_text": "hello",
            "translation_confidence": 0.95,
            "warning": False,
        }
    ]
    obj = InputSchema(**_base_input(multilingual_bundle=bundle))
    assert len(obj.multilingual_bundle) == 1


# ---------------------------------------------------------------------------
# MultilangBundleItem tests
# ---------------------------------------------------------------------------

def test_multilang_bundle_item_requires_raw_text():
    with pytest.raises(ValidationError):
        MultilangBundleItem(
            source_language="fra_Latn",
            translated_text="hello",
            translation_confidence=0.9,
            warning=False,
            # raw_text missing
        )


def test_multilang_bundle_item_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        MultilangBundleItem(
            source_language="fra_Latn",
            raw_text="bonjour",
            translated_text="hello",
            translation_confidence=1.5,
            warning=False,
        )


# ---------------------------------------------------------------------------
# DimensionScore tests
# ---------------------------------------------------------------------------

def _base_anchor() -> dict:
    return {"framework": "NIST AI RMF 1.0", "section": "Map 1.5", "provision": "Some provision"}


def test_dimension_score_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension="Jailbreak Resistance",
            criticality="CORE",
            severity="CRITICAL",  # invalid
            triggered_signals=[],
            evidence_quote="",
            reasoning="test",
            evidence_anchor=_base_anchor(),
        )


def test_dimension_score_accepts_valid_severities():
    for sev in ("LOW", "MEDIUM", "HIGH"):
        DimensionScore(
            dimension="Jailbreak Resistance",
            criticality="CORE",
            severity=sev,
            triggered_signals=[],
            evidence_quote="",
            reasoning="ok",
            evidence_anchor=_base_anchor(),
        )


def test_dimension_score_rejects_invalid_criticality():
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension="Jailbreak Resistance",
            criticality="MANDATORY",  # invalid
            severity="LOW",
            triggered_signals=[],
            evidence_quote="",
            reasoning="test",
            evidence_anchor=_base_anchor(),
        )


# ---------------------------------------------------------------------------
# CouncilOutput tests
# ---------------------------------------------------------------------------

def _base_expert_output(expert_id: str = "expert_1") -> dict:
    return {
        "expert_id": expert_id,
        "expert_name": "Security & Adversarial Robustness",
        "submission_id": "eval-20260101-001",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "dimension_scores": [],
        "expert_risk_level": "LOW",
        "aggregation_trace": "Default LOW",
        "multilingual_flag_applied": False,
    }


def _base_council(**overrides) -> dict:
    data = {
        "submission_id": "eval-20260101-001",
        "agent_name": "TestAgent",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "final_decision": "PASS",
        "decision_tier": None,
        "decision_rule_triggered": "Rule 6: All LOW → PASS",
        "expert_summary": {
            "expert_1_security": "LOW",
            "expert_2_content": "LOW",
            "expert_3_governance": "LOW",
        },
        "expert_outputs": {
            "expert_1": _base_expert_output("expert_1"),
            "expert_2": _base_expert_output("expert_2"),
            "expert_3": _base_expert_output("expert_3"),
        },
        "multilingual_metadata": {
            "source_language": "eng_Latn",
            "translation_confidence": 1.0,
            "uncertainty_flag": False,
            "all_non_english_low_confidence": False,
        },
        "council_reasoning": "All LOW.",
        "governance_action": {
            "decision": "PASS",
            "deployment_allowed": True,
            "requires_mitigation_plan": False,
            "requires_retest": False,
            "escalate_to_human": False,
            "notes": "No issues.",
        },
        "audit_log_reference": "audit-eval-20260101-001-T00:00:00Z",
    }
    data.update(overrides)
    return data


def test_council_rejects_invalid_final_decision():
    with pytest.raises(ValidationError):
        CouncilOutput(**_base_council(final_decision="MAYBE"))


def test_council_accepts_valid_final_decisions():
    for decision in ("PASS", "CONDITIONAL", "HOLD", "REJECT"):
        CouncilOutput(**_base_council(final_decision=decision))


# ---------------------------------------------------------------------------
# build_anchor_table tests
# ---------------------------------------------------------------------------

EXPERT_IDS = ["expert_1_security", "expert_2_content", "expert_3_governance"]

EXPECTED_DIMENSIONS = {
    "expert_1_security": [
        "Jailbreak Resistance",
        "Prompt Injection Robustness",
        "Multilingual Jailbreak",
        "Multi-turn Stability",
        "Tool/Agent Manipulation",
    ],
    "expert_2_content": [
        "Harmful Content Generation",
        "Sensitive Data Leakage",
        "Bias & Fairness",
        "Hallucination",
        "Manipulation / Deception",
    ],
    "expert_3_governance": [
        "Regulatory Compliance",
        "Global Equity & Accessibility",
        "Political / Conflict Neutrality",
        "Transparency & Explainability",
        "Human Oversight Compatibility",
    ],
}


def test_build_anchor_table_returns_only_primary_anchor(anchors):
    for expert_id in EXPERT_IDS:
        table = build_anchor_table(anchors, expert_id)
        for entry in table:
            assert "primary_anchor" in entry
            assert "supplementary_anchors" not in entry
            assert set(entry.keys()) == {"dimension", "criticality", "primary_anchor"}


def test_build_anchor_table_correct_dimensions_expert_1(anchors):
    table = build_anchor_table(anchors, "expert_1_security")
    names = [e["dimension"] for e in table]
    assert names == EXPECTED_DIMENSIONS["expert_1_security"]


def test_build_anchor_table_correct_dimensions_expert_2(anchors):
    table = build_anchor_table(anchors, "expert_2_content")
    names = [e["dimension"] for e in table]
    assert names == EXPECTED_DIMENSIONS["expert_2_content"]


def test_build_anchor_table_correct_dimensions_expert_3(anchors):
    table = build_anchor_table(anchors, "expert_3_governance")
    names = [e["dimension"] for e in table]
    assert names == EXPECTED_DIMENSIONS["expert_3_governance"]


# ---------------------------------------------------------------------------
# validate_anchors tests
# ---------------------------------------------------------------------------

def _make_expert_output_with_anchor(dimension: str, anchor: dict) -> dict:
    return {
        "dimension_scores": [
            {
                "dimension": dimension,
                "criticality": "CORE",
                "severity": "LOW",
                "triggered_signals": [],
                "evidence_quote": "",
                "reasoning": "test",
                "evidence_anchor": anchor,
            }
        ]
    }


def test_validate_anchors_autocorrects_wrong_anchor_and_returns_violation(anchors):
    wrong_anchor = {"framework": "WRONG", "section": "X", "provision": "Y"}
    expert_output = _make_expert_output_with_anchor("Jailbreak Resistance", wrong_anchor)

    violations = validate_anchors(expert_output, anchors, "expert_1_security")

    assert len(violations) == 1
    assert "Jailbreak Resistance" in violations[0]

    # Auto-corrected in-place
    corrected = expert_output["dimension_scores"][0]["evidence_anchor"]
    expected = build_anchor_table(anchors, "expert_1_security")[0]["primary_anchor"]
    assert corrected == expected


def test_validate_anchors_returns_empty_when_all_correct(anchors):
    table = build_anchor_table(anchors, "expert_1_security")
    dim_entry = table[0]
    expert_output = _make_expert_output_with_anchor(
        dim_entry["dimension"], dict(dim_entry["primary_anchor"])
    )

    violations = validate_anchors(expert_output, anchors, "expert_1_security")
    assert violations == []
