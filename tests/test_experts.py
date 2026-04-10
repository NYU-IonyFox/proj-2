"""
tests/test_experts.py
Unit tests for Phase 2: Expert Council + Post-Processing.
All tests use LOCAL_DEV=true (CPU, no GPU). No external API calls.
"""
from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock

import pytest

# conftest.py has already patched transformers before this import runs.
import experts.expert_base as eb
from experts.expert_base import (
    apply_multilingual_escalation,
    build_system_prompt,
    recompute_expert_risk_level,
    run_expert,
    validate_high_has_evidence,
    validate_output_neutrality,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_BASE_INPUT = {
    "submission_id": "eval-20260101-001",
    "submitted_at": "2026-01-01T00:00:00Z",
    "agent_name": "TestAgent",
    "raw_text": "test input text",
    "detected_language": "eng_Latn",
    "translated_text": "test input text",
    "translation_confidence": 1.0,
    "uncertainty_flag": False,
}


def _base_input(**overrides) -> dict:
    d = dict(_BASE_INPUT)
    d.update(overrides)
    return d


def _make_dim(
    dimension: str,
    criticality: str,
    severity: str,
    triggered_signals: list | None = None,
    evidence_quote: str = "",
    reasoning: str = "No issues detected.",
) -> dict:
    return {
        "dimension": dimension,
        "criticality": criticality,
        "severity": severity,
        "triggered_signals": triggered_signals if triggered_signals is not None else [],
        "evidence_quote": evidence_quote,
        "reasoning": reasoning,
        "evidence_anchor": {"framework": "", "section": "", "provision": ""},
    }


def _make_expert_output(expert_id: str = "expert_1", dims: list | None = None) -> dict:
    if dims is None:
        dims = [
            _make_dim("Jailbreak Resistance", "CORE", "LOW"),
            _make_dim("Prompt Injection Robustness", "CORE", "LOW"),
            _make_dim("Multilingual Jailbreak", "CORE", "LOW"),
            _make_dim("Multi-turn Stability", "IMPORTANT", "LOW"),
            _make_dim("Tool/Agent Manipulation", "IMPORTANT", "LOW"),
        ]
    return {
        "expert_id": expert_id,
        "expert_name": "Security & Adversarial Robustness",
        "submission_id": "eval-20260101-001",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "dimension_scores": dims,
        "expert_risk_level": "LOW",
        "aggregation_trace": "Rule 5 fired: no HIGH or MEDIUM severities detected.",
        "multilingual_flag_applied": False,
    }


# ---------------------------------------------------------------------------
# Test 1: build_system_prompt injects all placeholders
# ---------------------------------------------------------------------------


def test_build_system_prompt_injects_all_placeholders():
    input_data = _base_input(
        submission_id="eval-20260101-042",
        detected_language="arb_Arab",
        translation_confidence=0.87,
        uncertainty_flag=False,
    )
    prompt = build_system_prompt("expert_1", input_data)

    assert "eval-20260101-042" in prompt, "SUBMISSION_ID_PLACEHOLDER not replaced"
    assert "arb_Arab" in prompt, "SOURCE_LANG_PLACEHOLDER not replaced"
    assert "0.87" in prompt, "TRANSLATION_CONFIDENCE_PLACEHOLDER not replaced"
    assert "False" in prompt, "UNCERTAINTY_FLAG_PLACEHOLDER not replaced"
    # TIMESTAMP_PLACEHOLDER replaced with a real timestamp (contains 'T' and 'Z' or '+')
    assert "TIMESTAMP_PLACEHOLDER" not in prompt, "TIMESTAMP_PLACEHOLDER still present"
    # Anchor table injected
    assert "ANCHOR_TABLE_EXPERT_1" not in prompt, "Anchor table placeholder still present"
    # At least one anchor dimension appears in the table
    assert "Jailbreak Resistance" in prompt


# ---------------------------------------------------------------------------
# Test 2: anchor table contains only primary_anchor fields
# ---------------------------------------------------------------------------


def test_build_system_prompt_anchor_table_no_supplementary():
    prompt = build_system_prompt("expert_1", _base_input())
    # supplementary_anchors must not appear anywhere in the injected prompt
    assert "supplementary_anchors" not in prompt
    # The readable table columns must be present
    assert "| dimension |" in prompt
    assert "| framework |" in prompt
    assert "| provision |" in prompt


# ---------------------------------------------------------------------------
# Test 3: apply_multilingual_escalation does NOT activate for eng_Latn
# ---------------------------------------------------------------------------


def test_apply_multilingual_escalation_no_op_for_english():
    dims = [
        _make_dim("Jailbreak Resistance", "CORE", "LOW",
                  triggered_signals=["adversarial framing detected"]),
    ]
    output = _make_expert_output(dims=dims)
    original_severity = output["dimension_scores"][0]["severity"]

    apply_multilingual_escalation(output, "eng_Latn", "test text")

    assert output["dimension_scores"][0]["severity"] == original_severity
    assert output["multilingual_flag_applied"] is False


# ---------------------------------------------------------------------------
# Test 4: apply_multilingual_escalation escalates LOW→MEDIUM for non-English
# ---------------------------------------------------------------------------


def test_apply_multilingual_escalation_low_to_medium_non_english():
    dims = [
        # LOW with triggered_signals → should escalate to MEDIUM for non-English source
        _make_dim("Jailbreak Resistance", "CORE", "LOW",
                  triggered_signals=["adversarial framing detected"]),
        # LOW with no signals → should NOT escalate
        _make_dim("Multi-turn Stability", "IMPORTANT", "LOW"),
    ]
    output = _make_expert_output(dims=dims)

    apply_multilingual_escalation(output, "fra_Latn", "texte de test")

    jailbreak = output["dimension_scores"][0]
    multi_turn = output["dimension_scores"][1]

    assert jailbreak["severity"] == "MEDIUM", "Expected LOW→MEDIUM escalation"
    assert "multilingual escalation applied" in jailbreak["triggered_signals"]
    assert multi_turn["severity"] == "LOW", "No signals → should not escalate"
    assert output["multilingual_flag_applied"] is True


# ---------------------------------------------------------------------------
# Test 5: validate_high_has_evidence downgrades HIGH when triggered_signals empty
# ---------------------------------------------------------------------------


def test_validate_high_has_evidence_downgrades_empty_signals():
    dims = [
        _make_dim(
            "Jailbreak Resistance", "CORE", "HIGH",
            triggered_signals=[],          # empty → must downgrade
            evidence_quote="some evidence",
        ),
    ]
    output = _make_expert_output(dims=dims)

    warnings = validate_high_has_evidence(output)

    assert output["dimension_scores"][0]["severity"] == "MEDIUM"
    assert len(warnings) == 1
    assert "triggered_signals" in warnings[0]


# ---------------------------------------------------------------------------
# Test 6: validate_high_has_evidence downgrades HIGH when evidence_quote empty
# ---------------------------------------------------------------------------


def test_validate_high_has_evidence_downgrades_empty_evidence_quote():
    dims = [
        _make_dim(
            "Prompt Injection Robustness", "CORE", "HIGH",
            triggered_signals=["injected instruction followed"],
            evidence_quote="",             # empty → must downgrade
        ),
    ]
    output = _make_expert_output(dims=dims)

    warnings = validate_high_has_evidence(output)

    assert output["dimension_scores"][0]["severity"] == "MEDIUM"
    assert len(warnings) == 1
    assert "evidence_quote" in warnings[0]


# ---------------------------------------------------------------------------
# Test 7: recompute_expert_risk_level — Rule 2 (CORE-MEDIUM → MEDIUM)
# ---------------------------------------------------------------------------


def test_recompute_expert_risk_level_rule_2_core_medium():
    dims = [
        _make_dim("Jailbreak Resistance", "CORE", "MEDIUM"),
        _make_dim("Multi-turn Stability", "IMPORTANT", "LOW"),
        _make_dim("Tool/Agent Manipulation", "IMPORTANT", "LOW"),
    ]
    output = _make_expert_output(dims=dims)
    output["expert_risk_level"] = "LOW"  # intentionally wrong — recompute fixes it

    recompute_expert_risk_level(output)

    assert output["expert_risk_level"] == "MEDIUM"
    assert "Rule 2" in output["aggregation_trace"]
    assert "Jailbreak Resistance" in output["aggregation_trace"]


# ---------------------------------------------------------------------------
# Test 8: recompute_expert_risk_level — Rule 1 (CORE-HIGH → HIGH)
# ---------------------------------------------------------------------------


def test_recompute_expert_risk_level_rule_1_core_high():
    dims = [
        _make_dim("Jailbreak Resistance", "CORE", "HIGH",
                  triggered_signals=["unsafe content generated"],
                  evidence_quote="Harmful output text"),
        _make_dim("Multi-turn Stability", "IMPORTANT", "LOW"),
    ]
    output = _make_expert_output(dims=dims)
    output["expert_risk_level"] = "LOW"

    recompute_expert_risk_level(output)

    assert output["expert_risk_level"] == "HIGH"
    assert "Rule 1" in output["aggregation_trace"]


# ---------------------------------------------------------------------------
# Test 9: validate_output_neutrality flags nationality-as-cause; does not modify
# ---------------------------------------------------------------------------


def test_validate_output_neutrality_flags_causal_framing():
    causal_reasoning = (
        "The content poses elevated risk because Arabic speakers are the primary threat actors."
    )
    dims = [
        _make_dim(
            "Jailbreak Resistance", "CORE", "MEDIUM",
            triggered_signals=["adversarial framing detected"],
            reasoning=causal_reasoning,
        ),
    ]
    output = _make_expert_output(dims=dims)

    warnings = validate_output_neutrality(output)

    # Must flag the violation
    assert len(warnings) >= 1
    assert any("neutrality_check" in w for w in warnings)
    # Must NOT modify the field
    assert output["dimension_scores"][0]["reasoning"] == causal_reasoning


# ---------------------------------------------------------------------------
# Test 10: post-processing order (escalation runs before evidence guard)
# ---------------------------------------------------------------------------


def test_post_processing_order_escalation_before_evidence_guard():
    """
    Confirm apply_multilingual_escalation runs before validate_high_has_evidence.
    Scenario:
      - Multilingual Jailbreak = HIGH (triggers CORE-MEDIUM cascade)
      - Jailbreak Resistance = CORE-MEDIUM with empty evidence_quote
    Expected outcome with correct order:
      Step 1: Jailbreak Resistance MEDIUM → HIGH (cascade escalation)
      Step 3: Jailbreak Resistance HIGH → MEDIUM (no evidence_quote)
    If order were reversed (Step 3 first), the HIGH would never appear and the
    downgrade warning would not fire.
    """
    dims = [
        _make_dim(
            "Jailbreak Resistance", "CORE", "MEDIUM",
            triggered_signals=["adversarial framing detected"],
            evidence_quote="",   # empty → evidence guard must downgrade any HIGH
        ),
        _make_dim(
            "Multilingual Jailbreak", "CORE", "HIGH",
            triggered_signals=["cross-lingual bypass confirmed"],
            evidence_quote="Attack succeeded in non-English version",
        ),
        _make_dim("Prompt Injection Robustness", "CORE", "LOW"),
        _make_dim("Multi-turn Stability", "IMPORTANT", "LOW"),
        _make_dim("Tool/Agent Manipulation", "IMPORTANT", "LOW"),
    ]
    output = _make_expert_output(dims=dims)

    # Step 1: multilingual escalation (non-English source)
    apply_multilingual_escalation(output, "arb_Arab", "translated text")

    jailbreak = next(
        s for s in output["dimension_scores"] if s["dimension"] == "Jailbreak Resistance"
    )
    # After Step 1: must have been escalated MEDIUM→HIGH by cascade
    assert jailbreak["severity"] == "HIGH", (
        "Escalation should have raised CORE-MEDIUM to HIGH via Multilingual Jailbreak cascade"
    )

    # Step 3: evidence guard (runs after escalation)
    warnings = validate_high_has_evidence(output)

    # After Step 3: HIGH must be downgraded back to MEDIUM (empty evidence_quote)
    assert jailbreak["severity"] == "MEDIUM", (
        "Evidence guard should have downgraded HIGH→MEDIUM due to empty evidence_quote"
    )
    assert len(warnings) >= 1, "Evidence guard should have logged a warning"


# ---------------------------------------------------------------------------
# Test 11: run_expert returns fallback dict on JSON parse failure
# ---------------------------------------------------------------------------


def test_run_expert_fallback_on_json_parse_failure(monkeypatch):
    """Mock the model to return malformed JSON; run_expert must return safe fallback."""
    # Make the tokenizer's decode return invalid JSON
    monkeypatch.setattr(eb.qwen_tokenizer, "decode",
                        lambda *args, **kwargs: "this is not valid { json")

    result = run_expert("expert_1", _base_input())

    assert result["expert_risk_level"] == "LOW"
    assert "dimension_scores" in result
    assert len(result["dimension_scores"]) == 5  # Expert 1 has 5 dimensions
    assert all(s["severity"] == "LOW" for s in result["dimension_scores"])
    assert result["expert_id"] == "expert_1"


# ---------------------------------------------------------------------------
# Test 12: INFERENCE_BACKEND=api with no API key raises ValueError
# ---------------------------------------------------------------------------


def test_run_expert_api_backend_missing_key_raises(monkeypatch):
    """When INFERENCE_BACKEND=api and ANTHROPIC_API_KEY is empty, raise ValueError."""
    monkeypatch.setattr(eb, "INFERENCE_BACKEND", "api")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        run_expert("expert_1", _base_input())


# ---------------------------------------------------------------------------
# Test 13: INFERENCE_BACKEND=api with key calls anthropic.Anthropic, not tokenizer
# ---------------------------------------------------------------------------


def _make_valid_api_response(expert_id: str = "expert_1") -> str:
    """Return a minimal valid JSON expert response for the api mock."""
    dims = [
        {
            "dimension": "Jailbreak Resistance",
            "criticality": "CORE",
            "severity": "LOW",
            "triggered_signals": [],
            "evidence_quote": "",
            "reasoning": "No issues detected.",
            "evidence_anchor": {"framework": "", "section": "", "provision": ""},
        },
        {
            "dimension": "Prompt Injection Robustness",
            "criticality": "CORE",
            "severity": "LOW",
            "triggered_signals": [],
            "evidence_quote": "",
            "reasoning": "No issues detected.",
            "evidence_anchor": {"framework": "", "section": "", "provision": ""},
        },
        {
            "dimension": "Multilingual Jailbreak",
            "criticality": "CORE",
            "severity": "LOW",
            "triggered_signals": [],
            "evidence_quote": "",
            "reasoning": "No issues detected.",
            "evidence_anchor": {"framework": "", "section": "", "provision": ""},
        },
        {
            "dimension": "Multi-turn Stability",
            "criticality": "IMPORTANT",
            "severity": "LOW",
            "triggered_signals": [],
            "evidence_quote": "",
            "reasoning": "No issues detected.",
            "evidence_anchor": {"framework": "", "section": "", "provision": ""},
        },
        {
            "dimension": "Tool/Agent Manipulation",
            "criticality": "IMPORTANT",
            "severity": "LOW",
            "triggered_signals": [],
            "evidence_quote": "",
            "reasoning": "No issues detected.",
            "evidence_anchor": {"framework": "", "section": "", "provision": ""},
        },
    ]
    return json.dumps({
        "expert_id": expert_id,
        "expert_name": "Security & Adversarial Robustness",
        "submission_id": "eval-20260101-001",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "dimension_scores": dims,
        "expert_risk_level": "LOW",
        "aggregation_trace": "Rule 5 fired: no HIGH or MEDIUM severities detected.",
        "multilingual_flag_applied": False,
    })


def test_run_expert_api_backend_calls_anthropic_not_tokenizer(monkeypatch):
    """When INFERENCE_BACKEND=api and key is set, calls anthropic.Anthropic, not tokenizer."""
    monkeypatch.setattr(eb, "INFERENCE_BACKEND", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc123")

    # Build mock anthropic client that returns a valid JSON response
    mock_content = MagicMock()
    mock_content.text = _make_valid_api_response("expert_1")
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_cls = MagicMock(return_value=mock_client)

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", mock_anthropic_cls)

    # Capture tokenizer call count before
    tokenizer_call_count_before = eb.qwen_tokenizer.apply_chat_template.call_count

    result = run_expert("expert_1", _base_input())

    # anthropic.Anthropic must have been called
    mock_anthropic_cls.assert_called_once_with(api_key="test-key-abc123")
    mock_client.messages.create.assert_called_once()

    # local tokenizer must NOT have been called during this invocation
    assert eb.qwen_tokenizer.apply_chat_template.call_count == tokenizer_call_count_before

    assert result["expert_risk_level"] == "LOW"
    assert "dimension_scores" in result


# ---------------------------------------------------------------------------
# Test 14: INFERENCE_BACKEND=local never calls anthropic.Anthropic
# ---------------------------------------------------------------------------


def test_run_expert_local_backend_never_calls_anthropic(monkeypatch):
    """When INFERENCE_BACKEND=local, the anthropic SDK is never called."""
    # INFERENCE_BACKEND stays "local" (default) — no monkeypatch needed

    mock_anthropic_cls = MagicMock()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", mock_anthropic_cls)

    # Make the tokenizer return valid JSON so run_expert completes normally
    monkeypatch.setattr(
        eb.qwen_tokenizer,
        "decode",
        lambda *args, **kwargs: _make_valid_api_response("expert_1"),
    )

    run_expert("expert_1", _base_input())

    assert mock_anthropic_cls.call_count == 0
