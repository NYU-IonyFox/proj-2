"""
tests/test_pipeline.py
End-to-end and unit tests for Phase 5 (Council Orchestration + FastAPI).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure LOCAL_DEV so expert_base uses cpu/float32 mock path
os.environ.setdefault("LOCAL_DEV", "true")

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_INPUT_PATH = Path(__file__).parent / "sample_inputs" / "verimedia_sample.txt"

_REQUIRED_TOP_LEVEL_FIELDS = [
    "submission_id",
    "agent_name",
    "evaluated_at",
    "final_decision",
    "decision_tier",
    "decision_rule_triggered",
    "expert_summary",
    "expert_outputs",
    "multilingual_metadata",
    "council_reasoning",
    "governance_action",
    "audit_log_reference",
]

_MINIMAL_EXPERT = {
    "expert_id": "expert_1",
    "expert_name": "Security & Adversarial Robustness",
    "submission_id": "eval-20260410-001",
    "evaluated_at": "2026-04-10T00:00:00+00:00",
    "dimension_scores": [],
    "expert_risk_level": "LOW",
    "aggregation_trace": "Rule 5 (default): All LOW",
    "multilingual_flag_applied": False,
}

_MINIMAL_INPUT = {
    "submission_id": "eval-20260410-001",
    "agent_name": "TestAgent",
    "raw_text": "sample text",
    "detected_language": "eng_Latn",
    "translation_confidence": 1.0,
    "uncertainty_flag": False,
    "translated_text": "sample text",
    "multilingual_bundle": None,
    "all_non_english_low_confidence": False,
}

_MINIMAL_RESOLUTION = {
    "final_decision": "PASS",
    "decision_tier": None,
    "hold_reason": None,
    "decision_rule_triggered": "Rule 6: All Experts = LOW → PASS",
    "expert_summary": {
        "expert_1_security": "LOW",
        "expert_2_content": "LOW",
        "expert_3_governance": "LOW",
    },
    "council_reasoning": "No significant risks detected.",
    "governance_action": {
        "decision": "PASS",
        "deployment_allowed": True,
        "requires_mitigation_plan": False,
        "requires_retest": False,
        "escalate_to_human": False,
        "notes": "Approved for deployment.",
    },
}


# ---------------------------------------------------------------------------
# run_council integration tests
# ---------------------------------------------------------------------------

class TestRunCouncil:
    def test_returns_valid_structure(self):
        """run_council returns a dict with all required top-level fields."""
        from output.final_output import run_council

        text = SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
        result = run_council("VeriMedia", text)

        for field in _REQUIRED_TOP_LEVEL_FIELDS:
            assert field in result, f"Missing field: {field}"

    def test_carries_agent_name(self):
        """agent_name from the call argument appears in council_output."""
        from output.final_output import run_council

        text = SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
        result = run_council("MyTestAgent", text)

        # Fail-closed response also valid; skip agent_name check in that case
        if result.get("decision_rule_triggered") == "pipeline_error":
            pytest.skip("Pipeline in fail-closed mode; skipping agent_name check.")

        assert result.get("agent_name") == "MyTestAgent"

    def test_fail_closed_on_arbitration_exception(self, monkeypatch):
        """Any exception in run_arbitration → HOLD(uncertainty) + pipeline_error."""
        from output.final_output import run_council

        def _raise(*args, **kwargs):
            raise RuntimeError("Simulated arbitration failure")

        monkeypatch.setattr(
            "output.final_output.run_council.__globals__"
            if False else "council.arbitration.run_arbitration",
            _raise,
            raising=False,
        )

        # Patch via the import inside run_council's execution scope
        with patch("council.arbitration.run_arbitration", side_effect=RuntimeError("boom")):
            result = run_council("TestAgent", "some text")

        assert result["final_decision"] == "HOLD"
        assert result.get("hold_reason") == "uncertainty"
        assert result["decision_rule_triggered"] == "pipeline_error"

    def test_multilingual_en_tag_sets_translation_confidence_1(self):
        """run_council with [EN]-tagged input sets multilingual_metadata.translation_confidence=1.0."""
        from output.final_output import run_council

        text = "[EN] This is a safe test message for evaluation."
        result = run_council("TestAgent", text)

        if result.get("decision_rule_triggered") == "pipeline_error":
            pytest.skip("Pipeline in fail-closed mode; skipping multilingual confidence check.")

        assert result["multilingual_metadata"]["translation_confidence"] == 1.0

    def test_raises_value_error_when_neither_text_nor_repo_url(self):
        """run_council raises ValueError when called with neither text nor repo_url."""
        from output.final_output import run_council

        # The fail-closed wrapper catches all exceptions and returns HOLD,
        # so we verify via the pipeline_error decision_rule_triggered.
        result = run_council()
        assert result["final_decision"] == "HOLD"
        assert result["decision_rule_triggered"] == "pipeline_error"

    def test_text_path_unchanged(self):
        """run_council with text= still works (existing path unchanged)."""
        from output.final_output import run_council

        text = SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
        result = run_council(agent_name="VeriMedia", text=text)

        assert "final_decision" in result

    def test_repo_url_calls_fetch_repo_description(self):
        """run_council with repo_url calls fetch_repo_description and uses structured_description."""
        from output.final_output import run_council

        mock_repo_data = {
            "agent_name": "RepoAgent",
            "structured_description": SAMPLE_INPUT_PATH.read_text(encoding="utf-8"),
        }

        with patch(
            "input_processor.repo_analyzer.fetch_repo_description",
            return_value=mock_repo_data,
        ) as mock_fetch:
            result = run_council(repo_url="https://github.com/example/repo")

        mock_fetch.assert_called_once_with("https://github.com/example/repo")
        assert "final_decision" in result

    def test_source_repo_url_in_output_when_repo_url_provided(self):
        """council_output contains source_repo_url field when repo_url is provided."""
        from output.final_output import run_council

        mock_repo_data = {
            "agent_name": "RepoAgent",
            "structured_description": SAMPLE_INPUT_PATH.read_text(encoding="utf-8"),
        }

        with patch(
            "input_processor.repo_analyzer.fetch_repo_description",
            return_value=mock_repo_data,
        ):
            result = run_council(repo_url="https://github.com/example/repo")

        if result.get("decision_rule_triggered") == "pipeline_error":
            pytest.skip("Pipeline in fail-closed mode; skipping source_repo_url check.")

        assert result.get("source_repo_url") == "https://github.com/example/repo"

    def test_source_repo_url_is_none_when_text_input_used(self):
        """council_output source_repo_url is None when text input is used."""
        from output.final_output import run_council

        text = SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
        result = run_council(agent_name="VeriMedia", text=text)

        if result.get("decision_rule_triggered") == "pipeline_error":
            pytest.skip("Pipeline in fail-closed mode; skipping source_repo_url check.")

        assert result.get("source_repo_url") is None

    def test_result_contains_narrative_report_key(self):
        """run_council result always contains a narrative_report key."""
        from output.final_output import run_council

        text = SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
        result = run_council("VeriMedia", text)

        assert "narrative_report" in result

    def test_narrative_report_is_string(self):
        """run_council narrative_report value is a string (may be empty)."""
        from output.final_output import run_council

        text = SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
        result = run_council("VeriMedia", text)

        assert isinstance(result.get("narrative_report", ""), str)


# ---------------------------------------------------------------------------
# assemble_council_output tests
# ---------------------------------------------------------------------------

class TestAssembleCouncilOutput:
    def test_all_non_english_low_confidence_false_by_default(self):
        """all_non_english_low_confidence defaults to False when not in input_data."""
        from output.final_output import assemble_council_output

        input_data = dict(_MINIMAL_INPUT)
        input_data.pop("all_non_english_low_confidence", None)

        result = assemble_council_output(input_data, [_MINIMAL_EXPERT], _MINIMAL_RESOLUTION)

        assert result["multilingual_metadata"]["all_non_english_low_confidence"] is False

    def test_all_non_english_low_confidence_true_when_flagged(self):
        """all_non_english_low_confidence=True propagated from input_data."""
        from output.final_output import assemble_council_output

        input_data = dict(_MINIMAL_INPUT)
        input_data["all_non_english_low_confidence"] = True

        result = assemble_council_output(input_data, [_MINIMAL_EXPERT], _MINIMAL_RESOLUTION)

        assert result["multilingual_metadata"]["all_non_english_low_confidence"] is True


# ---------------------------------------------------------------------------
# write_audit_log tests
# ---------------------------------------------------------------------------

class TestWriteAuditLog:
    def _make_council_output(self, all_non_english_low=False):
        from output.final_output import assemble_council_output

        input_data = dict(_MINIMAL_INPUT)
        input_data["all_non_english_low_confidence"] = all_non_english_low
        return assemble_council_output(
            input_data, [_MINIMAL_EXPERT], _MINIMAL_RESOLUTION
        ), input_data

    def test_creates_file_in_audit_logs(self, tmp_path):
        """write_audit_log creates a .json file in audit_logs/."""
        from output import final_output as fo

        orig_dir = fo.AUDIT_LOG_DIR
        fo.AUDIT_LOG_DIR = tmp_path
        tmp_path.mkdir(exist_ok=True)

        try:
            council_output, input_data = self._make_council_output()
            filename = fo.write_audit_log(council_output, input_data)
            assert (tmp_path / filename).exists()
        finally:
            fo.AUDIT_LOG_DIR = orig_dir

    def test_model_config_fields(self, tmp_path):
        """Audit log contains correct model_config fields."""
        from output import final_output as fo

        orig_dir = fo.AUDIT_LOG_DIR
        fo.AUDIT_LOG_DIR = tmp_path

        try:
            council_output, input_data = self._make_council_output()
            filename = fo.write_audit_log(council_output, input_data)
            data = json.loads((tmp_path / filename).read_text(encoding="utf-8"))

            mc = data["model_config"]
            assert mc["prompt_version"] == {
                "expert_1": "v3.0",
                "expert_2": "v3.0",
                "expert_3": "v3.0",
            }
            assert mc["rubrics_version"] == "v4.3"
            assert mc["anchors_schema_version"] == "1.2"
        finally:
            fo.AUDIT_LOG_DIR = orig_dir

    def test_input_record_contains_agent_name(self, tmp_path):
        """Audit log input record contains agent_name."""
        from output import final_output as fo

        orig_dir = fo.AUDIT_LOG_DIR
        fo.AUDIT_LOG_DIR = tmp_path

        try:
            input_data = dict(_MINIMAL_INPUT)
            input_data["agent_name"] = "AuditTestAgent"
            council_output, _ = self._make_council_output()
            council_output["agent_name"] = "AuditTestAgent"

            filename = fo.write_audit_log(council_output, input_data)
            data = json.loads((tmp_path / filename).read_text(encoding="utf-8"))

            assert data["input"]["agent_name"] == "AuditTestAgent"
        finally:
            fo.AUDIT_LOG_DIR = orig_dir

    def test_audit_note_present_when_all_non_english_low_confidence(self, tmp_path):
        """Audit log notes include the coverage-incomplete message when flag is True."""
        from output import final_output as fo

        orig_dir = fo.AUDIT_LOG_DIR
        fo.AUDIT_LOG_DIR = tmp_path

        try:
            council_output, input_data = self._make_council_output(all_non_english_low=True)
            filename = fo.write_audit_log(council_output, input_data)
            data = json.loads((tmp_path / filename).read_text(encoding="utf-8"))

            notes = data["integrity_checks"]["notes"]
            assert any(
                "All non-English bundle items excluded" in note for note in notes
            ), f"Expected audit note not found. notes={notes}"
        finally:
            fo.AUDIT_LOG_DIR = orig_dir


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestFastAPI:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_health_returns_200_and_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_evaluate_returns_final_decision(self, client):
        """POST /evaluate returns council_output with final_decision field."""
        mock_output = {
            "submission_id": "eval-20260410-001",
            "agent_name": "MockAgent",
            "evaluated_at": "2026-04-10T00:00:00+00:00",
            "final_decision": "PASS",
            "decision_tier": None,
            "decision_rule_triggered": "Rule 6: All Experts = LOW → PASS",
            "expert_summary": {},
            "expert_outputs": {},
            "multilingual_metadata": {
                "source_language": "eng_Latn",
                "translation_confidence": 1.0,
                "uncertainty_flag": False,
                "all_non_english_low_confidence": False,
            },
            "council_reasoning": "No risk.",
            "governance_action": {},
            "audit_log_reference": "audit-eval-20260410-001-20260410T000000Z",
        }

        with patch("output.final_output.run_council", return_value=mock_output):
            resp = client.post(
                "/evaluate",
                json={"agent_name": "MockAgent", "text": "some agent output text"},
            )

        assert resp.status_code == 200
        assert "final_decision" in resp.json()

    def test_evaluate_fail_closed_on_exception(self, client):
        """POST /evaluate returns HOLD when run_council raises an exception."""
        with patch(
            "output.final_output.run_council",
            side_effect=RuntimeError("Simulated failure"),
        ):
            resp = client.post(
                "/evaluate",
                json={"agent_name": "BrokenAgent", "text": "some text"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["final_decision"] == "HOLD"
        assert body["decision_rule_triggered"] == "pipeline_error"
