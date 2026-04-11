"""
tests/test_report_generator.py
Tests for output/report_generator.py (PART C).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_COUNCIL_OUTPUT = {
    "submission_id": "eval-20260410-001",
    "agent_name": "TestAgent",
    "final_decision": "PASS",
    "decision_tier": None,
    "decision_rule_triggered": "Rule 6: All Experts = LOW → PASS",
    "expert_outputs": {
        "expert_1": {"expert_id": "expert_1", "expert_risk_level": "LOW"},
        "expert_2": {"expert_id": "expert_2", "expert_risk_level": "LOW"},
        "expert_3": {"expert_id": "expert_3", "expert_risk_level": "LOW"},
    },
    "council_reasoning": "No significant risks detected.",
}


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _mock_anthropic_response(self, text: str):
        """Build a mock Anthropic client that returns the given text."""
        mock_content = MagicMock()
        mock_content.text = text

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        return mock_client

    def test_returns_non_empty_string_on_success(self, monkeypatch):
        """generate_report returns a non-empty string when the API call succeeds."""
        from output import report_generator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xxx")

        mock_client = self._mock_anthropic_response("# Report\nTest content")

        with patch("anthropic.Anthropic", return_value=mock_client):
            result = report_generator.generate_report(_MINIMAL_COUNCIL_OUTPUT)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "# Report" in result

    def test_returns_empty_string_when_api_key_missing(self, monkeypatch):
        """generate_report returns '' when ANTHROPIC_API_KEY is missing."""
        from output import report_generator

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = report_generator.generate_report(_MINIMAL_COUNCIL_OUTPUT)

        assert result == ""

    def test_returns_empty_string_when_api_key_empty(self, monkeypatch):
        """generate_report returns '' when ANTHROPIC_API_KEY is empty string."""
        from output import report_generator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        result = report_generator.generate_report(_MINIMAL_COUNCIL_OUTPUT)

        assert result == ""

    def test_returns_empty_string_on_api_exception(self, monkeypatch):
        """generate_report returns '' when the Anthropic API raises an exception."""
        import anthropic
        from output import report_generator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xxx")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIStatusError(
            message="Server error",
            response=MagicMock(status_code=500),
            body={},
        )

        with patch("anthropic.Anthropic", return_value=mock_client):
            result = report_generator.generate_report(_MINIMAL_COUNCIL_OUTPUT)

        assert result == ""

    def test_returns_empty_string_on_generic_exception(self, monkeypatch):
        """generate_report returns '' on any unexpected exception."""
        from output import report_generator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xxx")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("Unexpected failure")

        with patch("anthropic.Anthropic", return_value=mock_client):
            result = report_generator.generate_report(_MINIMAL_COUNCIL_OUTPUT)

        assert result == ""

    def test_passes_structured_description_in_user_message(self, monkeypatch):
        """generate_report includes the structured_description in the user message."""
        from output import report_generator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xxx")
        mock_client = self._mock_anthropic_response("# Report\nContent here")

        with patch("anthropic.Anthropic", return_value=mock_client):
            report_generator.generate_report(
                _MINIMAL_COUNCIL_OUTPUT,
                structured_description="This is a test system description.",
            )

        call_kwargs = mock_client.messages.create.call_args
        user_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "This is a test system description." in user_content

    def test_uses_haiku_model(self, monkeypatch):
        """generate_report calls the API with claude-haiku-4-5 model."""
        from output import report_generator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xxx")
        mock_client = self._mock_anthropic_response("# Report\nTest")

        with patch("anthropic.Anthropic", return_value=mock_client):
            report_generator.generate_report(_MINIMAL_COUNCIL_OUTPUT)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_max_tokens_is_2048(self, monkeypatch):
        """generate_report calls the API with max_tokens=2048."""
        from output import report_generator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xxx")
        mock_client = self._mock_anthropic_response("# Report\nTest")

        with patch("anthropic.Anthropic", return_value=mock_client):
            report_generator.generate_report(_MINIMAL_COUNCIL_OUTPUT)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# save_report tests
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_creates_file_with_correct_filename(self, tmp_path, monkeypatch):
        """save_report creates outputs/{submission_id}_report.md."""
        from output import report_generator

        monkeypatch.setattr(report_generator, "OUTPUTS_DIR", tmp_path)

        path = report_generator.save_report("# Report\nContent", "eval-20260410-001")

        assert path is not None
        assert path == tmp_path / "eval-20260410-001_report.md"
        assert path.exists()

    def test_file_content_matches_report_text(self, tmp_path, monkeypatch):
        """save_report writes the exact report_text to the file."""
        from output import report_generator

        monkeypatch.setattr(report_generator, "OUTPUTS_DIR", tmp_path)
        content = "# AI Safety Evaluation Report\n\nSome findings."

        path = report_generator.save_report(content, "eval-test-001")
        assert path.read_text(encoding="utf-8") == content

    def test_returns_none_when_report_text_empty(self, tmp_path, monkeypatch):
        """save_report returns None when report_text is empty string."""
        from output import report_generator

        monkeypatch.setattr(report_generator, "OUTPUTS_DIR", tmp_path)

        result = report_generator.save_report("", "eval-20260410-001")

        assert result is None

    def test_does_not_create_file_when_report_text_empty(self, tmp_path, monkeypatch):
        """save_report does not write a file when report_text is empty."""
        from output import report_generator

        monkeypatch.setattr(report_generator, "OUTPUTS_DIR", tmp_path)

        report_generator.save_report("", "eval-20260410-001")

        assert not (tmp_path / "eval-20260410-001_report.md").exists()

    def test_creates_outputs_directory_if_not_exists(self, tmp_path, monkeypatch):
        """save_report creates the outputs/ directory if it does not exist."""
        from output import report_generator

        new_dir = tmp_path / "new_outputs"
        assert not new_dir.exists()

        monkeypatch.setattr(report_generator, "OUTPUTS_DIR", new_dir)

        path = report_generator.save_report("# Report", "eval-dir-test")

        assert new_dir.exists()
        assert path is not None
        assert path.exists()

    def test_returns_path_object(self, tmp_path, monkeypatch):
        """save_report returns a Path object."""
        from output import report_generator

        monkeypatch.setattr(report_generator, "OUTPUTS_DIR", tmp_path)

        result = report_generator.save_report("# Report", "eval-path-test")

        assert isinstance(result, Path)
