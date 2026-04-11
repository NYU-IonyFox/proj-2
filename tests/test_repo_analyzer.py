"""
tests/test_repo_analyzer.py
Unit tests for input_processor.repo_analyzer.fetch_repo_description.
All external I/O (requests, anthropic) is mocked.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from input_processor.repo_analyzer import fetch_repo_description


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status: int, text: str = "") -> MagicMock:
    """Return a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


def _make_anthropic_client(description: str = "Test description") -> MagicMock:
    """Return a mock anthropic.Anthropic client whose messages.create returns description."""
    content_block = MagicMock()
    content_block.text = description

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = message
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepoAnalyzerURLParsing(unittest.TestCase):
    def test_raises_on_invalid_url(self):
        with self.assertRaises(ValueError) as ctx:
            fetch_repo_description("https://gitlab.com/owner/repo")
        self.assertIn("Invalid GitHub URL", str(ctx.exception))

    def test_raises_on_non_github_url(self):
        with self.assertRaises(ValueError):
            fetch_repo_description("https://bitbucket.org/owner/repo")

    def test_extracts_agent_name_plain(self):
        """agent_name == 'VeriMedia' from plain GitHub URL."""
        readme_resp = _make_response(200, "# VeriMedia\nA media verification tool.")
        not_found = _make_response(404)

        with patch("requests.get") as mock_get, \
             patch("anthropic.Anthropic", return_value=_make_anthropic_client()), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_get.side_effect = lambda url, timeout: (
                readme_resp if "README.md" in url else not_found
            )
            result = fetch_repo_description("https://github.com/FlashCarrot/VeriMedia")

        self.assertEqual(result["agent_name"], "VeriMedia")

    def test_extracts_agent_name_git_suffix(self):
        """agent_name stripped of .git suffix."""
        readme_resp = _make_response(200, "# MyRepo")
        not_found = _make_response(404)

        with patch("requests.get") as mock_get, \
             patch("anthropic.Anthropic", return_value=_make_anthropic_client()), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_get.side_effect = lambda url, timeout: (
                readme_resp if "README.md" in url else not_found
            )
            result = fetch_repo_description("https://github.com/owner/MyRepo.git")

        self.assertEqual(result["agent_name"], "MyRepo")

    def test_extracts_agent_name_tree_suffix(self):
        """agent_name extracted from URL with /tree/main."""
        readme_resp = _make_response(200, "# AnotherRepo")
        not_found = _make_response(404)

        with patch("requests.get") as mock_get, \
             patch("anthropic.Anthropic", return_value=_make_anthropic_client()), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_get.side_effect = lambda url, timeout: (
                readme_resp if "README.md" in url else not_found
            )
            result = fetch_repo_description(
                "https://github.com/owner/AnotherRepo/tree/main"
            )

        self.assertEqual(result["agent_name"], "AnotherRepo")


class TestRepoAnalyzerReadme(unittest.TestCase):
    def test_raises_when_readme_404_on_both_branches(self):
        not_found = _make_response(404)

        with patch("requests.get", return_value=not_found), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with self.assertRaises(ValueError) as ctx:
                fetch_repo_description("https://github.com/owner/repo")

        self.assertIn("Could not fetch README.md", str(ctx.exception))


class TestRepoAnalyzerFileLists(unittest.TestCase):
    def _run(self, get_side_effect):
        with patch("requests.get") as mock_get, \
             patch("anthropic.Anthropic", return_value=_make_anthropic_client()), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_get.side_effect = get_side_effect
            return fetch_repo_description("https://github.com/owner/repo")

    def test_files_fetched_contains_readme(self):
        readme_resp = _make_response(200, "# Repo")
        not_found = _make_response(404)

        result = self._run(
            lambda url, timeout: readme_resp if "README.md" in url else not_found
        )

        self.assertIn("README.md", result["files_fetched"])

    def test_files_not_found_contains_optional_when_404(self):
        readme_resp = _make_response(200, "# Repo")
        not_found = _make_response(404)

        result = self._run(
            lambda url, timeout: readme_resp if "README.md" in url else not_found
        )

        # All optional files should be in not_found
        for fname in ["requirements.txt", "pyproject.toml", ".env.example"]:
            self.assertIn(fname, result["files_not_found"])

    def test_requirements_fetched_suppresses_pyproject(self):
        """pyproject.toml is only checked if requirements.txt is absent."""
        readme_resp = _make_response(200, "# Repo")
        req_resp = _make_response(200, "requests==2.28.0")
        not_found = _make_response(404)

        def side_effect(url, timeout):
            if "README.md" in url:
                return readme_resp
            if "requirements.txt" in url:
                return req_resp
            return not_found

        result = self._run(side_effect)

        self.assertIn("requirements.txt", result["files_fetched"])
        self.assertNotIn("pyproject.toml", result["files_fetched"])
        self.assertNotIn("pyproject.toml", result["files_not_found"])


class TestRepoAnalyzerApiCall(unittest.TestCase):
    def test_returns_structured_description(self):
        readme_resp = _make_response(200, "# Repo")
        not_found = _make_response(404)
        client_mock = _make_anthropic_client("Test description")

        with patch("requests.get") as mock_get, \
             patch("anthropic.Anthropic", return_value=client_mock), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_get.side_effect = lambda url, timeout: (
                readme_resp if "README.md" in url else not_found
            )
            result = fetch_repo_description("https://github.com/owner/repo")

        self.assertEqual(result["structured_description"], "Test description")

    def test_raises_when_api_key_empty(self):
        readme_resp = _make_response(200, "# Repo")
        not_found = _make_response(404)

        with patch("requests.get") as mock_get, \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            mock_get.side_effect = lambda url, timeout: (
                readme_resp if "README.md" in url else not_found
            )
            with self.assertRaises(ValueError) as ctx:
                fetch_repo_description("https://github.com/owner/repo")

        self.assertIn("ANTHROPIC_API_KEY is not set", str(ctx.exception))

    def test_raises_on_anthropic_api_error(self):
        import anthropic as _anthropic

        readme_resp = _make_response(200, "# Repo")
        not_found = _make_response(404)

        client_mock = MagicMock()
        client_mock.messages.create.side_effect = _anthropic.APIStatusError(
            "bad request",
            response=MagicMock(status_code=400, headers={}),
            body={},
        )

        with patch("requests.get") as mock_get, \
             patch("anthropic.Anthropic", return_value=client_mock), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_get.side_effect = lambda url, timeout: (
                readme_resp if "README.md" in url else not_found
            )
            with self.assertRaises(ValueError) as ctx:
                fetch_repo_description("https://github.com/owner/repo")

        self.assertIn("repo_analyzer API call failed", str(ctx.exception))


class TestRepoAnalyzerTimeout(unittest.TestCase):
    def test_raises_value_error_on_timeout(self):
        import requests as _requests

        with patch("requests.get", side_effect=_requests.exceptions.Timeout), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with self.assertRaises(ValueError) as ctx:
                fetch_repo_description("https://github.com/owner/repo")

        self.assertIn("Timeout fetching files", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
