"""
input_processor/repo_analyzer.py
Fetches files from a GitHub repository and produces a structured AI system
description via the Anthropic API.

GLOBAL CONSTRAINTS:
- Only this file may make external API calls in the input_processor layer.
- Fail-closed: unhandled exceptions re-raise as-is.
- Non-discrimination: output describes framework violations only.
"""
from __future__ import annotations

import os
import re

import anthropic
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_RE = re.compile(
    r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/[^/]+)?/?$"
)

_RAW_BASE = "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"

_BRANCHES = ["main", "master"]

_OPTIONAL_FILES = ["requirements.txt", "pyproject.toml", ".env.example"]
_ENTRY_FILES = ["app.py", "main.py", "server.py", "index.js"]

_SYSTEM_PROMPT = """
You are a technical analyst preparing an AI system description for
safety evaluation. You will receive the contents of files from a
GitHub repository. Produce a structured description of this AI system
covering: purpose and use case, technical architecture and frameworks,
external API dependencies and data flows, input and output interfaces
(file uploads, web endpoints, data formats), authentication and access
control mechanisms, data handling and retention practices, and any
stated safety or moderation measures.

STRICT RULES:
- Base your description ONLY on the provided file contents.
- Never infer, assume, or add information not present in the files.
- If information is absent, write exactly: not specified in provided files
- Do NOT make safety judgments — describe facts only.
- Do NOT search the internet or use external knowledge.
- Write in clear prose, 300-500 words.
- Do NOT use bullet points — write in paragraphs.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_github_url(repo_url: str) -> tuple[str, str]:
    """Return (owner, repo) from a GitHub URL or raise ValueError."""
    m = _GITHUB_RE.match(repo_url.strip())
    if not m:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")
    return m.group(1), m.group(2)


def _fetch_raw(owner: str, repo: str, filename: str, timeout: int = 10) -> str | None:
    """
    Try main then master branch. Return file content or None if not found.
    Raises requests.exceptions.Timeout on timeout.
    """
    for branch in _BRANCHES:
        url = _RAW_BASE.format(owner=owner, repo=repo, branch=branch, filename=filename)
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.exceptions.Timeout:
            raise
        if resp.status_code == 200:
            return resp.text
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_repo_description(repo_url: str) -> dict:
    """
    Fetch key files from a GitHub repository and return a structured
    AI system description produced by the Anthropic API.

    Parameters
    ----------
    repo_url:
        A GitHub repository URL in one of the supported formats.

    Returns
    -------
    dict with keys:
        structured_description, source_repo_url, agent_name,
        files_fetched, files_not_found
    """
    # --- Step 1: Parse URL ---------------------------------------------------
    owner, repo_name = _parse_github_url(repo_url)

    # --- Step 2: Fetch files -------------------------------------------------
    files_fetched: list[str] = []
    files_not_found: list[str] = []
    file_contents: dict[str, str] = {}

    try:
        # README.md is REQUIRED
        readme = _fetch_raw(owner, repo_name, "README.md")
        if readme is None:
            raise ValueError(f"Could not fetch README.md from {repo_url}")
        files_fetched.append("README.md")
        file_contents["README.md"] = readme

        # requirements.txt — optional
        req = _fetch_raw(owner, repo_name, "requirements.txt")
        if req is not None:
            files_fetched.append("requirements.txt")
            file_contents["requirements.txt"] = req
        else:
            files_not_found.append("requirements.txt")

            # pyproject.toml — only if requirements.txt not found
            pyproject = _fetch_raw(owner, repo_name, "pyproject.toml")
            if pyproject is not None:
                files_fetched.append("pyproject.toml")
                file_contents["pyproject.toml"] = pyproject
            else:
                files_not_found.append("pyproject.toml")

        # .env.example — optional
        env_example = _fetch_raw(owner, repo_name, ".env.example")
        if env_example is not None:
            files_fetched.append(".env.example")
            file_contents[".env.example"] = env_example
        else:
            files_not_found.append(".env.example")

        # Main entry file — first match wins, truncated to 300 lines
        entry_found = False
        for entry_name in _ENTRY_FILES:
            entry = _fetch_raw(owner, repo_name, entry_name)
            if entry is not None:
                lines = entry.splitlines()[:300]
                file_contents[entry_name] = "\n".join(lines)
                files_fetched.append(entry_name)
                entry_found = True
                break
        if not entry_found:
            for entry_name in _ENTRY_FILES:
                files_not_found.append(entry_name)

    except requests.exceptions.Timeout:
        raise ValueError(f"Timeout fetching files from {repo_url}")

    # --- Step 3: Call Anthropic API ------------------------------------------
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Set it in your .env file.")

    # Build user message
    user_message = f"Repository: {repo_url}\n\n"
    for filename, content in file_contents.items():
        user_message += f"=== {filename} ===\n{content}\n\n"
    for filename in files_not_found:
        user_message += f"=== {filename} ===\nnot found\n\n"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        structured_description = message.content[0].text
    except anthropic.APIError as e:
        raise ValueError(f"repo_analyzer API call failed: {str(e)}")

    # --- Step 4: Return dict -------------------------------------------------
    return {
        "structured_description": structured_description,
        "source_repo_url": repo_url,
        "agent_name": repo_name,
        "files_fetched": files_fetched,
        "files_not_found": files_not_found,
    }
