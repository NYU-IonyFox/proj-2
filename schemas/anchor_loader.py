from __future__ import annotations

import json
import os


def load_anchors(path: str) -> dict:
    """Load framework_anchors.json and return the raw parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_anchor_table(anchors: dict, expert_id: str) -> list[dict]:
    """
    Return a list of {dimension, criticality, primary_anchor} dicts
    for every dimension in expert_id's dimensions array.
    expert_id is one of: expert_1_security, expert_2_content, expert_3_governance.
    Only primary_anchor is included; supplementary_anchors are excluded.
    """
    expert_data = anchors[expert_id]
    result = []
    for dim in expert_data["dimensions"]:
        result.append({
            "dimension": dim["dimension"],
            "criticality": dim["criticality"],
            "primary_anchor": dim["primary_anchor"],
        })
    return result


def validate_anchors(expert_output: dict, anchors: dict, expert_id: str) -> list[str]:
    """
    For each dimension_score in expert_output, verify evidence_anchor matches
    the expected primary_anchor (framework, section, provision all exact).
    Auto-corrects mismatches in-place and returns a list of violation strings.
    Returns empty list if all anchors are correct.
    """
    anchor_table = {
        entry["dimension"]: entry["primary_anchor"]
        for entry in build_anchor_table(anchors, expert_id)
    }

    violations: list[str] = []
    for score in expert_output.get("dimension_scores", []):
        dimension = score.get("dimension")
        expected = anchor_table.get(dimension)
        if expected is None:
            continue

        actual = score.get("evidence_anchor", {})
        if (
            actual.get("framework") != expected["framework"]
            or actual.get("section") != expected["section"]
            or actual.get("provision") != expected["provision"]
        ):
            violations.append(
                f"evidence_anchor mismatch for dimension '{dimension}': "
                f"got {actual!r}, expected {expected!r}. Auto-corrected."
            )
            score["evidence_anchor"] = dict(expected)

    return violations
