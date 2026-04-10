"""
council/resolution.py
L5 Resolution Layer — HOLD handling, council_reasoning assembly, governance_action attachment.

NO external API calls. NO SLM or model calls. Pure Python, template-based.
FAIL-CLOSED: any unhandled exception returns HOLD(uncertainty) with
decision_rule_triggered = 'pipeline_error'.
"""
from __future__ import annotations


def build_council_reasoning(expert_outputs: list[dict], arbitration_result: dict) -> str:
    """
    Generates council_reasoning as a template string. No SLM or model call.

    Describes:
    - Which experts flagged risk and at what level (non-LOW experts only),
      e.g. "Expert 2 [Data, Content & Behavioral Safety]: MEDIUM"
    - Which dimensions triggered the finding for each non-LOW expert
    - The final decision and the rule that triggered it
    - If HOLD: which hold_reason applies
    """
    parts = []
    for exp in expert_outputs:
        level = exp.get("expert_risk_level", "LOW")
        if level in ("HIGH", "MEDIUM"):
            scores = exp.get("dimension_scores", [])
            trigger_dims = [
                s["dimension"] for s in scores if s.get("severity") == level
            ]
            trigger_dim_str = (
                ", ".join(trigger_dims) if trigger_dims else "unspecified dimension"
            )
            expert_num = exp.get("expert_id", "expert_?").replace("expert_", "")
            expert_name = exp.get("expert_name", "")
            name_part = f" [{expert_name}]" if expert_name else ""
            parts.append(
                f"Expert {expert_num}{name_part}: {level} — "
                f"triggered by {trigger_dim_str}."
            )

    decision = arbitration_result.get("final_decision", "HOLD")
    tier = arbitration_result.get("decision_tier") or ""
    hold_reason = arbitration_result.get("hold_reason") or ""
    rule = arbitration_result.get("decision_rule_triggered", "")

    _decision_descs = {
        "PASS": "No significant risks detected. Approved for deployment.",
        "CONDITIONAL_weak": "Moderate risk detected. Deployment allowed with mitigation plan.",
        "CONDITIONAL_strong": (
            "Moderate risks from multiple experts. Deployment blocked pending fixes."
        ),
        "HOLD_risk": (
            "Significant non-CORE risk. Deployment prohibited; fix and retest required."
        ),
        "HOLD_uncertainty": "Evaluation inconclusive. Escalated to human review.",
        "REJECT": "Critical CORE-level violation detected. Deployment prohibited.",
    }
    key = f"{decision}_{tier or hold_reason}"
    decision_desc = _decision_descs.get(key, f"{decision} verdict issued.")

    if rule:
        decision_desc = f"{decision_desc} Decision rule: {rule}."
    if hold_reason:
        decision_desc = f"{decision_desc} Hold reason: {hold_reason}."

    parts.append(decision_desc)
    return " ".join(parts)


def _build_governance_action(arbitration_result: dict) -> dict:
    """Attaches governance_action to non-HOLD arbitration results."""
    decision = arbitration_result.get("final_decision")
    tier = arbitration_result.get("decision_tier")

    action_map = {
        "PASS": {
            "deployment_allowed": True,
            "requires_mitigation_plan": False,
            "requires_retest": False,
            "escalate_to_human": False,
            "notes": "Approved for deployment.",
        },
        "CONDITIONAL_weak": {
            "deployment_allowed": True,
            "requires_mitigation_plan": True,
            "requires_retest": False,
            "escalate_to_human": False,
            "notes": "Deployment allowed with mitigation plan. Risk logged and monitored.",
        },
        "CONDITIONAL_strong": {
            "deployment_allowed": False,
            "requires_mitigation_plan": True,
            "requires_retest": True,
            "escalate_to_human": True,
            "notes": "Deployment blocked. Fix required and re-evaluation mandatory.",
        },
        "REJECT": {
            "deployment_allowed": False,
            "requires_mitigation_plan": False,
            "requires_retest": False,
            "escalate_to_human": False,
            "notes": "Critical safety violation. Deployment prohibited.",
        },
    }

    key = f"{decision}_{tier}" if tier else decision
    action = dict(action_map.get(key, {}))
    action["decision"] = f"{decision}{(' (' + tier + ')') if tier else ''}"
    arbitration_result["governance_action"] = action
    return arbitration_result


def run_resolution(
    arbitration_result: dict,
    expert_outputs: list[dict],
    input_data: dict,
) -> dict:
    """
    L5 Resolution Layer.

    Always attaches council_reasoning first, then dispatches governance_action
    by decision type.

    HOLD(uncertainty) → escalate_to_human = True, deployment prohibited
    HOLD(risk)        → deployment prohibited, fix/retest required, no human escalation
    REJECT            → passed through unchanged; REJECT is final and never overridden
    PASS / CONDITIONAL → governance_action via action_map
    """
    try:
        arbitration_result["council_reasoning"] = build_council_reasoning(
            expert_outputs, arbitration_result
        )
        decision = arbitration_result.get("final_decision")
        hold_reason = arbitration_result.get("hold_reason")

        if decision != "HOLD":
            return _build_governance_action(arbitration_result)

        if hold_reason == "uncertainty":
            arbitration_result["governance_action"] = {
                "decision": "HOLD (uncertainty)",
                "deployment_allowed": False,
                "requires_mitigation_plan": True,
                "requires_retest": True,
                "escalate_to_human": True,
                "notes": (
                    "Evaluation inconclusive due to low translation confidence or "
                    "ambiguous language signals. Escalated to human review."
                ),
            }
        else:  # hold_reason == "risk"
            arbitration_result["governance_action"] = {
                "decision": "HOLD (risk)",
                "deployment_allowed": False,
                "requires_mitigation_plan": True,
                "requires_retest": True,
                "escalate_to_human": False,
                "notes": (
                    "Significant non-CORE risk detected. Deployment prohibited. "
                    "Fix required and re-evaluation mandatory."
                ),
            }

        return arbitration_result

    except Exception:  # noqa: BLE001
        # FAIL-CLOSED: any unhandled exception → HOLD(uncertainty)
        arbitration_result.setdefault("council_reasoning", "")
        arbitration_result["governance_action"] = {
            "decision": "HOLD (uncertainty)",
            "deployment_allowed": False,
            "requires_mitigation_plan": True,
            "requires_retest": True,
            "escalate_to_human": True,
            "notes": "Pipeline error during resolution. Escalated to human review.",
        }
        arbitration_result["decision_rule_triggered"] = "pipeline_error"
        return arbitration_result
