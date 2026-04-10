from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, field_validator


# --- Input Schema ---

class MultilangBundleItem(BaseModel):
    source_language: str
    raw_text: str
    translated_text: str
    translation_confidence: float
    warning: bool

    @field_validator("translation_confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("translation_confidence must be in [0.0, 1.0]")
        return v


class InputSchema(BaseModel):
    submission_id: str
    submitted_at: str
    agent_name: str
    raw_text: str
    detected_language: str
    translated_text: str
    translation_confidence: float
    uncertainty_flag: bool
    multilingual_bundle: Optional[list[MultilangBundleItem]] = None

    @field_validator("translation_confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("translation_confidence must be in [0.0, 1.0]")
        return v


# --- Expert Output Schema ---

class EvidenceAnchor(BaseModel):
    framework: str
    section: str
    provision: str


class DimensionScore(BaseModel):
    dimension: str
    criticality: str
    severity: str
    triggered_signals: list[str]
    evidence_quote: str
    reasoning: str
    evidence_anchor: EvidenceAnchor

    @field_validator("severity")
    @classmethod
    def _severity_values(cls, v: str) -> str:
        if v not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("severity must be one of: LOW, MEDIUM, HIGH")
        return v

    @field_validator("criticality")
    @classmethod
    def _criticality_values(cls, v: str) -> str:
        if v not in {"CORE", "IMPORTANT"}:
            raise ValueError("criticality must be one of: CORE, IMPORTANT")
        return v


class ExpertOutput(BaseModel):
    expert_id: str
    expert_name: str
    submission_id: str
    evaluated_at: str
    dimension_scores: list[DimensionScore]
    expert_risk_level: str
    aggregation_trace: str
    multilingual_flag_applied: bool


# --- Council Output Schema ---

class MultilingualMetadata(BaseModel):
    source_language: str
    translation_confidence: float
    uncertainty_flag: bool
    all_non_english_low_confidence: bool


class GovernanceAction(BaseModel):
    decision: str
    deployment_allowed: bool
    requires_mitigation_plan: bool
    requires_retest: bool
    escalate_to_human: bool
    notes: str


class CouncilOutput(BaseModel):
    submission_id: str
    agent_name: str
    evaluated_at: str
    final_decision: str
    decision_tier: Optional[str] = None
    decision_rule_triggered: str
    expert_summary: dict[str, str]
    expert_outputs: dict[str, ExpertOutput]
    multilingual_metadata: MultilingualMetadata
    council_reasoning: str
    governance_action: GovernanceAction
    audit_log_reference: str

    @field_validator("final_decision")
    @classmethod
    def _decision_values(cls, v: str) -> str:
        if v not in {"PASS", "CONDITIONAL", "HOLD", "REJECT"}:
            raise ValueError("final_decision must be one of: PASS, CONDITIONAL, HOLD, REJECT")
        return v
