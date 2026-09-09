"""Data models and schemas for Multi-Agent PDF Extraction and Consensus Evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import re
from typing import Any, Dict, List, Optional, Union


class CheckResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class FailureMode(str, Enum):
    NONE = "NONE"
    FORMAT_MISMATCH = "FORMAT_MISMATCH"
    MISSING_VALUE = "MISSING_VALUE"
    HALLUCINATION = "HALLUCINATION"
    WRONG_ENTITY = "WRONG_ENTITY"
    UNREADABLE_SOURCE = "UNREADABLE_SOURCE"
    PREFERRED_NAME_CONFUSED_WITH_LEGAL = "PREFERRED_NAME_CONFUSED_WITH_LEGAL"
    NAME_TRUNCATED = "NAME_TRUNCATED"
    CONFUSED_WITH_ENCOUNTER_DATE = "CONFUSED_WITH_ENCOUNTER_DATE"
    DIGIT_TRANSPOSITION = "DIGIT_TRANSPOSITION"
    FACILITY_ADDRESS_CONFUSED_AS_PATIENT = "FACILITY_ADDRESS_CONFUSED_AS_PATIENT"
    MISSING_SUITE_OR_APT = "MISSING_SUITE_OR_APT"
    WRONG_ZIP = "WRONG_ZIP"
    CONFUSED_WITH_GROUP_NUMBER = "CONFUSED_WITH_GROUP_NUMBER"
    CONFUSED_WITH_PAYER_ID = "CONFUSED_WITH_PAYER_ID"
    CHARACTER_ERROR = "CHARACTER_ERROR"
    SECONDARY_CODE_EXTRACTED_AS_PRIMARY = "SECONDARY_CODE_EXTRACTED_AS_PRIMARY"
    INVALID_ICD_FORMAT = "INVALID_ICD_FORMAT"
    OTHER = "OTHER"


class StatusClassification(str, Enum):
    UNANIMOUS_PASS = "UNANIMOUS_PASS"
    MAJORITY_PASS = "MAJORITY_PASS"
    CONTESTED = "CONTESTED"
    REJECTED = "REJECTED"


class RoutingAction(str, Enum):
    AUTO_ACCEPT = "Auto-Accept"
    AUTO_ACCEPT_WITH_WARNING = "Auto-Accept with Warning"
    ROUTE_TO_HITL = "Route to Human-in-the-Loop (HITL)"
    FLAG_FOR_RESUBMISSION = "Flag for Extraction Failure / Resubmission"


@dataclass
class SyntacticRule:
    required: bool = True
    type: str = "string"
    min_tokens: Optional[int] = None
    format_regex: Optional[str] = None
    masking_allowed: Optional[bool] = None
    expected_subfields: Optional[List[str]] = None


@dataclass
class SemanticGroundingRule:
    source_section: str = ""
    verification_criteria: str = ""
    failure_modes: List[str] = field(default_factory=list)


@dataclass
class FieldExtractionCriteria:
    field_name: str
    syntactic_rules: SyntacticRule
    semantic_grounding_rules: SemanticGroundingRule

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FieldExtractionCriteria:
        syn_raw = data.get("syntactic_rules", {})
        syn = SyntacticRule(
            required=syn_raw.get("required", True),
            type=syn_raw.get("type", "string"),
            min_tokens=syn_raw.get("min_tokens"),
            format_regex=syn_raw.get("format_regex"),
            masking_allowed=syn_raw.get("masking_allowed"),
            expected_subfields=syn_raw.get("expected_subfields"),
        )
        sem_raw = data.get("semantic_grounding_rules", {})
        sem = SemanticGroundingRule(
            source_section=sem_raw.get("source_section", ""),
            verification_criteria=sem_raw.get("verification_criteria", ""),
            failure_modes=sem_raw.get("failure_modes", []),
        )
        return cls(
            field_name=data["field_name"],
            syntactic_rules=syn,
            semantic_grounding_rules=sem,
        )


@dataclass
class RubricSpec:
    rubric_version: str
    document_type: str
    extraction_criteria: List[FieldExtractionCriteria]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricSpec:
        criteria = [
            FieldExtractionCriteria.from_dict(item)
            for item in data.get("extraction_criteria", [])
        ]
        return cls(
            rubric_version=data.get("rubric_version", "1.0"),
            document_type=data.get("document_type", "Document"),
            extraction_criteria=criteria,
        )

    @classmethod
    def from_json_file(cls, filepath: str) -> RubricSpec:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_field_criteria(self, field_name: str) -> Optional[FieldExtractionCriteria]:
        for c in self.extraction_criteria:
            if c.field_name == field_name:
                return c
        return None


@dataclass
class JudgeEvaluation:
    field_name: str
    syntactic_check: CheckResult
    grounding_check: CheckResult
    verdict: CheckResult
    failure_mode: str
    justification: str
    proposed_correction: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JudgeEvaluation:
        syntactic_str = str(data.get("syntactic_check", "FAIL")).upper()
        grounding_str = str(data.get("grounding_check", "FAIL")).upper()
        verdict_str = str(data.get("verdict", "FAIL")).upper()

        return cls(
            field_name=data.get("field_name", ""),
            syntactic_check=CheckResult.PASS if syntactic_str == "PASS" else CheckResult.FAIL,
            grounding_check=CheckResult.PASS if grounding_str == "PASS" else CheckResult.FAIL,
            verdict=CheckResult.PASS if verdict_str == "PASS" else CheckResult.FAIL,
            failure_mode=data.get("failure_mode", "NONE"),
            justification=data.get("justification", ""),
            proposed_correction=data.get("proposed_correction"),
        )


@dataclass
class JudgeReport:
    judge_id: str
    model_name: str
    thinking_budget: int
    evaluations: List[JudgeEvaluation]
    raw_response: Optional[str] = None
    execution_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class FieldConsensus:
    field_name: str
    candidate_value: Any
    agreement_count: int
    total_judges: int
    agreement_ratio: float
    status: StatusClassification
    confidence_band: str
    routing_action: RoutingAction
    accepted_value: Optional[Any]
    dissent_reasons: List[str]
    failure_modes_detected: List[str]
    proposed_corrections: List[Any]


@dataclass
class ExceptionDetail:
    field_name: str
    status: str
    candidate_value: Any
    agreement_ratio: str
    failure_modes: List[str]
    justifications: List[str]
    proposed_corrections: List[Any]
    recommended_routing: str


@dataclass
class PipelineReport:
    document_type: str
    rubric_version: str
    timestamp: str
    total_fields: int
    accepted_field_count: int
    contested_field_count: int
    rejected_field_count: int
    quorum_metadata: Dict[str, Any]
    candidate_extraction: Dict[str, Any]
    accepted_payload: Dict[str, Any]
    exceptions: List[Dict[str, Any]]
    consensus_breakdown: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    extractor_model: str = "gemini-3.8-flash"
    judge_model: str = "gemini-3.6-flash"
    thinking_budget: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
