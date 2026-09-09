"""Tests for models and RubricSpec parsing."""

import json
import os
import pytest
from pdf_consensus_evaluator.models import (
    CheckResult,
    FieldExtractionCriteria,
    JudgeEvaluation,
    RubricSpec,
    StatusClassification,
    SyntacticRule,
    SemanticGroundingRule,
)

SAMPLE_RUBRIC_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/rubric_spec_healthcare_patient_intake_form.json",
)


def test_rubric_spec_from_json():
    assert os.path.exists(SAMPLE_RUBRIC_PATH), f"Sample rubric missing at {SAMPLE_RUBRIC_PATH}"
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)

    assert rubric.document_type == "Healthcare_Patient_Intake_Form"
    assert rubric.rubric_version == "2.0"
    assert len(rubric.extraction_criteria) == 6

    # Test individual field criteria
    ssn_crit = rubric.get_field_criteria("social_security_number")
    assert ssn_crit is not None
    assert ssn_crit.syntactic_rules.required is True
    assert ssn_crit.syntactic_rules.format_regex == "^\\d{3}-\\d{2}-\\d{4}$"
    assert "DIGIT_TRANSPOSITION" in ssn_crit.semantic_grounding_rules.failure_modes

    addr_crit = rubric.get_field_criteria("residential_address")
    assert addr_crit is not None
    assert addr_crit.syntactic_rules.type == "object"
    assert "street" in addr_crit.syntactic_rules.expected_subfields


def test_judge_evaluation_deserialization():
    raw_eval = {
        "field_name": "patient_full_name",
        "syntactic_check": "PASS",
        "grounding_check": "PASS",
        "verdict": "PASS",
        "failure_mode": "NONE",
        "justification": "Verified on page 1.",
        "proposed_correction": None,
    }
    je = JudgeEvaluation.from_dict(raw_eval)
    assert je.field_name == "patient_full_name"
    assert je.syntactic_check == CheckResult.PASS
    assert je.verdict == CheckResult.PASS
    assert je.failure_mode == "NONE"


def test_judge_evaluation_failure_deserialization():
    raw_eval = {
        "field_name": "social_security_number",
        "syntactic_check": "FAIL",
        "grounding_check": "FAIL",
        "verdict": "FAIL",
        "failure_mode": "DIGIT_TRANSPOSITION",
        "justification": "Page 1 shows 987-65-4320 but candidate had 987-65-4321.",
        "proposed_correction": "987-65-4320",
    }
    je = JudgeEvaluation.from_dict(raw_eval)
    assert je.verdict == CheckResult.FAIL
    assert je.failure_mode == "DIGIT_TRANSPOSITION"
    assert je.proposed_correction == "987-65-4320"
