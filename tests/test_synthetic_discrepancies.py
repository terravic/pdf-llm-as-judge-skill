"""Synthetic Discrepancy Tests verifying failure detection on corrupted candidate extractions."""

import json
import os
import pytest

from pdf_consensus_evaluator.models import (
    CheckResult,
    JudgeEvaluation,
    JudgeReport,
    RubricSpec,
    StatusClassification,
    RoutingAction,
)
from pdf_consensus_evaluator.stage3_consensus import ConsensusEngine
from pdf_consensus_evaluator.pipeline import ExtractionConsensusPipeline

SAMPLE_RUBRIC_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/rubric_spec_healthcare_patient_intake_form.json",
)
SAMPLE_PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/healthcare_patient_intake_form.pdf",
)


def test_injected_invalid_ssn_rejection():
    """Verifies that an injected erroneous SSN triggers majority rejection and failure mode attribution."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    engine = ConsensusEngine()

    # Artificially corrupted candidate payload
    corrupted_candidate = {
        "patient_full_name": "Eleanor Vance Sterling",
        "date_of_birth": "1984-11-23",
        "social_security_number": "000-00-0000",  # Injected hallucinated / invalid SSN
        "residential_address": {
            "street": "742 Evergreen Terrace, Apt 3B",
            "city": "St. Paul",
            "state": "MN",
            "zip_code": "55102"
        },
        "insurance_member_id": "XMN-849201840-01",
        "primary_diagnosis_code": "C50.912"
    }

    # Simulate 5 judges evaluating the corrupted SSN
    # 5 out of 5 judges fail the SSN due to HALLUCINATION and DIGIT_TRANSPOSITION
    judge_reports = []
    for i in range(5):
        evals = [
            JudgeEvaluation(
                field_name="social_security_number",
                syntactic_check=CheckResult.PASS,
                grounding_check=CheckResult.FAIL,
                verdict=CheckResult.FAIL,
                failure_mode="HALLUCINATION" if i % 2 == 0 else "DIGIT_TRANSPOSITION",
                justification=f"Judge {i+1}: Source PDF page 1 states SSN is '987-65-4320', not '000-00-0000'.",
                proposed_correction="987-65-4320",
            ),
            JudgeEvaluation(
                field_name="patient_full_name",
                syntactic_check=CheckResult.PASS,
                grounding_check=CheckResult.PASS,
                verdict=CheckResult.PASS,
                failure_mode="NONE",
                justification="Verified on page 1.",
            ),
        ]
        judge_reports.append(
            JudgeReport(
                judge_id=f"judge_{i+1}",
                model_name="gemini-3.6-flash",
                thinking_budget=2048,
                evaluations=evals,
            )
        )

    report = engine.aggregate(
        rubric=rubric,
        candidate_extraction=corrupted_candidate,
        judge_reports=judge_reports,
    )

    # Verify that SSN is flagged in exceptions and rejected
    assert report.rejected_field_count >= 1
    assert "social_security_number" not in report.accepted_payload

    ssn_exception = next(
        (e for e in report.exceptions if e["field_name"] == "social_security_number"),
        None,
    )
    assert ssn_exception is not None
    assert ssn_exception["status"] == StatusClassification.REJECTED.value
    assert ssn_exception["recommended_routing"] == RoutingAction.FLAG_FOR_RESUBMISSION.value
    assert (
        "HALLUCINATION" in ssn_exception["failure_modes"]
        or "DIGIT_TRANSPOSITION" in ssn_exception["failure_modes"]
    )
    assert "987-65-4320" in ssn_exception["proposed_corrections"]


def test_injected_facility_address_confusion():
    """Verifies that confusing patient residential address with facility address is rejected."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    engine = ConsensusEngine()

    # Injected facility address
    corrupted_candidate = {
        "residential_address": {
            "street": "1200 Healthcare Blvd, Suite 400",
            "city": "Minneapolis",
            "state": "MN",
            "zip_code": "55415"
        }
    }

    judge_reports = []
    for i in range(5):
        evals = [
            JudgeEvaluation(
                field_name="residential_address",
                syntactic_check=CheckResult.PASS,
                grounding_check=CheckResult.FAIL,
                verdict=CheckResult.FAIL,
                failure_mode="FACILITY_ADDRESS_CONFUSED_AS_PATIENT",
                justification="Extracted clinic address instead of patient residential address from Section 1.",
                proposed_correction={
                    "street": "742 Evergreen Terrace, Apt 3B",
                    "city": "St. Paul",
                    "state": "MN",
                    "zip_code": "55102"
                },
            )
        ]
        judge_reports.append(
            JudgeReport(
                judge_id=f"judge_{i+1}",
                model_name="gemini-3.6-flash",
                thinking_budget=2048,
                evaluations=evals,
            )
        )

    report = engine.aggregate(
        rubric=rubric,
        candidate_extraction=corrupted_candidate,
        judge_reports=judge_reports,
    )

    assert "residential_address" not in report.accepted_payload
    addr_exception = next(
        e for e in report.exceptions if e["field_name"] == "residential_address"
    )
    assert addr_exception["status"] == StatusClassification.REJECTED.value
    assert "FACILITY_ADDRESS_CONFUSED_AS_PATIENT" in addr_exception["failure_modes"]
