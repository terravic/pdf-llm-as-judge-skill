"""Tests for Stage 3 deterministic consensus engine."""

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


def create_mock_judge_reports(
    num_judges: int,
    field_verdicts: dict[str, list[CheckResult]],
    failure_modes: dict[str, list[str]] = None,
    justifications: dict[str, list[str]] = None,
) -> list[JudgeReport]:
    """Helper to generate mock JudgeReport lists for testing consensus logic."""
    failure_modes = failure_modes or {}
    justifications = justifications or {}
    reports = []

    for j_idx in range(num_judges):
        evals = []
        for field_name, verdicts in field_verdicts.items():
            verdict = verdicts[j_idx] if j_idx < len(verdicts) else CheckResult.PASS
            f_mode = (
                failure_modes.get(field_name, ["NONE"] * num_judges)[j_idx]
                if verdict == CheckResult.FAIL
                else "NONE"
            )
            if verdict == CheckResult.FAIL:
                default_just = f"Dissent reason from judge {j_idx + 1}"
                just = justifications.get(field_name, [default_just] * num_judges)[j_idx]
            else:
                just = "Verified on page 1"
            evals.append(
                JudgeEvaluation(
                    field_name=field_name,
                    syntactic_check=verdict,
                    grounding_check=verdict,
                    verdict=verdict,
                    failure_mode=f_mode,
                    justification=just,
                    proposed_correction="CorrectValue" if verdict == CheckResult.FAIL else None,
                )
            )

        reports.append(
            JudgeReport(
                judge_id=f"judge_{j_idx + 1}",
                model_name="gemini-3.6-flash",
                thinking_budget=2048,
                evaluations=evals,
            )
        )
    return reports


def test_unanimous_pass_5_of_5():
    engine = ConsensusEngine()
    reports = create_mock_judge_reports(
        num_judges=5,
        field_verdicts={"patient_name": [CheckResult.PASS] * 5},
    )
    fc = engine.evaluate_field("patient_name", "Jane Doe", reports)
    assert fc.agreement_count == 5
    assert fc.total_judges == 5
    assert fc.agreement_ratio == 1.0
    assert fc.status == StatusClassification.UNANIMOUS_PASS
    assert fc.routing_action == RoutingAction.AUTO_ACCEPT
    assert fc.accepted_value == "Jane Doe"
    assert len(fc.dissent_reasons) == 0


def test_majority_pass_4_of_5():
    engine = ConsensusEngine()
    reports = create_mock_judge_reports(
        num_judges=5,
        field_verdicts={
            "patient_name": [
                CheckResult.PASS,
                CheckResult.PASS,
                CheckResult.PASS,
                CheckResult.PASS,
                CheckResult.FAIL,
            ]
        },
        failure_modes={"patient_name": ["NONE", "NONE", "NONE", "NONE", "NAME_TRUNCATED"]},
        justifications={"patient_name": ["", "", "", "", "Middle name initial missing."]},
    )
    fc = engine.evaluate_field("patient_name", "Jane Doe", reports)
    assert fc.agreement_count == 4
    assert fc.total_judges == 5
    assert fc.agreement_ratio == 0.8
    assert fc.status == StatusClassification.MAJORITY_PASS
    assert fc.routing_action == RoutingAction.AUTO_ACCEPT_WITH_WARNING
    assert fc.accepted_value == "Jane Doe"
    assert len(fc.dissent_reasons) == 1
    assert "NAME_TRUNCATED" in fc.failure_modes_detected


def test_contested_split_verdict_3_of_5():
    engine = ConsensusEngine()
    reports = create_mock_judge_reports(
        num_judges=5,
        field_verdicts={
            "diagnosis_code": [
                CheckResult.PASS,
                CheckResult.PASS,
                CheckResult.PASS,
                CheckResult.FAIL,
                CheckResult.FAIL,
            ]
        },
        failure_modes={"diagnosis_code": ["NONE", "NONE", "NONE", "SECONDARY_CODE_EXTRACTED_AS_PRIMARY", "SECONDARY_CODE_EXTRACTED_AS_PRIMARY"]},
    )
    fc = engine.evaluate_field("diagnosis_code", "I10", reports)
    assert fc.agreement_count == 3
    assert fc.total_judges == 5
    assert fc.agreement_ratio == 0.6
    assert fc.status == StatusClassification.CONTESTED
    assert fc.routing_action == RoutingAction.ROUTE_TO_HITL
    assert fc.accepted_value is None
    assert len(fc.dissent_reasons) == 2


def test_rejected_verdict_2_of_5_and_lower():
    engine = ConsensusEngine()
    reports = create_mock_judge_reports(
        num_judges=5,
        field_verdicts={
            "ssn": [
                CheckResult.PASS,
                CheckResult.PASS,
                CheckResult.FAIL,
                CheckResult.FAIL,
                CheckResult.FAIL,
            ]
        },
        failure_modes={"ssn": ["NONE", "NONE", "DIGIT_TRANSPOSITION", "DIGIT_TRANSPOSITION", "DIGIT_TRANSPOSITION"]},
    )
    fc = engine.evaluate_field("ssn", "987-65-4321", reports)
    assert fc.agreement_count == 2
    assert fc.total_judges == 5
    assert fc.agreement_ratio == 0.4
    assert fc.status == StatusClassification.REJECTED
    assert fc.routing_action == RoutingAction.FLAG_FOR_RESUBMISSION
    assert fc.accepted_value is None
    assert "DIGIT_TRANSPOSITION" in fc.failure_modes_detected


def test_degraded_quorum_resilience():
    engine = ConsensusEngine()
    # 4 judges successful, 1 errored out
    reports = create_mock_judge_reports(
        num_judges=4,
        field_verdicts={"field_a": [CheckResult.PASS] * 4},
    )
    # Add an errored judge
    reports.append(
        JudgeReport(
            judge_id="judge_5",
            model_name="gemini-3.6-flash",
            thinking_budget=2048,
            evaluations=[],
            error="Rate limit exceeded",
        )
    )
    fc = engine.evaluate_field("field_a", "Val", reports)
    assert fc.agreement_count == 4
    assert fc.total_judges == 4
    assert fc.status == StatusClassification.UNANIMOUS_PASS
    assert fc.accepted_value == "Val"
