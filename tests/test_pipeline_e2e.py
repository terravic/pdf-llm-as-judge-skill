"""End-to-end integration tests for Multi-Agent PDF Extraction and Consensus Pipeline."""

import json
import os
from unittest.mock import patch
import pytest

from pdf_consensus_evaluator.models import RubricSpec, StatusClassification
from pdf_consensus_evaluator.pipeline import ExtractionConsensusPipeline

SAMPLE_RUBRIC_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/rubric_spec_healthcare_patient_intake_form.json",
)
SAMPLE_PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/healthcare_patient_intake_form.pdf",
)
EXPECTED_CANDIDATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/expected_candidate_healthcare_patient_intake_form.json",
)

MOCK_EXTRACTION = {
    "patient_full_name": "Eleanor Vance Sterling",
    "date_of_birth": "1984-11-23",
    "social_security_number": "987-65-4320",
    "residential_address": {
        "street": "742 Evergreen Terrace, Apt 3B",
        "city": "St. Paul",
        "state": "MN",
        "zip_code": "55102"
    },
    "insurance_member_id": "XMN-849201840-01",
    "primary_diagnosis_code": "C50.912"
}

MOCK_JUDGE_EVALUATION = {
    "evaluations": [
        {
            "field_name": "patient_full_name",
            "syntactic_check": "PASS",
            "grounding_check": "PASS",
            "verdict": "PASS",
            "failure_mode": "NONE",
            "justification": "Verified legal name on page 1.",
            "proposed_correction": None
        },
        {
            "field_name": "date_of_birth",
            "syntactic_check": "PASS",
            "grounding_check": "PASS",
            "verdict": "PASS",
            "failure_mode": "NONE",
            "justification": "Verified DOB 1984-11-23 on page 1.",
            "proposed_correction": None
        },
        {
            "field_name": "social_security_number",
            "syntactic_check": "PASS",
            "grounding_check": "PASS",
            "verdict": "PASS",
            "failure_mode": "NONE",
            "justification": "Verified SSN 987-65-4320 on page 1.",
            "proposed_correction": None
        },
        {
            "field_name": "residential_address",
            "syntactic_check": "PASS",
            "grounding_check": "PASS",
            "verdict": "PASS",
            "failure_mode": "NONE",
            "justification": "Verified address on page 1.",
            "proposed_correction": None
        },
        {
            "field_name": "insurance_member_id",
            "syntactic_check": "PASS",
            "grounding_check": "PASS",
            "verdict": "PASS",
            "failure_mode": "NONE",
            "justification": "Verified Member ID on page 2.",
            "proposed_correction": None
        },
        {
            "field_name": "primary_diagnosis_code",
            "syntactic_check": "PASS",
            "grounding_check": "PASS",
            "verdict": "PASS",
            "failure_mode": "NONE",
            "justification": "Verified ICD-10 C50.912 on page 2.",
            "proposed_correction": None
        }
    ]
}


def mock_generate_content_dispatcher(model, prompt, **kwargs):
    if "PRIMARY MULTIMODAL EXTRACTION" in prompt or "extract target fields" in prompt:
        return MOCK_EXTRACTION
    return MOCK_JUDGE_EVALUATION


@patch("pdf_consensus_evaluator.gemini_client.GeminiClient.generate_content", side_effect=mock_generate_content_dispatcher)
def test_pipeline_e2e(mock_gen):
    """Runs full pipeline with sample PDF and rubric."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    pipeline = ExtractionConsensusPipeline(api_key="test-key", num_judges=5)

    report = pipeline.run(
        pdf_path=SAMPLE_PDF_PATH,
        rubric=rubric,
    )

    assert report.document_type == "Healthcare_Patient_Intake_Form"
    assert report.rubric_version == "2.0"
    assert report.total_fields == 6
    assert report.accepted_field_count == 6
    assert report.rejected_field_count == 0
    assert report.contested_field_count == 0

    with open(EXPECTED_CANDIDATE_PATH, "r", encoding="utf-8") as f:
        expected = json.load(f)

    for k, v in expected.items():
        assert k in report.accepted_payload
        assert report.accepted_payload[k] == v

    assert len(report.audit_trail) == 6
    for field_name in expected.keys():
        assert field_name in report.consensus_breakdown
        assert report.consensus_breakdown[field_name]["status"] == StatusClassification.UNANIMOUS_PASS.value


@patch("pdf_consensus_evaluator.gemini_client.GeminiClient.generate_content", side_effect=mock_generate_content_dispatcher)
def test_pipeline_with_pre_extracted_candidate(mock_gen):
    """Tests evaluating a pre-extracted candidate payload directly."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    with open(EXPECTED_CANDIDATE_PATH, "r", encoding="utf-8") as f:
        candidate_data = json.load(f)

    pipeline = ExtractionConsensusPipeline(api_key="test-key", num_judges=5)
    report = pipeline.run(
        pdf_path=SAMPLE_PDF_PATH,
        rubric=rubric,
        candidate_extraction=candidate_data,
    )

    assert report.accepted_field_count == 6
    assert report.candidate_extraction == candidate_data


@patch("pdf_consensus_evaluator.gemini_client.GeminiClient.generate_content", side_effect=mock_generate_content_dispatcher)
def test_generate_dashboard_html(mock_gen, tmp_path):
    """Tests generating self-contained HTML Dashboard."""
    from pdf_consensus_evaluator.dashboard_generator import generate_dashboard_html

    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    pipeline = ExtractionConsensusPipeline(api_key="test-key", num_judges=5)
    report = pipeline.run(pdf_path=SAMPLE_PDF_PATH, rubric=rubric)

    out_html = tmp_path / "dashboard.html"
    html_content = generate_dashboard_html(
        report=report,
        pdf_path=SAMPLE_PDF_PATH,
        rubric=rubric,
        output_html_path=str(out_html),
    )

    assert out_html.exists()
    assert "Multi-Agent PDF Extraction & Consensus Evaluator" in html_content
    assert "theme-toggle" in html_content
    assert "data-theme" in html_content
    assert len(html_content) > 10000


CANCER_RUBRIC_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/rubric_spec_cancer_screening_lab_report.json",
)
CANCER_PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/cancer_screening_lab_report.pdf",
)
CANCER_CANDIDATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/expected_candidate_cancer_screening_lab_report.json",
)


def mock_cancer_generate_dispatcher(model, prompt, **kwargs):
    with open(CANCER_CANDIDATE_PATH, "r", encoding="utf-8") as f:
        cancer_candidate = json.load(f)

    if "PRIMARY MULTIMODAL EXTRACTION" in prompt or "extract target fields" in prompt:
        return cancer_candidate

    rubric = RubricSpec.from_json_file(CANCER_RUBRIC_PATH)
    evals = []
    for criterion in rubric.extraction_criteria:
        evals.append({
            "field_name": criterion.field_name,
            "syntactic_check": "PASS",
            "grounding_check": "PASS",
            "verdict": "PASS",
            "failure_mode": "NONE",
            "justification": f"Verified field {criterion.field_name} from clinical report.",
            "proposed_correction": None
        })
    return {"evaluations": evals}


@patch("pdf_consensus_evaluator.gemini_client.GeminiClient.generate_content", side_effect=mock_cancer_generate_dispatcher)
def test_cancer_screening_pipeline_e2e(mock_gen, tmp_path):
    """Tests full pipeline run on cancer screening lab report."""
    from pdf_consensus_evaluator.dashboard_generator import generate_dashboard_html

    assert os.path.exists(CANCER_PDF_PATH)
    assert os.path.exists(CANCER_RUBRIC_PATH)
    assert os.path.exists(CANCER_CANDIDATE_PATH)

    rubric = RubricSpec.from_json_file(CANCER_RUBRIC_PATH)
    pipeline = ExtractionConsensusPipeline(api_key="test-key", num_judges=5)
    report = pipeline.run(pdf_path=CANCER_PDF_PATH, rubric=rubric)

    assert report.total_fields == len(rubric.extraction_criteria)
    assert report.accepted_field_count == len(rubric.extraction_criteria)
    assert report.contested_field_count == 0
    assert report.rejected_field_count == 0

    out_html = tmp_path / "cancer_dashboard.html"
    generate_dashboard_html(
        report=report,
        pdf_path=CANCER_PDF_PATH,
        rubric=rubric,
        output_html_path=str(out_html),
    )
    assert out_html.exists()

