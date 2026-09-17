"""Tests verifying prompt construction for Stage 1 extractor and Stage 2 judge panel,
specifically ensuring handwritten text ink color discrimination, comb tick template subtraction,
and the Adversarial Forensic Protocol are correctly injected into system instructions.
"""

import os
import pytest

from pdf_consensus_evaluator.models import FailureMode, RubricSpec
from pdf_consensus_evaluator.stage1_extractor import Stage1Extractor
from pdf_consensus_evaluator.stage2_judge_panel import Stage2JudgePanel

SAMPLE_RUBRIC_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/rubric_spec_healthcare_patient_intake_form.json",
)


def test_stage1_extractor_prompt_includes_ink_color_guidelines():
    """Verifies that Stage 1 build_system_instruction includes handwritten ink color and template subtraction."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    extractor = Stage1Extractor()
    system_instruction = extractor.build_system_instruction(rubric)

    # Verify ink color discrimination instructions are present
    assert "ink color" in system_instruction
    assert "stroke tone" in system_instruction
    assert "touching or overlapping" in system_instruction
    assert "handwritten" in system_instruction.lower()

    # Verify comb tick and template subtraction rules
    assert "Template Subtraction" in system_instruction
    assert "comb tick" in system_instruction
    assert "mentally subtract" in system_instruction

    # Verify preservation of digital PDF text processing
    assert "digital PDFs" in system_instruction

    # Verify standard form layout guidelines remain intact
    assert "Checkboxes & Selection Controls" in system_instruction
    assert "Strikethroughs & Corrections" in system_instruction
    assert "Ambiguity Resolution" in system_instruction


def test_stage2_judge_prompt_includes_adversarial_forensic_protocol():
    """Verifies that Stage 2 build_judge_prompt includes the Adversarial Forensic Protocol."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    judge_panel = Stage2JudgePanel()
    candidate = {"patient_full_name": "Eleanor Vance Sterling"}

    system_instruction, prompt = judge_panel.build_judge_prompt(
        rubric=rubric,
        candidate_extraction=candidate,
        judge_id="judge_1",
    )

    # Verify Adversarial Forensic Protocol instructions
    assert "Adversarial Forensic Protocol" in system_instruction
    assert "Template Baseline Mapping" in system_instruction
    assert "Forensic Stroke Isolation" in system_instruction
    assert "Conflation Falsification Check" in system_instruction
    assert "TEMPLATE_INK_CONFLATION" in system_instruction
    assert "proposed_correction" in system_instruction

    # Verify prompt contains judge ID and rubric criteria
    assert "JUDGE ID: judge_1" in prompt
    assert "patient_full_name" in prompt


def test_failure_mode_template_ink_conflation():
    """Verifies that TEMPLATE_INK_CONFLATION is defined in the FailureMode enum."""
    assert FailureMode.TEMPLATE_INK_CONFLATION == "TEMPLATE_INK_CONFLATION"
    assert FailureMode.TEMPLATE_INK_CONFLATION.value == "TEMPLATE_INK_CONFLATION"
