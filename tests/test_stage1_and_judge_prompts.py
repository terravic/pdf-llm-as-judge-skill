"""Tests verifying prompt construction for Stage 1 extractor and Stage 2 judge panel,
specifically ensuring handwritten text ink color discrimination and form overlap guidelines
are correctly injected into system instructions.
"""

import os
import pytest

from pdf_consensus_evaluator.models import RubricSpec
from pdf_consensus_evaluator.stage1_extractor import Stage1Extractor
from pdf_consensus_evaluator.stage2_judge_panel import Stage2JudgePanel

SAMPLE_RUBRIC_PATH = os.path.join(
    os.path.dirname(__file__),
    "../samples/rubric_spec_healthcare_patient_intake_form.json",
)


def test_stage1_extractor_prompt_includes_ink_color_guidelines():
    """Verifies that Stage 1 build_system_instruction includes handwritten ink color discrimination."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    extractor = Stage1Extractor()
    system_instruction = extractor.build_system_instruction(rubric)

    # Verify ink color discrimination instructions are present
    assert "ink color" in system_instruction
    assert "stroke tone" in system_instruction
    assert "touch or overlap" in system_instruction
    assert "handwritten text" in system_instruction

    # Verify preservation of digital PDF text processing
    assert "digital PDFs" in system_instruction

    # Verify standard form layout guidelines remain intact
    assert "Form Structures & Handwritten Text" in system_instruction
    assert "Checkboxes & Selection Controls" in system_instruction
    assert "Strikethroughs & Corrections" in system_instruction
    assert "Ambiguity Resolution" in system_instruction


def test_stage2_judge_prompt_includes_ink_color_grounding():
    """Verifies that Stage 2 build_judge_prompt includes ink color verification for handwritten form text."""
    rubric = RubricSpec.from_json_file(SAMPLE_RUBRIC_PATH)
    judge_panel = Stage2JudgePanel()
    candidate = {"patient_full_name": "Eleanor Vance Sterling"}

    system_instruction, prompt = judge_panel.build_judge_prompt(
        rubric=rubric,
        candidate_extraction=candidate,
        judge_id="judge_1",
    )

    # Verify ink color and overlapping printed form instructions in judge system instruction
    assert "ink color and tone" in system_instruction
    assert "touching or overlapping printed form lines" in system_instruction
    assert "Semantic Grounding & Normalization" in system_instruction

    # Verify prompt contains judge ID and rubric criteria
    assert "JUDGE ID: judge_1" in prompt
    assert "patient_full_name" in prompt
