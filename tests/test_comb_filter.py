"""Tests for Surgical ROI Cropping & Comb-Line Suppression Pipeline.

Validates:
1. utils.comb_filter: crop_field_roi, suppress_comb_lines, and prepare_crop_payload.
2. models: field_type and bounding_box support in FieldExtractionCriteria.
3. stage1_extractor: high-resolution cropped comb box prompt construction and surgical intercept.
4. stage2_judge_panel: adversarial forensic comb crop audit prompt and candidate overturning.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image
import pytest

from pdf_consensus_evaluator.models import (
    CheckResult,
    FieldExtractionCriteria,
    RubricSpec,
    SemanticGroundingRule,
    SyntacticRule,
)
from pdf_consensus_evaluator.stage1_extractor import Stage1Extractor
from pdf_consensus_evaluator.stage2_judge_panel import Stage2JudgePanel
from utils.comb_filter import (
    auto_detect_comb_rois,
    crop_field_roi,
    prepare_crop_payload,
    suppress_comb_lines,
    suppress_vertical_ticks,
)


def test_crop_field_roi_coordinates_and_padding():
    """Verifies that crop_field_roi correctly scales normalized coordinates with 3% padding."""
    # Create a 200x100 synthetic test image (H=200, W=100)
    img = Image.new("RGB", (100, 200), color=(255, 255, 255))
    
    # Target bbox: ymin=0.2, xmin=0.1, ymax=0.4, xmax=0.5
    # Box height = 0.2 (40 px), box width = 0.4 (40 px)
    # 3% padding: pad_y = 200 * 0.03 = 6 px, pad_x = 100 * 0.03 = 3 px
    cropped = crop_field_roi(img, (0.2, 0.1, 0.4, 0.5), padding_ratio=0.03)
    
    # Expected shape: (H, W, C) -> (52, 46, 3)
    assert cropped.shape[:2] == (52, 46)


def test_crop_field_roi_clips_at_boundaries():
    """Verifies that crop_field_roi clamps bounds to [0, W] and [0, H]."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    cropped = crop_field_roi(img, (0.0, 0.0, 1.0, 1.0), padding_ratio=0.05)
    assert cropped.shape[:2] == (100, 100)


def test_suppress_comb_lines_bleaches_gray_ticks_and_preserves_blue_ink():
    """Verifies that suppress_comb_lines bleaches neutral gray ticks to 255 while preserving blue ink."""
    # Create 100x100 image
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    
    # Draw gray tick mark: RGB [170, 170, 170] (low saturation, mid value)
    arr[40:60, 20:30] = [170, 170, 170]
    
    # Draw saturated blue ink: RGB [20, 50, 200] (high saturation, pen stroke)
    arr[40:60, 70:80] = [20, 50, 200]
    
    img = Image.fromarray(arr, "RGB")
    filtered = suppress_comb_lines(img, saturation_threshold=28, value_threshold=225)
    
    # Gray tick area should be bleached to background white (all 255)
    gray_tick_patch = filtered[40:60, 20:30]
    assert np.all(gray_tick_patch == 255), "Gray tick marks must be bleached to white (255)"
    
    # Blue ink area should preserve blue color and not be bleached to 255
    blue_patch = filtered[40:60, 70:80]
    assert not np.all(blue_patch == 255), "Blue pen ink must not be bleached"


def test_suppress_comb_lines_grayscale_mode():
    """Verifies that suppress_comb_lines handles grayscale images gracefully."""
    arr = np.full((50, 50), 200, dtype=np.uint8)
    img = Image.fromarray(arr, "L")
    filtered = suppress_comb_lines(img)
    assert filtered.shape[:2] == (50, 50)


def test_prepare_crop_payload_returns_valid_png_bytes():
    """Verifies that prepare_crop_payload crops, filters, and encodes to valid PNG bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        img.save(tmp_path, format="PNG")
        
    try:
        png_bytes = prepare_crop_payload(tmp_path, (0.2, 0.2, 0.6, 0.6))
        assert isinstance(png_bytes, bytes)
        # PNG magic header
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        
        # Verify decoding
        decoded = Image.open(io.BytesIO(png_bytes))
        assert decoded.format == "PNG"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_models_support_field_type_and_bounding_box():
    """Verifies that FieldExtractionCriteria deserializes field_type and bounding_box."""
    data = {
        "field_name": "patient_registration_id",
        "field_type": "segmented_comb_box",
        "bounding_box": [0.35, 0.68, 0.38, 0.88],
        "syntactic_rules": {
            "required": True,
            "type": "string",
            "format_regex": "^[A-Z0-9]{6}$",
        },
        "semantic_grounding_rules": {
            "source_section": "HEADER",
            "verification_criteria": "Extract 6-character registration ID from comb boxes.",
            "failure_modes": ["TEMPLATE_INK_CONFLATION"],
        },
    }
    
    crit = FieldExtractionCriteria.from_dict(data)
    assert crit.field_type == "segmented_comb_box"
    assert crit.bounding_box == (0.35, 0.68, 0.38, 0.88)
    assert crit.syntactic_rules.field_type == "segmented_comb_box"
    assert crit.syntactic_rules.bounding_box == (0.35, 0.68, 0.38, 0.88)


def test_stage1_comb_crop_prompt_construction():
    """Verifies that Stage 1 build_cropped_comb_prompt specifies slot-by-slot transcription."""
    extractor = Stage1Extractor()
    sys_prompt, user_prompt = extractor.build_cropped_comb_prompt("registration_id")
    
    assert "HIGH-RESOLUTION CROPPED form field" in sys_prompt
    assert "CURVATURE VS. TICKS" in sys_prompt
    assert "SLOT MAPPING" in sys_prompt
    assert "segmented_slots" in sys_prompt
    assert "extracted_text" in sys_prompt
    assert "registration_id" in user_prompt


def test_stage2_comb_crop_judge_prompt_construction():
    """Verifies that Stage 2 build_cropped_comb_judge_prompt contains adversarial audit rules."""
    panel = Stage2JudgePanel()
    sys_prompt, user_prompt = panel.build_cropped_comb_judge_prompt(
        field_name="registration_id",
        candidate_text="0397H4",
        judge_id="judge_1",
    )
    
    assert "Adversarial Forensic Judge" in sys_prompt
    assert "Grid Alignment & Tick Mark Mapping" in sys_prompt
    assert "Conflation Falsification Check" in sys_prompt
    assert "Adversarial Overrule" in sys_prompt
    assert "audit_verdict" in sys_prompt
    assert "OVERTURNED" in sys_prompt
    assert "final_verified_text" in sys_prompt
    assert "0397H4" in user_prompt
    assert "judge_1" in user_prompt


def test_stage2_judge_comb_audit_overturns_conflation():
    """Verifies that Stage 2 evaluate_single_judge executes the comb crop audit and overturns candidate."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        img.save(tmp_path, format="PNG")
        
    try:
        rubric = RubricSpec(
            rubric_version="1.0",
            document_type="IntakeForm",
            extraction_criteria=[
                FieldExtractionCriteria(
                    field_name="registration_id",
                    field_type="segmented_comb_box",
                    bounding_box=(0.1, 0.1, 0.3, 0.5),
                    syntactic_rules=SyntacticRule(required=True, type="string"),
                    semantic_grounding_rules=SemanticGroundingRule(
                        source_section="HEADER",
                        verification_criteria="Registration ID",
                        failure_modes=["TEMPLATE_INK_CONFLATION"],
                    ),
                )
            ],
        )
        
        # Candidate value was misread as 0397H4 (C touching tick was read as 4)
        candidate = {"registration_id": "0397H4"}
        
        # Mock GeminiClient
        mock_client = MagicMock()
        
        # First call: main document evaluation returns PASS initially
        main_eval_resp = {
            "evaluations": [
                {
                    "field_name": "registration_id",
                    "syntactic_check": "PASS",
                    "grounding_check": "PASS",
                    "verdict": "PASS",
                    "failure_mode": "NONE",
                    "justification": "Matches candidate",
                    "proposed_correction": None,
                }
            ]
        }
        
        # Second call: comb crop audit returns OVERTURNED to 0397HC
        audit_resp = {
            "audit_verdict": "OVERTURNED",
            "conflated_slots": [
                {
                    "slot_index": 5,
                    "candidate_char": "4",
                    "falsification_reason": "Letter C touches baseline tick mark, giving the illusion of a closed 4 stem",
                    "corrected_char": "C",
                }
            ],
            "final_verified_text": "0397HC",
            "confidence_score": 0.98,
            "forensic_notes": "Forensic comb audit confirmed slot 6 is C touching vertical tick mark.",
        }
        
        mock_client.generate_content_async = AsyncMock(side_effect=[main_eval_resp, audit_resp])
        
        panel = Stage2JudgePanel(client=mock_client)
        report = asyncio.run(panel.evaluate_single_judge(
            judge_idx=0,
            document_path=tmp_path,
            rubric=rubric,
            candidate_extraction=candidate,
        ))
        
        assert len(report.evaluations) == 1
        reg_eval = report.evaluations[0]
        
        # Verify the candidate was overturned
        assert reg_eval.field_name == "registration_id"
        assert reg_eval.grounding_check == CheckResult.FAIL
        assert reg_eval.verdict == CheckResult.FAIL
        assert reg_eval.failure_mode == "TEMPLATE_INK_CONFLATION"
        assert reg_eval.proposed_correction == "0397HC"
        assert "Forensic comb audit" in reg_eval.justification
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_stage1_extractor_comb_intercept():
    """Verifies that Stage 1 extract runs high-res comb crop extraction for segmented_comb_box fields."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        img.save(tmp_path, format="PNG")
        
    try:
        rubric = RubricSpec(
            rubric_version="1.0",
            document_type="IntakeForm",
            extraction_criteria=[
                FieldExtractionCriteria(
                    field_name="registration_id",
                    field_type="segmented_comb_box",
                    bounding_box=(0.1, 0.1, 0.3, 0.5),
                    syntactic_rules=SyntacticRule(required=True, type="string"),
                    semantic_grounding_rules=SemanticGroundingRule(
                        source_section="HEADER",
                        verification_criteria="Registration ID",
                    ),
                )
            ],
        )
        
        mock_client = MagicMock()
        # Primary full-page extraction returns candidate with 0397H4
        primary_resp = {"registration_id": "0397H4"}
        # High-res comb crop extraction returns slot-level output with corrected text 0397HC
        crop_resp = {
            "segmented_slots": [
                {"slot_number": 1, "detected_character": "0", "stroke_confidence": "HIGH"},
                {"slot_number": 2, "detected_character": "3", "stroke_confidence": "HIGH"},
                {"slot_number": 3, "detected_character": "9", "stroke_confidence": "HIGH"},
                {"slot_number": 4, "detected_character": "7", "stroke_confidence": "HIGH"},
                {"slot_number": 5, "detected_character": "H", "stroke_confidence": "HIGH"},
                {"slot_number": 6, "detected_character": "C", "stroke_confidence": "HIGH"},
            ],
            "extracted_text": "0397HC",
        }
        mock_client.generate_content.side_effect = [primary_resp, crop_resp]
        
        extractor = Stage1Extractor(client=mock_client)
        candidate = extractor.extract(document_path=tmp_path, rubric=rubric)
        
        assert candidate["registration_id"] == "0397HC"
        assert mock_client.generate_content.call_count == 2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_suppress_vertical_ticks_black_ink():
    """Verifies that suppress_vertical_ticks bleaches thin vertical ticks on black ink while preserving curved strokes."""
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    # Draw curved black ink stroke ('C' arc)
    for angle in np.linspace(-np.pi * 0.7, np.pi * 0.7, 50):
        y = int(50 + 20 * np.sin(angle))
        x = int(50 + 20 * np.cos(angle))
        arr[max(0, y - 1):min(100, y + 2), max(0, x - 1):min(100, x + 2)] = [30, 30, 30]

    # Draw thin vertical tick mark intersecting the bottom curve (width 2, height 20)
    arr[65:85, 55:57] = [40, 40, 40]

    cleaned = suppress_vertical_ticks(arr, min_tick_height=7, max_tick_width=2)

    # Tick mark area below the curve should be bleached to 255
    assert np.all(cleaned[75:85, 55:57] == 255), "Vertical tick mark must be bleached"
    # Back of the 'C' curve should remain dark ink
    assert np.any(cleaned[45:55, 65:75] < 100), "Curved ink stroke must be preserved"


def test_auto_detect_comb_rois_on_synthetic_form():
    """Verifies that auto_detect_comb_rois detects candidate comb box rows."""
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    # Draw baseline
    img[200:202, 250:550] = [50, 50, 50]
    # Draw periodic comb tick marks
    for x in range(250, 551, 50):
        img[180:200, x:x + 2] = [50, 50, 50]

    rois = auto_detect_comb_rois(img, min_width_ratio=0.08)
    assert len(rois) >= 1, "Should detect at least one comb ROI"
    ymin, xmin, ymax, xmax = rois[0]
    assert 0.25 <= ymin <= 0.35, f"ymin {ymin} should be near 0.30"
    assert 0.28 <= xmin <= 0.35, f"xmin {xmin} should be near 0.31"
    assert 0.65 <= xmax <= 0.75, f"xmax {xmax} should be near 0.69"


def test_stage2_judge_auto_detect_comb_roi_fallback():
    """Verifies that Stage 2 falls back to auto_detect_comb_rois when bounding_box is None."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        # Create image with comb box
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        img[200:202, 250:550] = [50, 50, 50]
        for x in range(250, 551, 50):
            img[180:200, x:x + 2] = [50, 50, 50]
        Image.fromarray(img).save(tmp_path, format="PNG")

    try:
        # Rubric has segmented_comb_box but NO bounding_box
        rubric = RubricSpec(
            rubric_version="1.0",
            document_type="Form",
            extraction_criteria=[
                FieldExtractionCriteria(
                    field_name="comb_id",
                    field_type="segmented_comb_box",
                    bounding_box=None,
                    syntactic_rules=SyntacticRule(required=True, type="string"),
                    semantic_grounding_rules=SemanticGroundingRule(
                        source_section="ID",
                        verification_criteria="ID",
                        failure_modes=["TEMPLATE_INK_CONFLATION"],
                    ),
                )
            ],
        )
        candidate = {"comb_id": "0397H4"}
        mock_client = MagicMock()
        main_resp = {
            "evaluations": [
                {
                    "field_name": "comb_id",
                    "syntactic_check": "PASS",
                    "grounding_check": "PASS",
                    "verdict": "PASS",
                    "failure_mode": "NONE",
                    "justification": "Candidate value",
                    "proposed_correction": None,
                }
            ]
        }
        audit_resp = {
            "audit_verdict": "OVERTURNED",
            "conflated_slots": [
                {
                    "slot_index": 5,
                    "candidate_char": "4",
                    "falsification_reason": "Letter C touches baseline tick",
                    "corrected_char": "C",
                }
            ],
            "final_verified_text": "0397HC",
            "confidence_score": 0.99,
            "forensic_notes": "Overturned 4 to C",
        }
        mock_client.generate_content_async = AsyncMock(side_effect=[main_resp, audit_resp])

        panel = Stage2JudgePanel(client=mock_client)
        report = asyncio.run(panel.evaluate_single_judge(
            judge_idx=0,
            document_path=tmp_path,
            rubric=rubric,
            candidate_extraction=candidate,
        ))

        assert len(report.evaluations) == 1
        ev = report.evaluations[0]
        assert ev.verdict == CheckResult.FAIL
        assert ev.failure_mode == "TEMPLATE_INK_CONFLATION"
        assert ev.proposed_correction == "0397HC"
        # Verify two calls were made (main eval + comb crop audit)
        assert mock_client.generate_content_async.call_count == 2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_stage1_extractor_maintains_zero_thinking():
    """Verifies that Stage 1 primary extractor strictly uses thinking_budget=0."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(tmp_path, format="PNG")

    try:
        rubric = RubricSpec(
            rubric_version="1.0",
            document_type="Form",
            extraction_criteria=[
                FieldExtractionCriteria(
                    field_name="name",
                    syntactic_rules=SyntacticRule(required=True, type="string"),
                    semantic_grounding_rules=SemanticGroundingRule(
                        source_section="SEC1",
                        verification_criteria="Name",
                    ),
                )
            ],
        )
        mock_client = MagicMock()
        mock_client.generate_content.return_value = {"name": "Test User"}

        extractor = Stage1Extractor(client=mock_client)
        extractor.extract(document_path=tmp_path, rubric=rubric)

        assert mock_client.generate_content.call_count == 1
        call_kwargs = mock_client.generate_content.call_args.kwargs
        assert call_kwargs.get("thinking_budget") == 0, "Stage 1 must strictly have thinking_budget=0"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_stage2_judge_model_and_thinking_configuration():
    """Verifies that Stage 2 judge panel defaults to gemini-3.8-flash and uses thinking_budget=2048."""
    panel = Stage2JudgePanel()
    assert panel.model == "gemini-3.8-flash"
    assert panel.thinking_budget >= 2048

