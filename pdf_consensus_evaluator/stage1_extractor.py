"""Stage 1: Primary Multimodal Extractor with zero-thinking budget."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from pdf_consensus_evaluator.gemini_client import GeminiClient
from pdf_consensus_evaluator.models import RubricSpec

logger = logging.getLogger("pdf_consensus_evaluator.stage1")


class Stage1Extractor:
    """Primary multimodal extractor using high-throughput Gemini model with thinking disabled."""

    DEFAULT_MODEL = "gemini-3.8-flash"

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
    ):
        self.client = client or GeminiClient()
        self.model = model
        self.temperature = temperature

    def build_system_instruction(self, rubric: RubricSpec) -> str:
        """Constructs strict multimodal extraction prompt and schema instructions."""
        schema_fields = []
        for c in rubric.extraction_criteria:
            subfields_info = ""
            if c.syntactic_rules.expected_subfields:
                subfields_info = f" with subfields {c.syntactic_rules.expected_subfields}"
            regex_info = ""
            if c.syntactic_rules.format_regex:
                regex_info = f" conforming to regex /{c.syntactic_rules.format_regex}/"
            
            field_desc = (
                f"- '{c.field_name}' (type: {c.syntactic_rules.type}{subfields_info}{regex_info}): "
                f"Extract from '{c.semantic_grounding_rules.source_section}'. "
                f"Criteria: {c.semantic_grounding_rules.verification_criteria}"
            )
            schema_fields.append(field_desc)

        schema_text = "\n".join(schema_fields)

        return (
            f"You are a high-precision multimodal document extractor for document type '{rubric.document_type}'.\n"
            f"Your task is to inspect the attached document (PDF or image) and extract the specified target fields into a valid JSON object.\n\n"
            f"Target Fields:\n{schema_text}\n\n"
            f"Extraction Rules & Multimodal Form Guidelines:\n"
            f"1. Form Structures, Handwritten Text & Template Subtraction: Accurately read printed text, fill-in-the-blank entries, handwritten cursive/print, numbers, and notes within structured or semi-structured form layouts. On forms with segmented character boxes, comb tick marks, or baseline dividers, mentally subtract rigid pre-printed template elements—do not conflate a vertical comb divider or baseline tick mark with a character stroke (e.g., do not misread an 'o' touching a baseline tick as an 'a', or a 'C'/'L' touching a vertical comb tick as a '4'). Examine pen ink color, stroke tone, and curvature to isolate true handwritten characters and numbers from touching or overlapping printed form lines. For standard digital PDFs or documents without handwriting, perform standard high-precision extraction without altering normal processing.\n"
            f"2. Checkboxes & Selection Controls: Accurately identify checkmarks, crossed boxes, filled circles/radio buttons, and distinct checked vs unchecked states.\n"
            f"3. Strikethroughs & Corrections: If handwritten text has been crossed out or corrected with an updated value nearby, extract the final intended correction.\n"
            f"4. Ambiguity Resolution: Use visual context and schema rules (e.g., date formats, numeric ranges) to disambiguate unclear handwriting without hallucinating.\n"
            f"5. Return ONLY a valid JSON object where keys correspond exactly to the requested field names without markdown preamble or commentary."
        )

    def build_cropped_comb_prompt(self, field_name: str) -> Tuple[str, str]:
        """Constructs prompt for high-resolution cropped comb field extraction."""
        system_instruction = (
            "You are an expert handwriting transcription engine evaluating a HIGH-RESOLUTION CROPPED form field.\n\n"
            "INSTRUCTIONS:\n"
            "1. TARGET CONTENT: Transcribe human pen ink only.\n"
            "2. IGNORE FORM ARTIFACTS: Disregard any residual bounding boxes, tick marks, or horizontal baseline rules.\n"
            "3. CURVATURE VS. TICKS:\n"
            "   - A continuous counter-clockwise curved stroke open to the right is the letter 'C' (or 'c').\n"
            "   - Do NOT combine an open curved arc with an intersecting bottom baseline tick to form a digit '4'.\n"
            "   - An isolated oval loop is '0' or 'O', not an 'a' or 'd'.\n"
            "4. SLOT MAPPING: Transcribe characters slot-by-slot from left to right.\n\n"
            "OUTPUT SCHEMA (JSON ONLY):\n"
            "{\n"
            '  "segmented_slots": [\n'
            "    {\n"
            '      "slot_number": <int>,\n'
            '      "detected_character": "<char or null>",\n'
            '      "stroke_confidence": "HIGH" | "MEDIUM" | "LOW"\n'
            "    }\n"
            "  ],\n"
            '  "extracted_text": "<concatenated non-null characters>"\n'
            "}"
        )
        prompt = (
            f"Transcribe the handwritten characters in this high-resolution cropped field for '{field_name}'. "
            f"Follow all instructions and return valid JSON adhering to the schema."
        )
        return system_instruction, prompt

    def _is_image_file(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path.lower())[1]
        return ext in [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".heic", ".heif", ".gif"]

    def extract(
        self,
        document_path: Optional[str] = None,
        rubric: Optional[RubricSpec] = None,
        pdf_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous primary extraction from document (PDF or image)."""
        target_path = document_path or pdf_path
        if not target_path:
            raise ValueError("document_path or pdf_path must be provided")
        if rubric is None:
            raise ValueError("rubric must be provided")

        system_instruction = self.build_system_instruction(rubric)
        prompt = (
            f"PRIMARY MULTIMODAL EXTRACTION:\n"
            f"Extract all specified fields for document type '{rubric.document_type}' "
            f"from the attached document according to the rubric criteria. Return valid JSON."
        )

        logger.info(
            "Primary Extraction: Running model=%s, thinking_budget=0, temp=%.2f",
            self.model,
            self.temperature,
        )

        candidate = self.client.generate_content(
            model=self.model,
            system_instruction=system_instruction,
            prompt=prompt,
            document_path=target_path,
            thinking_budget=0,
            temperature=self.temperature,
            response_json=True,
        )

        # Surgical Intercept: Check for segmented_comb_box fields on image inputs
        if self._is_image_file(target_path):
            comb_fields = [
                c for c in rubric.extraction_criteria
                if (c.field_type == "segmented_comb_box" or getattr(c.syntactic_rules, "field_type", None) == "segmented_comb_box")
                and (c.bounding_box or getattr(c.syntactic_rules, "bounding_box", None))
            ]
            for cf in comb_fields:
                bbox = cf.bounding_box or getattr(cf.syntactic_rules, "bounding_box", None)
                if bbox:
                    try:
                        from utils.comb_filter import prepare_crop_payload
                        crop_bytes = prepare_crop_payload(target_path, bbox)
                        crop_b64 = base64.b64encode(crop_bytes).decode("utf-8")
                        crop_sys, crop_prompt = self.build_cropped_comb_prompt(cf.field_name)

                        logger.info("Primary Extraction: Running high-res comb crop extraction on field '%s'", cf.field_name)
                        crop_result = self.client.generate_content(
                            model=self.model,
                            system_instruction=crop_sys,
                            prompt=crop_prompt,
                            document_b64=crop_b64,
                            mime_type="image/png",
                            thinking_budget=0,
                            temperature=self.temperature,
                            response_json=True,
                        )
                        ext_text = crop_result.get("extracted_text")
                        if not ext_text and "segmented_slots" in crop_result:
                            ext_text = "".join(
                                str(s.get("detected_character", ""))
                                for s in crop_result["segmented_slots"]
                                if s.get("detected_character")
                            )
                        if ext_text:
                            candidate[cf.field_name] = ext_text
                    except Exception as e:
                        logger.warning("Comb crop extraction failed for '%s': %s", cf.field_name, e)

        return candidate

    async def extract_async(
        self,
        document_path: Optional[str] = None,
        rubric: Optional[RubricSpec] = None,
        pdf_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Asynchronous primary extraction from document (PDF or image)."""
        target_path = document_path or pdf_path
        if not target_path:
            raise ValueError("document_path or pdf_path must be provided")
        if rubric is None:
            raise ValueError("rubric must be provided")

        system_instruction = self.build_system_instruction(rubric)
        prompt = (
            f"PRIMARY MULTIMODAL EXTRACTION:\n"
            f"Extract all specified fields for document type '{rubric.document_type}' "
            f"from the attached document according to the rubric criteria. Return valid JSON."
        )

        logger.info(
            "Primary Extraction (Async): Running model=%s, thinking_budget=0, temp=%.2f",
            self.model,
            self.temperature,
        )

        candidate = await self.client.generate_content_async(
            model=self.model,
            system_instruction=system_instruction,
            prompt=prompt,
            document_path=target_path,
            thinking_budget=0,
            temperature=self.temperature,
            response_json=True,
        )

        # Surgical Intercept: Check for segmented_comb_box fields on image inputs
        if self._is_image_file(target_path):
            comb_fields = [
                c for c in rubric.extraction_criteria
                if (c.field_type == "segmented_comb_box" or getattr(c.syntactic_rules, "field_type", None) == "segmented_comb_box")
                and (c.bounding_box or getattr(c.syntactic_rules, "bounding_box", None))
            ]
            for cf in comb_fields:
                bbox = cf.bounding_box or getattr(cf.syntactic_rules, "bounding_box", None)
                if bbox:
                    try:
                        from utils.comb_filter import prepare_crop_payload
                        crop_bytes = prepare_crop_payload(target_path, bbox)
                        crop_b64 = base64.b64encode(crop_bytes).decode("utf-8")
                        crop_sys, crop_prompt = self.build_cropped_comb_prompt(cf.field_name)

                        logger.info("Primary Extraction (Async): Running high-res comb crop extraction on field '%s'", cf.field_name)
                        crop_result = await self.client.generate_content_async(
                            model=self.model,
                            system_instruction=crop_sys,
                            prompt=crop_prompt,
                            document_b64=crop_b64,
                            mime_type="image/png",
                            thinking_budget=0,
                            temperature=self.temperature,
                            response_json=True,
                        )
                        ext_text = crop_result.get("extracted_text")
                        if not ext_text and "segmented_slots" in crop_result:
                            ext_text = "".join(
                                str(s.get("detected_character", ""))
                                for s in crop_result["segmented_slots"]
                                if s.get("detected_character")
                            )
                        if ext_text:
                            candidate[cf.field_name] = ext_text
                    except Exception as e:
                        logger.warning("Comb crop extraction failed for '%s': %s", cf.field_name, e)

        return candidate
