"""Stage 1: Primary Multimodal Extractor with zero-thinking budget."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

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
            f"Your task is to inspect the attached PDF document and extract the specified target fields into a valid JSON object.\n\n"
            f"Target Fields:\n{schema_text}\n\n"
            f"Rules:\n"
            f"1. Return ONLY a valid JSON object where keys correspond exactly to the requested field names.\n"
            f"2. Extract values accurately from the visual PDF pages without fabrication or guessing.\n"
            f"3. Do not include markdown preamble or explanations, only valid JSON."
        )

    def extract(
        self,
        pdf_path: str,
        rubric: RubricSpec,
    ) -> Dict[str, Any]:
        """Synchronous primary extraction from PDF document."""
        system_instruction = self.build_system_instruction(rubric)
        prompt = (
            f"PRIMARY MULTIMODAL EXTRACTION:\n"
            f"Extract all specified fields for document type '{rubric.document_type}' "
            f"from the attached PDF according to the rubric criteria. Return valid JSON."
        )

        logger.info(
            "Primary Extraction: Running model=%s, thinking_budget=0, temp=%.2f",
            self.model,
            self.temperature,
        )

        return self.client.generate_content(
            model=self.model,
            system_instruction=system_instruction,
            prompt=prompt,
            pdf_path=pdf_path,
            thinking_budget=0,
            temperature=self.temperature,
            response_json=True,
        )

    async def extract_async(
        self,
        pdf_path: str,
        rubric: RubricSpec,
    ) -> Dict[str, Any]:
        """Asynchronous primary extraction from PDF document."""
        system_instruction = self.build_system_instruction(rubric)
        prompt = (
            f"PRIMARY MULTIMODAL EXTRACTION:\n"
            f"Extract all specified fields for document type '{rubric.document_type}' "
            f"from the attached PDF according to the rubric criteria. Return valid JSON."
        )

        logger.info(
            "Primary Extraction (Async): Running model=%s, thinking_budget=0, temp=%.2f",
            self.model,
            self.temperature,
        )

        return await self.client.generate_content_async(
            model=self.model,
            system_instruction=system_instruction,
            prompt=prompt,
            pdf_path=pdf_path,
            thinking_budget=0,
            temperature=self.temperature,
            response_json=True,
        )
