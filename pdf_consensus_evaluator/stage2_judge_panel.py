"""Stage 2: Parallel LLM-as-a-Judge Panel with thinking mode enabled."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from pdf_consensus_evaluator.gemini_client import GeminiClient
from pdf_consensus_evaluator.models import (
    CheckResult,
    JudgeEvaluation,
    JudgeReport,
    RubricSpec,
)

logger = logging.getLogger("pdf_consensus_evaluator.stage2")


class Stage2JudgePanel:
    """Orchestrates parallel LLM judges with reasoning/thinking enabled."""

    DEFAULT_MODEL = "gemini-3.6-flash"
    DEFAULT_THINKING_BUDGET = 2048
    DEFAULT_NUM_JUDGES = 5
    DEFAULT_JUDGE_TIMEOUT = 90.0

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        model: str = DEFAULT_MODEL,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        num_judges: int = DEFAULT_NUM_JUDGES,
        judge_timeout: float = DEFAULT_JUDGE_TIMEOUT,
        base_temperature: float = 0.7,
    ):
        self.client = client or GeminiClient()
        self.model = model
        self.thinking_budget = thinking_budget
        self.num_judges = num_judges
        self.judge_timeout = judge_timeout
        self.base_temperature = base_temperature

    def build_judge_prompt(
        self,
        rubric: RubricSpec,
        candidate_extraction: Dict[str, Any],
        judge_id: str,
    ) -> Tuple[str, str]:
        """Constructs the system instruction and prompt for an independent LLM judge."""
        system_instruction = (
            f"You are an expert, meticulous LLM Judge evaluating an extracted JSON payload against a source PDF document.\n"
            f"Document Type: {rubric.document_type}\n"
            f"Rubric Version: {rubric.rubric_version}\n\n"
            f"Your Core Duties:\n"
            f"1. Syntactic Verification: Check if candidate values satisfy type, regex patterns, token count, and required constraints.\n"
            f"2. Semantic Grounding: Deeply inspect the source PDF visual pages to confirm that the extracted value factually and verbatim matches the document text without hallucination, transposition, field mix-up, or misattribution.\n\n"
            f"For every evaluated field, produce:\n"
            f"- field_name: Target attribute name.\n"
            f"- syntactic_check: 'PASS' or 'FAIL'.\n"
            f"- grounding_check: 'PASS' or 'FAIL'.\n"
            f"- verdict: 'PASS' if and only if both syntactic and grounding checks pass; otherwise 'FAIL'.\n"
            f"- failure_mode: 'NONE' if PASS, or one of the rubric failure modes / ['FORMAT_MISMATCH', 'MISSING_VALUE', 'HALLUCINATION', 'WRONG_ENTITY', 'UNREADABLE_SOURCE'].\n"
            f"- justification: Concise reasoning citing exact page number, section, or line from the PDF.\n"
            f"- proposed_correction: The actual correct value from the PDF if candidate was marked FAIL, or null if PASS.\n\n"
            f"Output must be a valid JSON object containing an 'evaluations' list of objects conforming to the schema."
        )

        criteria_dump = []
        for c in rubric.extraction_criteria:
            crit = {
                "field_name": c.field_name,
                "syntactic_rules": {
                    "required": c.syntactic_rules.required,
                    "type": c.syntactic_rules.type,
                    "min_tokens": c.syntactic_rules.min_tokens,
                    "format_regex": c.syntactic_rules.format_regex,
                    "masking_allowed": c.syntactic_rules.masking_allowed,
                    "expected_subfields": c.syntactic_rules.expected_subfields,
                },
                "semantic_grounding_rules": {
                    "source_section": c.semantic_grounding_rules.source_section,
                    "verification_criteria": c.semantic_grounding_rules.verification_criteria,
                    "failure_modes": c.semantic_grounding_rules.failure_modes,
                },
            }
            criteria_dump.append(crit)

        prompt = (
            f"JUDGE ID: {judge_id}\n\n"
            f"--- VERIFICATION RUBRIC ---\n"
            f"{json.dumps(criteria_dump, indent=2)}\n\n"
            f"--- CANDIDATE EXTRACTION TO EVALUATE ---\n"
            f"{json.dumps(candidate_extraction, indent=2)}\n\n"
            f"Evaluate every field in the candidate extraction against the attached PDF and rubric rules.\n"
            f"Return JSON format: {{\"evaluations\": [...]}}"
        )

        return system_instruction, prompt

    async def evaluate_single_judge(
        self,
        judge_idx: int,
        pdf_path: str,
        rubric: RubricSpec,
        candidate_extraction: Dict[str, Any],
    ) -> JudgeReport:
        """Executes a single judge evaluation with thinking tokens and timeouts."""
        judge_id = f"judge_{judge_idx + 1}"
        start_time = time.time()

        # Slight temperature variation for sample diversity across judges
        temperature = max(0.2, min(0.9, self.base_temperature + (judge_idx - 2) * 0.05))

        system_instruction, prompt = self.build_judge_prompt(
            rubric=rubric,
            candidate_extraction=candidate_extraction,
            judge_id=judge_id,
        )

        try:
            raw_response = await asyncio.wait_for(
                self.client.generate_content_async(
                    model=self.model,
                    system_instruction=system_instruction,
                    prompt=prompt,
                    pdf_path=pdf_path,
                    thinking_budget=self.thinking_budget,
                    temperature=temperature,
                    response_json=True,
                ),
                timeout=self.judge_timeout,
            )

            exec_time = time.time() - start_time
            eval_list_raw = raw_response.get("evaluations", [])
            
            evaluations = [
                JudgeEvaluation.from_dict(item) for item in eval_list_raw
            ]

            logger.info(
                "Judge Panel: %s completed in %.2fs (%d fields evaluated)",
                judge_id,
                exec_time,
                len(evaluations),
            )

            return JudgeReport(
                judge_id=judge_id,
                model_name=self.model,
                thinking_budget=self.thinking_budget,
                evaluations=evaluations,
                raw_response=json.dumps(raw_response),
                execution_time_seconds=exec_time,
                error=None,
            )

        except asyncio.TimeoutError:
            exec_time = time.time() - start_time
            error_msg = f"Judge {judge_id} timed out after {self.judge_timeout}s"
            logger.error(error_msg)
            return JudgeReport(
                judge_id=judge_id,
                model_name=self.model,
                thinking_budget=self.thinking_budget,
                evaluations=[],
                execution_time_seconds=exec_time,
                error=error_msg,
            )

        except Exception as exc:
            exec_time = time.time() - start_time
            error_msg = f"Judge {judge_id} encountered exception: {exc}"
            logger.error(error_msg)
            return JudgeReport(
                judge_id=judge_id,
                model_name=self.model,
                thinking_budget=self.thinking_budget,
                evaluations=[],
                execution_time_seconds=exec_time,
                error=error_msg,
            )

    async def evaluate_panel_async(
        self,
        pdf_path: str,
        rubric: RubricSpec,
        candidate_extraction: Dict[str, Any],
    ) -> List[JudgeReport]:
        """Runs all 5 judge instances concurrently using asyncio.gather."""
        logger.info(
            "Judge Panel: Launching %d parallel judges (model=%s, thinking_budget=%d)...",
            self.num_judges,
            self.model,
            self.thinking_budget,
        )

        tasks = [
            self.evaluate_single_judge(
                judge_idx=i,
                pdf_path=pdf_path,
                rubric=rubric,
                candidate_extraction=candidate_extraction,
            )
            for i in range(self.num_judges)
        ]

        reports = await asyncio.gather(*tasks)
        return list(reports)

    def evaluate_panel(
        self,
        pdf_path: str,
        rubric: RubricSpec,
        candidate_extraction: Dict[str, Any],
    ) -> List[JudgeReport]:
        """Synchronous entry point for Stage 2 panel evaluation."""
        return asyncio.run(
            self.evaluate_panel_async(
                pdf_path=pdf_path,
                rubric=rubric,
                candidate_extraction=candidate_extraction,
            )
        )
