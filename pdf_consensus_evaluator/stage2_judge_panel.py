"""Stage 2: Parallel LLM-as-a-Judge Panel with thinking mode enabled."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
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

    DEFAULT_MODEL = "gemini-3.8-flash"
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

    @staticmethod
    def _is_image_file(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")

    def build_cropped_comb_judge_prompt(
        self,
        field_name: str,
        candidate_text: str,
        judge_id: str,
    ) -> Tuple[str, str]:
        """Constructs adversarial forensic audit prompt for a cropped segmented comb field."""
        system_instruction = (
            f"You are an Adversarial Forensic Judge auditing a segmented comb field transcription.\n"
            f"Target Field: {field_name}\n"
            f"Candidate Value to Audit: {candidate_text}\n\n"
            f"AUDIT PROTOCOL (EXECUTE STEP-BY-STEP IN YOUR THINKING PROCESS):\n"
            f"1. Blind Independent Tracing: First, look ONLY at the cropped image and mentally trace the pen strokes independently from left to right. Do NOT anchor to or assume the candidate value '{candidate_text}' is correct.\n"
            f"2. Grid Alignment & Tick Mark Mapping: Identify where repeating pre-printed baseline ticks and vertical dividers are located along the cell boundaries.\n"
            f"3. Conflation Falsification Check: Compare your independent ink tracing against candidate '{candidate_text}'. Specifically test whether any candidate character relies on an intersecting baseline tick or vertical divider to justify its identity:\n"
            f"   - Is a candidate '4' actually an open counter-clockwise curved 'C' (or 'c') that merely touched a baseline tick mark?\n"
            f"   - Is a candidate 'a', 'd', or 'q' actually an isolated oval loop 'o' (or '0') that intersected a vertical divider or tick?\n"
            f"   - Is a candidate 'H' actually two strokes, or an 'I'/'1' touching a cell wall?\n"
            f"   - Is a candidate '8' or 'B' an open loop (such as '3' or 'C') touching a cell border?\n"
            f"4. Adversarial Overrule: If the candidate value incorporated any pre-printed template mark into character identity, you MUST set audit_verdict to 'OVERTURNED', describe the falsification in conflated_slots and forensic_notes, and output the true ink-only transcription in final_verified_text.\n"
            f"5. Confirmation: If and only if the candidate value factually matches the true pen ink strokes without conflating template marks, set audit_verdict to 'CONFIRMED'.\n\n"
            f"Output valid JSON conforming to this schema:\n"
            f"{{\n"
            f'  "audit_verdict": "CONFIRMED" | "OVERTURNED",\n'
            f'  "conflated_slots": [\n'
            f"    {{\n"
            f'      "slot_index": <int>,\n'
            f'      "candidate_char": "<char>",\n'
            f'      "falsification_reason": "<explanation of conflation with tick/comb>",\n'
            f'      "corrected_char": "<char>"\n'
            f"    }}\n"
            f"  ],\n"
            f'  "final_verified_text": "<string>",\n'
            f'  "confidence_score": <float 0.0-1.0>,\n'
            f'  "forensic_notes": "<string>"\n'
            f"}}"
        )

        prompt = (
            f"JUDGE ID: {judge_id}\n\n"
            f"Audit the candidate value '{candidate_text}' for field '{field_name}' against the attached cropped comb image.\n"
            f"Return valid JSON conforming to the audit schema."
        )
        return system_instruction, prompt

    def build_judge_prompt(
        self,
        rubric: RubricSpec,
        candidate_extraction: Dict[str, Any],
        judge_id: str,
    ) -> Tuple[str, str]:
        """Constructs the system instruction and prompt for an independent LLM judge."""
        system_instruction = (
            f"You are an expert, meticulous LLM Judge evaluating an extracted JSON payload against a source document (PDF or image).\n"
            f"Document Type: {rubric.document_type}\n"
            f"Rubric Version: {rubric.rubric_version}\n\n"
            f"Your Core Duties:\n"
            f"1. Syntactic Verification: Check if candidate values satisfy type, regex patterns, token count, and required constraints.\n"
            f"2. Semantic Grounding & Adversarial Forensic Verification: Deeply inspect the source visual document/image (including form structures, table grids, checkboxes, handwritten notes/cursive, and strike-through corrections) to confirm that the extracted value factually matches the document without hallucination, transposition, field mix-up, or misattribution.\n"
            f"   - Adversarial Forensic Protocol for Form Fields with Comb Ticks or Segmented Boxes:\n"
            f"     Act as an Adversarial Forensic Judge specializing in document transcription verification to falsify potential extraction errors caused by template-ink conflation. Perform this step-by-step in your thinking process:\n"
            f"     a. Template Baseline Mapping: Identify repeating pre-printed elements (box borders, baseline ticks, comb dividers) and note the exact periodic intervals/spacing along the baseline or grid.\n"
            f"     b. Forensic Stroke Isolation: Mentally subtract all pre-printed template lines from the image. For every character where an ascender, descender, or vertical stem aligns with a known template tick interval, perform a stroke continuity test: does the stroke exhibit genuine pen ink characteristics (natural curvature, consistent ink saturation, pen pressure variations), or does the alleged stroke segment directly overlap the rigid, pre-printed template mark?\n"
            f"     c. Conflation Falsification Check: Specifically test whether the candidate character is an optical artifact of a simpler glyph touching a printed mark (e.g., is an alleged 'a' actually an 'o' touching a baseline tick? Is an alleged '4' actually a 'C' or 'L' touching a vertical comb tick? Is an alleged 'd' or 'q' actually an 'o' intersecting a vertical divider? Is an alleged 'H' actually two characters or an 'I'/'1' touching a divider? Is an alleged 'B' or '8' an open loop touching a cell wall?).\n"
            f"     d. Verdict Determination & Overturn: If any character in the candidate extraction relies on a pre-printed form mark to justify its classification, you MUST set grounding_check to 'FAIL', verdict to 'FAIL', assign failure_mode 'TEMPLATE_INK_CONFLATION', describe the falsification in justification, and output the true ink-only value in proposed_correction.\n"
            f"   - Format Normalization: Standard format normalization required by the rubric (e.g. converting raw or handwritten dates like '2 3 1944' or '2/3/1944' into the required rubric format '02/03/1944' or 'YYYY-MM-DD', trimming whitespace, standardizing delimiters) is EXPECTED and must PASS semantic grounding.\n"
            f"3. Consistency Requirement: If your reasoning confirms that the candidate value is correct according to the rubric format, or if your proposed correction would be identical/equivalent to the candidate extraction, your verdict MUST be 'PASS' and proposed_correction MUST be null.\n\n"
            f"For every evaluated field, produce:\n"
            f"- field_name: Target attribute name.\n"
            f"- syntactic_check: 'PASS' or 'FAIL'.\n"
            f"- grounding_check: 'PASS' or 'FAIL'.\n"
            f"- verdict: 'PASS' if and only if both syntactic and grounding checks pass; otherwise 'FAIL'.\n"
            f"- failure_mode: 'NONE' if PASS, or one of the rubric failure modes / ['FORMAT_MISMATCH', 'MISSING_VALUE', 'HALLUCINATION', 'WRONG_ENTITY', 'UNREADABLE_SOURCE', 'TEMPLATE_INK_CONFLATION', 'ILLEGIBLE_HANDWRITING', 'CHECKBOX_MISINTERPRETED', 'HANDWRITTEN_CORRECTION_IGNORED'].\n"
            f"- justification: Concise reasoning citing exact page number, section, checkbox state, or handwritten line from the document.\n"
            f"- proposed_correction: The actual correct value from the document ONLY if candidate was marked FAIL and differs from candidate value; otherwise null if PASS.\n\n"
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
            f"Evaluate every field in the candidate extraction against the attached document and rubric rules.\n"
            f"Return JSON format: {{\"evaluations\": [...]}}"
        )

        return system_instruction, prompt

    async def evaluate_single_judge(
        self,
        judge_idx: int,
        document_path: Optional[str] = None,
        rubric: Optional[RubricSpec] = None,
        candidate_extraction: Optional[Dict[str, Any]] = None,
        pdf_path: Optional[str] = None,
    ) -> JudgeReport:
        """Executes a single judge evaluation with thinking tokens and timeouts."""
        target_path = document_path or pdf_path
        if not target_path:
            raise ValueError("document_path or pdf_path must be provided")
        if rubric is None or candidate_extraction is None:
            raise ValueError("rubric and candidate_extraction must be provided")

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
                    document_path=target_path,
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

            # Surgical Intercept: Audit segmented_comb_box fields on image inputs
            if self._is_image_file(target_path):
                comb_fields = [
                    c for c in rubric.extraction_criteria
                    if (c.field_type == "segmented_comb_box" or getattr(c.syntactic_rules, "field_type", None) == "segmented_comb_box")
                ]
                auto_rois = None
                for cf in comb_fields:
                    bbox = cf.bounding_box or getattr(cf.syntactic_rules, "bounding_box", None)
                    if not bbox:
                        if auto_rois is None:
                            try:
                                from utils.comb_filter import auto_detect_comb_rois
                                auto_rois = auto_detect_comb_rois(target_path)
                            except Exception:
                                auto_rois = []
                        if auto_rois:
                            bbox = auto_rois[0]

                    cand_val = candidate_extraction.get(cf.field_name)
                    if bbox and cand_val is not None:
                        try:
                            from utils.comb_filter import prepare_crop_payload
                            crop_bytes = prepare_crop_payload(target_path, bbox)
                            crop_b64 = base64.b64encode(crop_bytes).decode("utf-8")
                            crop_sys, crop_prompt = self.build_cropped_comb_judge_prompt(
                                field_name=cf.field_name,
                                candidate_text=str(cand_val),
                                judge_id=judge_id,
                            )
                            logger.info(
                                "Judge Panel: %s running comb crop audit on field '%s' (candidate='%s')",
                                judge_id,
                                cf.field_name,
                                cand_val,
                            )
                            audit_raw = await asyncio.wait_for(
                                self.client.generate_content_async(
                                    model=self.model,
                                    system_instruction=crop_sys,
                                    prompt=crop_prompt,
                                    document_b64=crop_b64,
                                    mime_type="image/png",
                                    thinking_budget=self.thinking_budget,
                                    temperature=temperature,
                                    response_json=True,
                                ),
                                timeout=self.judge_timeout,
                            )
                            audit_verdict = str(audit_raw.get("audit_verdict", "")).upper()
                            final_text = audit_raw.get("final_verified_text")
                            forensic_notes = audit_raw.get("forensic_notes", "")

                            matching_eval = next((e for e in evaluations if e.field_name == cf.field_name), None)
                            if matching_eval is None:
                                matching_eval = JudgeEvaluation(
                                    field_name=cf.field_name,
                                    syntactic_check=CheckResult.PASS,
                                    grounding_check=CheckResult.PASS,
                                    verdict=CheckResult.PASS,
                                    failure_mode="NONE",
                                    justification="",
                                    proposed_correction=None,
                                )
                                evaluations.append(matching_eval)

                            if audit_verdict == "OVERTURNED":
                                matching_eval.grounding_check = CheckResult.FAIL
                                matching_eval.verdict = CheckResult.FAIL
                                matching_eval.failure_mode = "TEMPLATE_INK_CONFLATION"
                                matching_eval.justification = (
                                    forensic_notes
                                    or f"Forensic comb audit overturned candidate '{cand_val}': conflated comb lines/tick marks with pen strokes."
                                )
                                matching_eval.proposed_correction = final_text
                            elif audit_verdict == "CONFIRMED":
                                if matching_eval.failure_mode == "TEMPLATE_INK_CONFLATION":
                                    matching_eval.grounding_check = CheckResult.PASS
                                    matching_eval.verdict = (
                                        CheckResult.PASS if matching_eval.syntactic_check == CheckResult.PASS else CheckResult.FAIL
                                    )
                                    matching_eval.failure_mode = "NONE"
                                    matching_eval.proposed_correction = None
                                if forensic_notes and not matching_eval.justification:
                                    matching_eval.justification = forensic_notes
                        except Exception as e:
                            logger.warning("Comb crop audit failed for judge %s on '%s': %s", judge_id, cf.field_name, e)

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
        document_path: Optional[str] = None,
        rubric: Optional[RubricSpec] = None,
        candidate_extraction: Optional[Dict[str, Any]] = None,
        pdf_path: Optional[str] = None,
    ) -> List[JudgeReport]:
        """Runs all 5 judge instances concurrently using asyncio.gather."""
        target_path = document_path or pdf_path
        if not target_path:
            raise ValueError("document_path or pdf_path must be provided")
        if rubric is None or candidate_extraction is None:
            raise ValueError("rubric and candidate_extraction must be provided")

        logger.info(
            "Judge Panel: Launching %d parallel judges (model=%s, thinking_budget=%d)...",
            self.num_judges,
            self.model,
            self.thinking_budget,
        )

        tasks = [
            self.evaluate_single_judge(
                judge_idx=i,
                document_path=target_path,
                rubric=rubric,
                candidate_extraction=candidate_extraction,
            )
            for i in range(self.num_judges)
        ]

        reports = await asyncio.gather(*tasks)
        return list(reports)

    def evaluate_panel(
        self,
        document_path: Optional[str] = None,
        rubric: Optional[RubricSpec] = None,
        candidate_extraction: Optional[Dict[str, Any]] = None,
        pdf_path: Optional[str] = None,
    ) -> List[JudgeReport]:
        """Synchronous entry point for Stage 2 panel evaluation."""
        return asyncio.run(
            self.evaluate_panel_async(
                document_path=document_path or pdf_path,
                rubric=rubric,
                candidate_extraction=candidate_extraction,
            )
        )
