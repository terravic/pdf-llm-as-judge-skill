"""Pipeline Orchestrator coordinating Extraction, Judge Panel, and Consensus Engine."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from pdf_consensus_evaluator.gemini_client import GeminiClient
from pdf_consensus_evaluator.models import PipelineReport, RubricSpec
from pdf_consensus_evaluator.stage1_extractor import Stage1Extractor
from pdf_consensus_evaluator.stage2_judge_panel import Stage2JudgePanel
from pdf_consensus_evaluator.stage3_consensus import ConsensusEngine

logger = logging.getLogger("pdf_consensus_evaluator.pipeline")


class ExtractionConsensusPipeline:
    """End-to-end orchestrator for Multi-Agent PDF Extraction and Consensus Evaluation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        extractor_model: str = "gemini-3.8-flash",
        judge_model: str = "gemini-3.6-flash",
        thinking_budget: int = 2048,
        num_judges: int = 5,
        judge_timeout: float = 90.0,
        min_pass_ratio: float = 0.8,
    ):
        self.client = GeminiClient(api_key=api_key)
        self.stage1 = Stage1Extractor(client=self.client, model=extractor_model)
        self.stage2 = Stage2JudgePanel(
            client=self.client,
            model=judge_model,
            thinking_budget=thinking_budget,
            num_judges=num_judges,
            judge_timeout=judge_timeout,
        )
        self.stage3 = ConsensusEngine(min_pass_ratio=min_pass_ratio)

    async def run_async(
        self,
        document_path: Optional[str] = None,
        rubric: Optional[RubricSpec] = None,
        candidate_extraction: Optional[Dict[str, Any]] = None,
        pdf_path: Optional[str] = None,
    ) -> PipelineReport:
        """Asynchronously executes the end-to-end pipeline."""
        target_path = document_path or pdf_path
        if not target_path:
            raise ValueError("document_path or pdf_path must be provided")
        if rubric is None:
            raise ValueError("rubric must be provided")

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Source document not found: {target_path}")

        # Primary Multimodal Extraction
        if candidate_extraction is None:
            logger.info("Executing Primary Multimodal Extraction on %s...", target_path)
            candidate = await self.stage1.extract_async(
                document_path=target_path,
                rubric=rubric,
            )
        else:
            logger.info("Using provided candidate extraction payload.")
            candidate = candidate_extraction

        # Parallel Judge Panel (5x Concurrent Thinking LLM Judges)
        logger.info("Executing Parallel 5x Thinking Judge Panel...")
        judge_reports = await self.stage2.evaluate_panel_async(
            document_path=target_path,
            rubric=rubric,
            candidate_extraction=candidate,
        )

        # Consensus Aggregation and Routing
        logger.info("Executing Deterministic Consensus Aggregation & Routing...")
        report = self.stage3.aggregate(
            rubric=rubric,
            candidate_extraction=candidate,
            judge_reports=judge_reports,
            extractor_model=self.stage1.model,
            judge_model=self.stage2.model,
            thinking_budget=self.stage2.thinking_budget,
        )

        logger.info(
            "Pipeline completed: %d total fields, %d accepted, %d contested, %d rejected.",
            report.total_fields,
            report.accepted_field_count,
            report.contested_field_count,
            report.rejected_field_count,
        )

        return report

    def run(
        self,
        document_path: Optional[str] = None,
        rubric: Optional[RubricSpec] = None,
        candidate_extraction: Optional[Dict[str, Any]] = None,
        pdf_path: Optional[str] = None,
    ) -> PipelineReport:
        """Synchronous wrapper for pipeline execution."""
        return asyncio.run(
            self.run_async(
                document_path=document_path or pdf_path,
                rubric=rubric,
                candidate_extraction=candidate_extraction,
            )
        )
