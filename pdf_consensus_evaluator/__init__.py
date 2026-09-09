"""Multi-Agent PDF Extraction and Consensus Evaluator Package."""

from pdf_consensus_evaluator.dashboard_generator import generate_dashboard_html
from pdf_consensus_evaluator.gemini_client import GeminiClient
from pdf_consensus_evaluator.models import (
    CheckResult,
    ExceptionDetail,
    FailureMode,
    FieldConsensus,
    FieldExtractionCriteria,
    JudgeEvaluation,
    JudgeReport,
    PipelineReport,
    RoutingAction,
    RubricSpec,
    SemanticGroundingRule,
    StatusClassification,
    SyntacticRule,
)
from pdf_consensus_evaluator.pipeline import ExtractionConsensusPipeline
from pdf_consensus_evaluator.stage1_extractor import Stage1Extractor
from pdf_consensus_evaluator.stage2_judge_panel import Stage2JudgePanel
from pdf_consensus_evaluator.stage3_consensus import ConsensusEngine

__all__ = [
    "CheckResult",
    "ConsensusEngine",
    "ExceptionDetail",
    "ExtractionConsensusPipeline",
    "FailureMode",
    "FieldConsensus",
    "FieldExtractionCriteria",
    "GeminiClient",
    "generate_dashboard_html",
    "JudgeEvaluation",
    "JudgeReport",
    "PipelineReport",
    "RoutingAction",
    "RubricSpec",
    "SemanticGroundingRule",
    "Stage1Extractor",
    "Stage2JudgePanel",
    "StatusClassification",
    "SyntacticRule",
]
