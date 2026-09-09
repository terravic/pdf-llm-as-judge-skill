"""Stage 3: Deterministic Consensus Engine and Routing Aggregator."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pdf_consensus_evaluator.models import (
    CheckResult,
    ExceptionDetail,
    FieldConsensus,
    JudgeReport,
    PipelineReport,
    RoutingAction,
    RubricSpec,
    StatusClassification,
)

logger = logging.getLogger("pdf_consensus_evaluator.stage3")


class ConsensusEngine:
    """Deterministic consensus and routing aggregator."""

    def __init__(
        self,
        min_pass_ratio: float = 0.8,  # Default 4/5 (0.8) for auto-accept
        contested_min_ratio: float = 0.5,  # 3/5 (0.6) is contested HITL
    ):
        self.min_pass_ratio = min_pass_ratio
        self.contested_min_ratio = contested_min_ratio

    def evaluate_field(
        self,
        field_name: str,
        candidate_value: Any,
        judge_reports: List[JudgeReport],
    ) -> FieldConsensus:
        """Computes consensus metrics for a single field across all available judge reports."""
        valid_reports = [r for r in judge_reports if r.error is None and r.evaluations]
        total_judges = len(valid_reports)

        if total_judges == 0:
            return FieldConsensus(
                field_name=field_name,
                candidate_value=candidate_value,
                agreement_count=0,
                total_judges=0,
                agreement_ratio=0.0,
                status=StatusClassification.REJECTED,
                confidence_band="Quorum Failure (0 judges)",
                routing_action=RoutingAction.FLAG_FOR_RESUBMISSION,
                accepted_value=None,
                dissent_reasons=["All judge evaluations failed or timed out"],
                failure_modes_detected=["UNREADABLE_SOURCE"],
                proposed_corrections=[],
            )

        passes = 0
        dissent_reasons: List[str] = []
        failure_modes: List[str] = []
        proposed_corrections: List[Any] = []

        for report in valid_reports:
            # Locate evaluation for this field in the judge report
            field_eval = next(
                (ev for ev in report.evaluations if ev.field_name == field_name),
                None,
            )

            if field_eval is not None and field_eval.verdict == CheckResult.PASS:
                passes += 1
            else:
                if field_eval is not None:
                    if field_eval.failure_mode and field_eval.failure_mode != "NONE":
                        failure_modes.append(field_eval.failure_mode)
                    if field_eval.justification:
                        dissent_reasons.append(
                            f"[{report.judge_id}] {field_eval.justification}"
                        )
                    if field_eval.proposed_correction is not None:
                        proposed_corrections.append(field_eval.proposed_correction)
                else:
                    dissent_reasons.append(
                        f"[{report.judge_id}] Field omitted from judge output"
                    )
                    failure_modes.append("MISSING_VALUE")

        ratio = passes / total_judges

        # Status Classification Logic
        if total_judges >= 5:
            if passes == total_judges:
                status = StatusClassification.UNANIMOUS_PASS
                confidence = "Extreme Confidence (>=98%)"
                routing = RoutingAction.AUTO_ACCEPT
                accepted_val = candidate_value
            elif passes == total_judges - 1:
                status = StatusClassification.MAJORITY_PASS
                confidence = "High Confidence (90-95%)"
                routing = RoutingAction.AUTO_ACCEPT_WITH_WARNING
                accepted_val = candidate_value
            elif passes == total_judges - 2:
                status = StatusClassification.CONTESTED
                confidence = "Split Verdict (approx 60%)"
                routing = RoutingAction.ROUTE_TO_HITL
                accepted_val = None
            else:
                status = StatusClassification.REJECTED
                confidence = f"Low Confidence / Failure ({passes}/{total_judges})"
                routing = RoutingAction.FLAG_FOR_RESUBMISSION
                accepted_val = None
        else:
            # Scaled threshold for degraded quorum
            if ratio >= 0.99:
                status = StatusClassification.UNANIMOUS_PASS
                confidence = f"Unanimous Pass ({passes}/{total_judges})"
                routing = RoutingAction.AUTO_ACCEPT
                accepted_val = candidate_value
            elif ratio >= self.min_pass_ratio:
                status = StatusClassification.MAJORITY_PASS
                confidence = f"Majority Pass ({passes}/{total_judges})"
                routing = RoutingAction.AUTO_ACCEPT_WITH_WARNING
                accepted_val = candidate_value
            elif ratio >= self.contested_min_ratio:
                status = StatusClassification.CONTESTED
                confidence = f"Contested ({passes}/{total_judges})"
                routing = RoutingAction.ROUTE_TO_HITL
                accepted_val = None
            else:
                status = StatusClassification.REJECTED
                confidence = f"Rejected ({passes}/{total_judges})"
                routing = RoutingAction.FLAG_FOR_RESUBMISSION
                accepted_val = None

        return FieldConsensus(
            field_name=field_name,
            candidate_value=candidate_value,
            agreement_count=passes,
            total_judges=total_judges,
            agreement_ratio=round(ratio, 3),
            status=status,
            confidence_band=confidence,
            routing_action=routing,
            accepted_value=accepted_val,
            dissent_reasons=dissent_reasons,
            failure_modes_detected=list(set(failure_modes)),
            proposed_corrections=proposed_corrections,
        )

    def aggregate(
        self,
        rubric: RubricSpec,
        candidate_extraction: Dict[str, Any],
        judge_reports: List[JudgeReport],
        extractor_model: str = "gemini-3.8-flash",
        judge_model: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ) -> PipelineReport:
        """Aggregates all field evaluations into a consolidated PipelineReport."""
        if judge_model is None and judge_reports:
            judge_model = judge_reports[0].model_name
        if thinking_budget is None and judge_reports:
            thinking_budget = judge_reports[0].thinking_budget
        resolved_judge_model = judge_model or "gemini-3.6-flash"
        resolved_thinking_budget = thinking_budget if thinking_budget is not None else 2048
        target_fields = [c.field_name for c in rubric.extraction_criteria]
        # Include any extra keys present in candidate extraction
        all_field_names = list(
            dict.fromkeys(target_fields + list(candidate_extraction.keys()))
        )

        field_consensus_list: List[FieldConsensus] = []
        accepted_payload: Dict[str, Any] = {}
        exceptions: List[Dict[str, Any]] = []
        consensus_breakdown: Dict[str, Any] = {}
        audit_trail: List[Dict[str, Any]] = []

        for field_name in all_field_names:
            candidate_val = candidate_extraction.get(field_name)
            fc = self.evaluate_field(
                field_name=field_name,
                candidate_value=candidate_val,
                judge_reports=judge_reports,
            )
            field_consensus_list.append(fc)

            consensus_breakdown[field_name] = {
                "agreement_count": fc.agreement_count,
                "total_judges": fc.total_judges,
                "agreement_ratio": f"{fc.agreement_count}/{fc.total_judges} ({int(fc.agreement_ratio * 100)}%)",
                "status": fc.status.value,
                "confidence_band": fc.confidence_band,
                "routing_action": fc.routing_action.value,
            }

            if fc.status in (StatusClassification.UNANIMOUS_PASS, StatusClassification.MAJORITY_PASS):
                accepted_payload[field_name] = fc.accepted_value
            else:
                exceptions.append({
                    "field_name": field_name,
                    "status": fc.status.value,
                    "candidate_value": candidate_val,
                    "agreement_ratio": f"{fc.agreement_count}/{fc.total_judges}",
                    "failure_modes": fc.failure_modes_detected,
                    "dissent_reasons": fc.dissent_reasons,
                    "proposed_corrections": fc.proposed_corrections,
                    "recommended_routing": fc.routing_action.value,
                })

            # Field audit log
            audit_trail.append({
                "field_name": field_name,
                "candidate_value": candidate_val,
                "status": fc.status.value,
                "agreement_count": fc.agreement_count,
                "total_judges": fc.total_judges,
                "dissent_reasons": fc.dissent_reasons,
                "failure_modes": fc.failure_modes_detected,
                "proposed_corrections": fc.proposed_corrections,
            })

        total_fields = len(field_consensus_list)
        accepted_count = len(accepted_payload)
        contested_count = sum(
            1 for fc in field_consensus_list if fc.status == StatusClassification.CONTESTED
        )
        rejected_count = sum(
            1 for fc in field_consensus_list if fc.status == StatusClassification.REJECTED
        )

        valid_judges_count = sum(1 for r in judge_reports if r.error is None)

        report = PipelineReport(
            document_type=rubric.document_type,
            rubric_version=rubric.rubric_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_fields=total_fields,
            accepted_field_count=accepted_count,
            contested_field_count=contested_count,
            rejected_field_count=rejected_count,
            quorum_metadata={
                "requested_judges": len(judge_reports),
                "successful_judges": valid_judges_count,
                "quorum_degraded": valid_judges_count < len(judge_reports),
                "judge_errors": [
                    {"judge_id": r.judge_id, "error": r.error}
                    for r in judge_reports
                    if r.error is not None
                ],
            },
            candidate_extraction=candidate_extraction,
            accepted_payload=accepted_payload,
            exceptions=exceptions,
            consensus_breakdown=consensus_breakdown,
            audit_trail=audit_trail,
            extractor_model=extractor_model,
            judge_model=resolved_judge_model,
            thinking_budget=resolved_thinking_budget,
        )

        return report
