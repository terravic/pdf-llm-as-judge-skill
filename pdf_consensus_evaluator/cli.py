"""Command-line interface for Multi-Agent PDF Extraction and Consensus Evaluator."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from pdf_consensus_evaluator.models import RubricSpec
from pdf_consensus_evaluator.pipeline import ExtractionConsensusPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configures structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Document/PDF Extraction & Consensus Evaluator with LLM Judges"
    )
    parser.add_argument(
        "--document",
        "--pdf",
        "--image",
        dest="document",
        type=str,
        required=True,
        help="Path to the input document (PDF, JPEG, PNG, WebP, etc.)",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        required=True,
        help="Path to the JSON rubric specification file",
    )
    parser.add_argument(
        "--candidate-json",
        type=str,
        default=None,
        help="Optional path to pre-extracted candidate JSON to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output JSON report",
    )
    parser.add_argument(
        "--dashboard",
        type=str,
        default=None,
        help="Path to generate self-contained HTML UI Dashboard (defaults to <output_stem>_dashboard.html or ui/index.html)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable automatic HTML UI Dashboard generation",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API Key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--extractor-model",
        type=str,
        default="gemini-3.8-flash",
        help="Model for primary extraction (default: gemini-3.8-flash)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-3.6-flash",
        help="Model for judge panel (default: gemini-3.6-flash)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=2048,
        help="Reasoning budget token count for judges (default: 2048)",
    )
    parser.add_argument(
        "--num-judges",
        type=int,
        default=5,
        help="Number of concurrent judges in panel (default: 5)",
    )
    parser.add_argument(
        "--min-pass-ratio",
        type=float,
        default=0.8,
        help="Minimum agreement ratio for auto-accept (default: 0.8)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("pdf_consensus_evaluator.cli")

    if not os.path.exists(args.document):
        logger.error("Document file not found: %s", args.document)
        return 1

    if not os.path.exists(args.rubric):
        logger.error("Rubric file not found: %s", args.rubric)
        return 1

    try:
        rubric = RubricSpec.from_json_file(args.rubric)
    except Exception as e:
        logger.error("Failed to parse rubric file: %s", e)
        return 1

    candidate_data = None
    if args.candidate_json:
        if not os.path.exists(args.candidate_json):
            logger.error("Candidate JSON file not found: %s", args.candidate_json)
            return 1
        with open(args.candidate_json, "r", encoding="utf-8") as f:
            candidate_data = json.load(f)

    pipeline = ExtractionConsensusPipeline(
        api_key=args.api_key,
        extractor_model=args.extractor_model,
        judge_model=args.judge_model,
        thinking_budget=args.thinking_budget,
        num_judges=args.num_judges,
        min_pass_ratio=args.min_pass_ratio,
    )

    try:
        report = pipeline.run(
            document_path=args.document,
            rubric=rubric,
            candidate_extraction=candidate_data,
        )
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e, exc_info=args.verbose)
        return 1

    json_output = report.to_json(indent=2)

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        logger.info("Report successfully saved to: %s", args.output)
    else:
        print(json_output)

    dashboard_path = None
    if not args.no_dashboard:
        if args.dashboard:
            dashboard_path = args.dashboard
        elif args.output:
            if args.output.endswith(".json"):
                dashboard_path = args.output[:-5] + "_dashboard.html"
            else:
                dashboard_path = f"{args.output}_dashboard.html"
        else:
            dashboard_path = "ui/index.html"

    if dashboard_path:
        from pdf_consensus_evaluator.dashboard_generator import generate_dashboard_html
        generate_dashboard_html(
            report=report,
            document_path=args.document,
            rubric=rubric,
            output_html_path=dashboard_path,
        )
        logger.info("Interactive UI Dashboard saved to: %s", dashboard_path)

    # Print summary table
    print("\n--- Pipeline Summary ---")
    print(f"Document Type:      {report.document_type}")
    print(f"Rubric Version:     {report.rubric_version}")
    print(f"Total Fields:       {report.total_fields}")
    print(f"Accepted Fields:    {report.accepted_field_count}")
    print(f"Contested Fields:   {report.contested_field_count}")
    print(f"Rejected Fields:    {report.rejected_field_count}")
    print("------------------------\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
