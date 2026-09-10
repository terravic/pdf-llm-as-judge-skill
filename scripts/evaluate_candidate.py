#!/usr/bin/env python3
"""Evaluation-only script running 5x Judge Panel and Consensus Engine on candidate JSON."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure root directory is in python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pdf_consensus_evaluator.models import RubricSpec
from pdf_consensus_evaluator.pipeline import ExtractionConsensusPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Candidate Extraction against Source PDF and Rubric using 5x Judge Panel"
    )
    parser.add_argument("--pdf", type=str, required=True, help="Path to source PDF")
    parser.add_argument("--rubric", type=str, required=True, help="Path to rubric JSON")
    parser.add_argument(
        "--candidate-json",
        type=str,
        required=True,
        help="Path to candidate extraction JSON",
    )
    parser.add_argument("--output", type=str, default=None, help="Output report path")
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
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API Key")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-3.6-flash",
        help="Judge model (default: gemini-3.6-flash)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=2048,
        help="Thinking tokens budget (default: 2048)",
    )
    parser.add_argument(
        "--num-judges",
        type=int,
        default=5,
        help="Number of concurrent judges (default: 5)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("evaluate_candidate")

    if not os.path.exists(args.pdf):
        logger.error("PDF file not found: %s", args.pdf)
        return 1
    if not os.path.exists(args.rubric):
        logger.error("Rubric file not found: %s", args.rubric)
        return 1
    if not os.path.exists(args.candidate_json):
        logger.error("Candidate JSON file not found: %s", args.candidate_json)
        return 1

    rubric = RubricSpec.from_json_file(args.rubric)
    with open(args.candidate_json, "r", encoding="utf-8") as f:
        candidate_data = json.load(f)

    pipeline = ExtractionConsensusPipeline(
        api_key=args.api_key,
        judge_model=args.judge_model,
        thinking_budget=args.thinking_budget,
        num_judges=args.num_judges,
    )

    report = pipeline.run(
        pdf_path=args.pdf,
        rubric=rubric,
        candidate_extraction=candidate_data,
    )

    json_str = report.to_json(indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info("Evaluation report saved to: %s", args.output)
    else:
        print(json_str)

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
            pdf_path=args.pdf,
            rubric=rubric,
            output_html_path=dashboard_path,
        )
        logger.info("Interactive UI Dashboard saved to: %s", dashboard_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
