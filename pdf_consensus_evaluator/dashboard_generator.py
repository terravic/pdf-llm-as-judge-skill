"""HTML Dashboard Generator for Multi-Agent PDF Extraction and Consensus Evaluator."""

from __future__ import annotations

import base64
import html
import json
import os
from typing import Any, Dict, List, Optional

from pdf_consensus_evaluator.models import PipelineReport, RubricSpec


def generate_dashboard_html(
    report: PipelineReport,
    pdf_path: Optional[str] = None,
    rubric: Optional[RubricSpec] = None,
    output_html_path: Optional[str] = None,
) -> str:
    """Generates a self-contained, fully pre-rendered, interactive HTML Dashboard."""
    pdf_b64 = ""
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    rubric_data: Dict[str, Any] = {}
    if rubric:
        rubric_data = {
            "rubric_version": rubric.rubric_version,
            "document_type": rubric.document_type,
            "extraction_criteria": [
                {
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
                for c in rubric.extraction_criteria
            ],
        }

    report_data = report.to_dict()

    html_content = _build_html_template(
        report=report_data,
        pdf_b64=pdf_b64,
        rubric=rubric_data,
    )

    if output_html_path:
        out_dir = os.path.dirname(os.path.abspath(output_html_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


def _build_html_template(
    report: Dict[str, Any],
    pdf_b64: str,
    rubric: Dict[str, Any],
) -> str:
    report_json_escaped = json.dumps(report)
    rubric_json_escaped = json.dumps(rubric)
    pdf_b64_escaped = json.dumps(pdf_b64)

    extractor_model = html.escape(str(report.get("extractor_model") or "gemini-3.8-flash"))
    judge_model = html.escape(str(report.get("judge_model") or "gemini-3.6-flash"))
    thinking_budget = report.get("thinking_budget") if report.get("thinking_budget") is not None else 2048
    requested_judges = report.get("quorum_metadata", {}).get("requested_judges", 5)
    successful_judges = report.get("quorum_metadata", {}).get("successful_judges", requested_judges)
    document_type = html.escape(str(report.get("document_type", "Document")))
    rubric_version = html.escape(str(report.get("rubric_version", "1.0")))
    timestamp = html.escape(str(report.get("timestamp", ""))[:19].replace("T", " "))

    total_fields = report.get("total_fields", 0)
    accepted_fields = report.get("accepted_field_count", 0)
    contested_fields = report.get("contested_field_count", 0)
    rejected_fields = report.get("rejected_field_count", 0)

    # Pre-render Tab 1: Rubric Criteria Cards
    criteria_cards_html = _render_rubric_criteria_html(rubric)

    # Pre-render Tab 2: Candidate Extraction & Config
    candidate_json_str = json.dumps(report.get("candidate_extraction") or {}, indent=2)
    candidate_json_html = html.escape(candidate_json_str)

    # Pre-render Tab 3: 5x Judge Matrix Columns
    judge_matrix_html = _render_judge_matrix_html(report, requested_judges, thinking_budget, judge_model)

    # Pre-render Tab 4: Consensus Table & Validated Payload & Exceptions
    consensus_table_rows_html = _render_consensus_table_rows_html(report)
    accepted_payload_json_str = json.dumps(report.get("accepted_payload") or {}, indent=2)
    accepted_payload_json_html = html.escape(accepted_payload_json_str)
    exceptions_list_html = _render_exceptions_list_html(report)

    # Pre-render Tab 5: Provenance & Audit Trail
    full_audit_json_str = json.dumps(report, indent=2, default=str)
    full_audit_json_html = html.escape(full_audit_json_str)
    audit_trail_table_html = _render_audit_trail_table_html(report)

    pdf_embed_tag = f'<iframe class="pdf-embed-fallback" id="pdf-embed-frame" style="display: none;" src="data:application/pdf;base64,{pdf_b64}"></iframe>' if pdf_b64 else '<div style="padding: 2rem; color: var(--text-muted); font-size: 0.875rem;">No embedded PDF stream provided.</div>'


    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{document_type} - Multi-Agent Consensus Evaluation Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
  <style>
    :root {{
      --bg-primary: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-subtle: #f1f5f9;
      --bg-surface-hover: #e2e8f0;
      --border-color: #e2e8f0;
      --border-focus: #3b82f6;
      --text-primary: #0f172a;
      --text-secondary: #334155;
      --text-muted: #64748b;
      --accent-blue: #2563eb;
      --accent-blue-subtle: #eff6ff;
      --accent-green: #16a34a;
      --accent-green-subtle: #f0fdf4;
      --accent-amber: #d97706;
      --accent-amber-subtle: #fffbeb;
      --accent-red: #dc2626;
      --accent-red-subtle: #fef2f2;
      --accent-purple: #7c3aed;
      --accent-purple-subtle: #f5f3ff;
      --card-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.08);
      --card-shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.08);
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}

    [data-theme="dark"] {{
      --bg-primary: #090d16;
      --bg-surface: #111827;
      --bg-surface-subtle: #1f2937;
      --bg-surface-hover: #374151;
      --border-color: #1f2937;
      --border-focus: #60a5fa;
      --text-primary: #f9fafb;
      --text-secondary: #d1d5db;
      --text-muted: #9ca3af;
      --accent-blue: #3b82f6;
      --accent-blue-subtle: #172554;
      --accent-green: #22c55e;
      --accent-green-subtle: #052e16;
      --accent-amber: #f59e0b;
      --accent-amber-subtle: #451a03;
      --accent-red: #ef4444;
      --accent-red-subtle: #450a0a;
      --accent-purple: #a855f7;
      --accent-purple-subtle: #2e1065;
      --card-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.3);
      --card-shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.4);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      transition: background-color 0.2s ease, color 0.2s ease;
    }}

    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.875rem 1.75rem;
      background-color: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 50;
      box-shadow: var(--card-shadow);
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 0.875rem;
    }}

    .brand-icon {{
      width: 2.25rem;
      height: 2.25rem;
      border-radius: 0.5rem;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      display: flex;
      align-items: center;
      justify-content: center;
      color: #ffffff;
      flex-shrink: 0;
    }}

    .brand-title {{
      font-size: 1.125rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}

    .brand-subtitle {{
      font-size: 0.75rem;
      color: var(--text-muted);
      font-weight: 500;
    }}

    .header-actions {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.45rem 0.875rem;
      font-size: 0.8125rem;
      font-weight: 600;
      border-radius: 0.375rem;
      border: 1px solid var(--border-color);
      background-color: var(--bg-surface);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.15s ease;
      user-select: none;
      text-decoration: none;
    }}

    .btn:hover {{
      background-color: var(--bg-surface-subtle);
      border-color: var(--border-focus);
    }}

    .btn-primary {{
      background-color: var(--accent-blue);
      color: #ffffff;
      border-color: var(--accent-blue);
    }}

    .btn-primary:hover {{
      background-color: #1d4ed8;
      border-color: #1d4ed8;
    }}

    .btn-icon {{
      padding: 0.5rem;
      border-radius: 0.375rem;
    }}

    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 1.5rem 1.75rem 3rem 1.75rem;
    }}

    .control-bar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1.25rem;
      margin-bottom: 1.5rem;
      background-color: var(--bg-surface);
      padding: 1rem 1.25rem;
      border-radius: 0.625rem;
      border: 1px solid var(--border-color);
      box-shadow: var(--card-shadow);
    }}

    .meta-group {{
      display: flex;
      align-items: center;
      gap: 1.75rem;
      flex-wrap: wrap;
    }}

    .meta-item {{
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
    }}

    .meta-label {{
      font-size: 0.6875rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 700;
    }}

    .meta-value {{
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--text-primary);
    }}

    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}

    .stat-card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 0.625rem;
      padding: 1.25rem;
      box-shadow: var(--card-shadow);
      display: flex;
      flex-direction: column;
      gap: 0.375rem;
      transition: transform 0.15s ease;
    }}

    .stat-card:hover {{
      transform: translateY(-2px);
    }}

    .stat-title {{
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .stat-value {{
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.2;
    }}

    .stat-badge {{
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.55rem;
      border-radius: 0.375rem;
      width: fit-content;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}

    .badge-green {{
      background-color: var(--accent-green-subtle);
      color: var(--accent-green);
      border: 1px solid rgba(22, 163, 74, 0.2);
    }}

    .badge-blue {{
      background-color: var(--accent-blue-subtle);
      color: var(--accent-blue);
      border: 1px solid rgba(37, 99, 235, 0.2);
    }}

    .badge-amber {{
      background-color: var(--accent-amber-subtle);
      color: var(--accent-amber);
      border: 1px solid rgba(217, 119, 6, 0.2);
    }}

    .badge-red {{
      background-color: var(--accent-red-subtle);
      color: var(--accent-red);
      border: 1px solid rgba(220, 38, 38, 0.2);
    }}

    .badge-purple {{
      background-color: var(--accent-purple-subtle);
      color: var(--accent-purple);
      border: 1px solid rgba(124, 58, 237, 0.2);
    }}

    .nav-tabs {{
      display: flex;
      gap: 0.5rem;
      border-bottom: 2px solid var(--border-color);
      margin-bottom: 1.5rem;
      overflow-x: auto;
    }}

    .nav-tab {{
      padding: 0.75rem 1.25rem;
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
      background: none;
      border-top: none;
      border-left: none;
      border-right: none;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .nav-tab:hover {{
      color: var(--text-primary);
    }}

    .nav-tab.active {{
      color: var(--accent-blue);
      border-bottom-color: var(--accent-blue);
      background-color: var(--bg-surface-subtle);
      border-top-left-radius: 0.375rem;
      border-top-right-radius: 0.375rem;
    }}

    .tab-panel {{
      display: none;
      animation: fadeIn 0.15s ease-in;
    }}

    .tab-panel.active {{
      display: block;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    .split-layout {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}

    @media (max-width: 1080px) {{
      .split-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    .panel-card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 0.625rem;
      padding: 1.25rem;
      box-shadow: var(--card-shadow);
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}

    .panel-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 0.75rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}

    .panel-title {{
      font-size: 0.9375rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text-primary);
    }}

    .search-input {{
      padding: 0.4rem 0.75rem;
      font-size: 0.8125rem;
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      background-color: var(--bg-surface-subtle);
      color: var(--text-primary);
      width: 200px;
    }}

    .search-input:focus {{
      outline: none;
      border-color: var(--border-focus);
    }}

    .pdf-viewer-container {{
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--bg-surface-subtle);
      border: 1px solid var(--border-color);
      border-radius: 0.5rem;
      padding: 1rem;
      min-height: 560px;
      overflow-y: auto;
      position: relative;
    }}

    .pdf-controls {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .pdf-canvas-wrapper {{
      box-shadow: var(--card-shadow-lg);
      border-radius: 0.25rem;
      overflow: hidden;
      background-color: #ffffff;
      max-width: 100%;
    }}

    #pdf-canvas {{
      display: block;
      max-width: 100%;
      height: auto;
    }}

    .pdf-embed-fallback {{
      width: 100%;
      height: 540px;
      border: none;
      border-radius: 0.375rem;
    }}

    .criteria-list {{
      display: flex;
      flex-direction: column;
      gap: 0.875rem;
      max-height: 560px;
      overflow-y: auto;
      padding-right: 0.25rem;
    }}

    .criteria-card {{
      border: 1px solid var(--border-color);
      border-radius: 0.5rem;
      padding: 1rem;
      background-color: var(--bg-surface-subtle);
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      font-size: 0.875rem;
    }}

    .criteria-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 0.375rem;
    }}

    .criteria-name {{
      font-family: var(--font-mono);
      font-weight: 700;
      color: var(--accent-blue);
      font-size: 0.875rem;
    }}

    .criteria-row {{
      display: flex;
      gap: 0.75rem;
      font-size: 0.8125rem;
      align-items: baseline;
    }}

    .criteria-label {{
      color: var(--text-muted);
      font-weight: 600;
      min-width: 120px;
      flex-shrink: 0;
    }}

    .failure-tag {{
      display: inline-block;
      font-size: 0.7rem;
      font-family: var(--font-mono);
      background-color: var(--accent-red-subtle);
      color: var(--accent-red);
      padding: 0.1rem 0.4rem;
      border-radius: 0.25rem;
      margin: 0.1rem 0.2rem 0.1rem 0;
    }}

    pre.code-block {{
      background-color: var(--bg-surface-subtle);
      border: 1px solid var(--border-color);
      border-radius: 0.5rem;
      padding: 1rem;
      font-family: var(--font-mono);
      font-size: 0.8125rem;
      overflow-x: auto;
      max-height: 520px;
      color: var(--text-primary);
      line-height: 1.5;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}

    .data-table th {{
      text-align: left;
      padding: 0.6875rem 0.875rem;
      background-color: var(--bg-surface-subtle);
      border-bottom: 1px solid var(--border-color);
      color: var(--text-muted);
      font-weight: 700;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .data-table td {{
      padding: 0.75rem 0.875rem;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }}

    .data-table tr:hover td {{
      background-color: var(--bg-surface-hover);
    }}

    .judges-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 1rem;
      margin-top: 0.5rem;
    }}

    .judge-column {{
      background-color: var(--bg-surface-subtle);
      border: 1px solid var(--border-color);
      border-radius: 0.5rem;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}

    .judge-col-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 0.625rem;
    }}

    .judge-name {{
      font-weight: 700;
      font-size: 0.875rem;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }}

    .field-eval-card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      padding: 0.75rem;
      font-size: 0.8125rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      box-shadow: var(--card-shadow);
    }}

    .field-eval-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .field-eval-name {{
      font-weight: 700;
      font-family: var(--font-mono);
      font-size: 0.75rem;
    }}

    .field-eval-checks {{
      display: flex;
      gap: 0.35rem;
      font-size: 0.6875rem;
      margin-top: 0.15rem;
    }}

    .field-eval-just {{
      color: var(--text-secondary);
      font-size: 0.75rem;
      line-height: 1.4;
      margin-top: 0.25rem;
      border-left: 2px solid var(--border-color);
      padding-left: 0.5rem;
    }}

    .exception-card {{
      border: 1px solid var(--border-color);
      border-left: 4px solid var(--accent-amber);
      border-radius: 0.375rem;
      padding: 0.875rem;
      background-color: var(--bg-surface-subtle);
      margin-bottom: 0.75rem;
      font-size: 0.8125rem;
    }}

    .exception-card.rejected {{
      border-left-color: var(--accent-red);
    }}

    .icon {{
      width: 1rem;
      height: 1rem;
      display: inline-block;
      vertical-align: middle;
      stroke-width: 2;
      stroke: currentColor;
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
  </style>
</head>
<body data-theme="light">
  <header>
    <div class="brand">
      <div class="brand-icon">
        <svg class="icon" style="width: 1.25rem; height: 1.25rem;" viewBox="0 0 24 24">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
      </div>
      <div>
        <div class="brand-title">Multi-Agent PDF Extraction & Consensus Evaluator</div>
        <div class="brand-subtitle">5x Thinking LLM Judge Ensemble • Grounding & Consensus Verification</div>
      </div>
    </div>

    <div class="header-actions">
      <button class="btn btn-icon" id="theme-toggle" title="Toggle Light / Dark Mode" aria-label="Toggle Theme">
        <svg id="theme-icon-moon" class="icon" viewBox="0 0 24 24">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
        </svg>
        <svg id="theme-icon-sun" class="icon" style="display: none;" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="23"></line>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
          <line x1="1" y1="12" x2="3" y2="12"></line>
          <line x1="21" y1="12" x2="23" y2="12"></line>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </svg>
      </button>

      <button class="btn" id="export-json-btn">
        <svg class="icon" viewBox="0 0 24 24">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Export Report JSON
      </button>
    </div>
  </header>

  <main>
    <section class="control-bar">
      <div class="meta-group">
        <div class="meta-item">
          <span class="meta-label">Document Type</span>
          <span class="meta-value">{document_type}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Rubric Specification</span>
          <span class="meta-value">v{rubric_version}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Primary Extractor</span>
          <span class="meta-value">{extractor_model} <span class="stat-badge badge-blue" style="font-size: 0.65rem; padding: 0.1rem 0.35rem;">Budget: 0</span></span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Parallel Judge Panel</span>
          <span class="meta-value">{successful_judges}/{requested_judges}x {judge_model} <span class="stat-badge badge-purple" style="font-size: 0.65rem; padding: 0.1rem 0.35rem;">Thinking: {thinking_budget}</span></span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Execution Time</span>
          <span class="meta-value">{timestamp}</span>
        </div>
      </div>
    </section>

    <section class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">Total Evaluated Fields</div>
        <div class="stat-value">{total_fields}</div>
        <div class="stat-badge badge-blue">
          <svg class="icon" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
          100% Extraction Coverage
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Auto-Accepted Fields</div>
        <div class="stat-value" style="color: var(--accent-green);">{accepted_fields}</div>
        <div class="stat-badge badge-green">
          <svg class="icon" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>
          Unanimous / Majority (>=80%)
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Contested Fields (HITL)</div>
        <div class="stat-value" style="color: var(--accent-amber);">{contested_fields}</div>
        <div class="stat-badge badge-amber">
          <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          Split Verdicts (60%)
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Rejected Fields</div>
        <div class="stat-value" style="color: var(--accent-red);">{rejected_fields}</div>
        <div class="stat-badge badge-red">
          <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
          Failed Quorum (&lt;=40%)
        </div>
      </div>
    </section>

    <nav class="nav-tabs" role="tablist">
      <button class="nav-tab active" data-tab="tab-pdf-rubric">
        <svg class="icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
        1. PDF &amp; Rubric Inspector
      </button>
      <button class="nav-tab" data-tab="tab-extraction">
        <svg class="icon" viewBox="0 0 24 24"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
        2. Primary Extraction
      </button>
      <button class="nav-tab" data-tab="tab-judges">
        <svg class="icon" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
        3. Judge Panel Matrix (5x)
      </button>
      <button class="nav-tab" data-tab="tab-consensus">
        <svg class="icon" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
        4. Consensus &amp; Routing
      </button>
      <button class="nav-tab" data-tab="tab-audit">
        <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
        5. Provenance &amp; Audit Trail
      </button>
    </nav>

    <!-- TAB 1: PDF & Rubric Inspector -->
    <div id="tab-pdf-rubric" class="tab-panel active">
      <div class="split-layout">
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
              Multimodal Source PDF Viewer
            </div>
            <div class="pdf-controls">
              <button class="btn btn-icon" id="prev-page" title="Previous Page">
                <svg class="icon" viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>
              </button>
              <span id="page-num-display" style="font-size: 0.8125rem; font-weight: 600;">Page 1</span>
              <button class="btn btn-icon" id="next-page" title="Next Page">
                <svg class="icon" viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>
              </button>
              <button class="btn btn-icon" id="zoom-in" title="Zoom In">+</button>
              <button class="btn btn-icon" id="zoom-out" title="Zoom Out">-</button>
            </div>
          </div>
          <div class="pdf-viewer-container" id="pdf-container">
            <div class="pdf-canvas-wrapper" id="pdf-canvas-wrapper">
              <canvas id="pdf-canvas"></canvas>
            </div>
            {pdf_embed_tag}
          </div>
        </div>

        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
              Target Verification Rubric Specifications ({len(rubric.get("extraction_criteria", []))} Criteria)
            </div>
            <span class="stat-badge badge-blue">v{rubric_version}</span>
          </div>
          <div class="criteria-list" id="rubric-criteria-list">
            {criteria_cards_html}
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: Primary Extraction -->
    <div id="tab-extraction" class="tab-panel">
      <div class="split-layout">
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
              Candidate Extraction Payload
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <span class="stat-badge badge-blue">Thinking Budget: 0</span>
              <button class="btn" id="copy-candidate-btn" onclick="copyCandidateJson()">Copy JSON</button>
            </div>
          </div>
          <pre class="code-block" id="candidate-extraction-json">{candidate_json_html}</pre>
        </div>
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              Stage 1 Configuration &amp; Diagnostics
            </div>
            <span class="stat-badge badge-green">Stage 1 Complete</span>
          </div>
          <table class="data-table">
            <tbody>
              <tr><td style="font-weight: 600; width: 40%;">Primary Extractor Engine</td><td><code>{extractor_model}</code></td></tr>
              <tr><td style="font-weight: 600;">Thinking Tokens Budget</td><td><code>0 tokens</code> (Explicitly disabled for high-throughput)</td></tr>
              <tr><td style="font-weight: 600;">Sampling Temperature</td><td><code>0.10</code> (Low-variance deterministic schema extraction)</td></tr>
              <tr><td style="font-weight: 600;">Response Format</td><td><code>application/json</code> (Strict schema enforcement)</td></tr>
              <tr><td style="font-weight: 600;">Document Modality</td><td><code>Multimodal Inline PDF</code></td></tr>
              <tr><td style="font-weight: 600;">Total Extracted Attributes</td><td><code>{total_fields} fields</code></td></tr>
            </tbody>
          </table>
          <div style="margin-top: 1rem; padding: 0.875rem; background-color: var(--bg-surface-subtle); border-radius: 0.375rem; border: 1px solid var(--border-color); font-size: 0.8125rem;">
            <strong>Stage 1 Role:</strong> High-throughput initial extraction producing the candidate object. In Stage 2, five independent LLM judges with active reasoning tokens cross-examine this candidate payload against the original visual PDF pages and verification rubric.
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: Judge Panel Matrix -->
    <div id="tab-judges" class="tab-panel">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title">
            <svg class="icon" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
            Parallel LLM-as-a-Judge Panel ({requested_judges} Concurrent Thinkers)
          </div>
          <div style="display: flex; gap: 0.75rem; align-items: center;">
            <input type="text" id="judge-filter-input" class="search-input" placeholder="Filter fields..." onkeyup="filterJudgeFields()" />
            <span class="stat-badge badge-purple">Model: {judge_model}</span>
            <span class="stat-badge badge-purple">Thinking: {thinking_budget} tokens</span>
          </div>
        </div>
        <div class="judges-grid" id="judges-grid-container">
          {judge_matrix_html}
        </div>
      </div>
    </div>

    <!-- TAB 4: Consensus & Routing -->
    <div id="tab-consensus" class="tab-panel">
      <div class="split-layout">
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>
              Field-Level Consensus &amp; Routing Table
            </div>
            <input type="text" id="consensus-filter-input" class="search-input" placeholder="Search fields..." onkeyup="filterConsensusTable()" />
          </div>
          <div style="overflow-x: auto;">
            <table class="data-table" id="consensus-table">
              <thead>
                <tr>
                  <th>Field Name</th>
                  <th>Agreement (k / {requested_judges})</th>
                  <th>Status</th>
                  <th>Confidence</th>
                  <th>Routing Action</th>
                </tr>
              </thead>
              <tbody id="consensus-table-body">
                {consensus_table_rows_html}
              </tbody>
            </table>
          </div>
        </div>

        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
              Validated Clean Output Payload
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <span class="stat-badge badge-green" id="accepted-payload-count">{accepted_fields} fields auto-accepted</span>
              <button class="btn" id="copy-payload-btn" onclick="copyPayloadJson()">Copy Clean JSON</button>
            </div>
          </div>
          <pre class="code-block" id="accepted-payload-json">{accepted_payload_json_html}</pre>
          
          <div style="margin-top: 1rem;">
            <div style="font-weight: 700; font-size: 0.875rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.35rem;">
              <svg class="icon" style="color: var(--accent-amber);" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
              Exceptions &amp; Dissent Analysis ({len(report.get("exceptions", []))})
            </div>
            <div id="exceptions-container">
              {exceptions_list_html}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: Provenance & Audit Trail -->
    <div id="tab-audit" class="tab-panel">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title">
            <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
            Complete Provenance Audit Trail &amp; Execution Metadata
          </div>
          <div style="display: flex; gap: 0.5rem;">
            <button class="btn" id="copy-audit-btn" onclick="copyAuditJson()">Copy Full Audit JSON</button>
            <button class="btn btn-primary" onclick="exportReportJson()">Download Report</button>
          </div>
        </div>
        <div style="margin-bottom: 1rem;">
          <h4 style="font-size: 0.8125rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;">Field-by-Field Audit Record</h4>
          <div style="overflow-x: auto;">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Field Name</th>
                  <th>Candidate Value</th>
                  <th>Status</th>
                  <th>Passes (k / 5)</th>
                  <th>Failure Modes</th>
                  <th>Judge Dissent / Justifications</th>
                </tr>
              </thead>
              <tbody>
                {audit_trail_table_html}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h4 style="font-size: 0.8125rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;">Full Serialized Report JSON</h4>
          <pre class="code-block" id="full-audit-json" style="max-height: 480px;">{full_audit_json_html}</pre>
        </div>
      </div>
    </div>
  </main>

  <script>
    const RUN_REPORT = {report_json_escaped};
    const RUN_RUBRIC = {rubric_json_escaped};
    const RUN_PDF_B64 = {pdf_b64_escaped};

    const themeToggleBtn = document.getElementById("theme-toggle");
    const sunIcon = document.getElementById("theme-icon-sun");
    const moonIcon = document.getElementById("theme-icon-moon");

    function setTheme(theme) {{
      document.body.setAttribute("data-theme", theme);
      if (theme === "dark") {{
        if (sunIcon) sunIcon.style.display = "inline-block";
        if (moonIcon) moonIcon.style.display = "none";
      }} else {{
        if (sunIcon) sunIcon.style.display = "none";
        if (moonIcon) moonIcon.style.display = "inline-block";
      }}
      try {{ localStorage.setItem("pdf_evaluator_theme", theme); }} catch(e) {{}}
    }}

    if (themeToggleBtn) {{
      themeToggleBtn.addEventListener("click", () => {{
        const current = document.body.getAttribute("data-theme") || "light";
        setTheme(current === "light" ? "dark" : "light");
      }});
    }}

    try {{
      const savedTheme = localStorage.getItem("pdf_evaluator_theme") || "light";
      setTheme(savedTheme);
    }} catch(e) {{
      setTheme("light");
    }}

    document.querySelectorAll(".nav-tab").forEach(tab => {{
      tab.addEventListener("click", () => {{
        document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        const targetId = tab.getAttribute("data-tab");
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) targetPanel.classList.add("active");
      }});
    }});

    function copyCandidateJson() {{
      const text = JSON.stringify(RUN_REPORT.candidate_extraction || {{}}, null, 2);
      navigator.clipboard.writeText(text).then(() => {{
        const btn = document.getElementById("copy-candidate-btn");
        if (btn) {{
          btn.textContent = "Copied!";
          setTimeout(() => {{ btn.textContent = "Copy JSON"; }}, 2000);
        }}
      }});
    }}

    function copyPayloadJson() {{
      const text = JSON.stringify(RUN_REPORT.accepted_payload || {{}}, null, 2);
      navigator.clipboard.writeText(text).then(() => {{
        const btn = document.getElementById("copy-payload-btn");
        if (btn) {{
          btn.textContent = "Copied!";
          setTimeout(() => {{ btn.textContent = "Copy Clean JSON"; }}, 2000);
        }}
      }});
    }}

    function copyAuditJson() {{
      const text = JSON.stringify(RUN_REPORT, null, 2);
      navigator.clipboard.writeText(text).then(() => {{
        const btn = document.getElementById("copy-audit-btn");
        if (btn) {{
          btn.textContent = "Copied!";
          setTimeout(() => {{ btn.textContent = "Copy Full Audit JSON"; }}, 2000);
        }}
      }});
    }}

    function exportReportJson() {{
      const blob = new Blob([JSON.stringify(RUN_REPORT, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = (RUN_REPORT.document_type || "consensus") + "_evaluation_report.json";
      a.click();
      URL.revokeObjectURL(url);
    }}

    function filterConsensusTable() {{
      const query = (document.getElementById("consensus-filter-input").value || "").toLowerCase();
      const rows = document.querySelectorAll("#consensus-table-body tr");
      rows.forEach(row => {{
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? "" : "none";
      }});
    }}

    function filterJudgeFields() {{
      const query = (document.getElementById("judge-filter-input").value || "").toLowerCase();
      const cards = document.querySelectorAll(".field-eval-card");
      cards.forEach(card => {{
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(query) ? "" : "none";
      }});
    }}

    const exportBtn = document.getElementById("export-json-btn");
    if (exportBtn) {{
      exportBtn.addEventListener("click", exportReportJson);
    }}

    let pdfDoc = null;
    let pageNum = 1;
    let pdfScale = 1.2;
    const canvas = document.getElementById("pdf-canvas");
    const canvasWrapper = document.getElementById("pdf-canvas-wrapper");
    const embedFrame = document.getElementById("pdf-embed-frame");

    function renderPdfPage(num) {{
      if (!pdfDoc || !canvas) return;
      pdfDoc.getPage(num).then(page => {{
        const ctx = canvas.getContext("2d");
        const viewport = page.getViewport({{ scale: pdfScale }});
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {{
          canvasContext: ctx,
          viewport: viewport
        }};
        page.render(renderContext);
        const pageDisplay = document.getElementById("page-num-display");
        if (pageDisplay) pageDisplay.textContent = `Page ${{num}} of ${{pdfDoc.numPages}}`;
      }}).catch(err => {{
        console.warn("PDF page render error:", err);
        showEmbedFallback();
      }});
    }}

    function showEmbedFallback() {{
      if (canvasWrapper) canvasWrapper.style.display = "none";
      if (embedFrame) embedFrame.style.display = "block";
    }}

    const prevBtn = document.getElementById("prev-page");
    if (prevBtn) {{
      prevBtn.addEventListener("click", () => {{
        if (pageNum <= 1) return;
        pageNum--;
        renderPdfPage(pageNum);
      }});
    }}

    const nextBtn = document.getElementById("next-page");
    if (nextBtn) {{
      nextBtn.addEventListener("click", () => {{
        if (!pdfDoc || pageNum >= pdfDoc.numPages) return;
        pageNum++;
        renderPdfPage(pageNum);
      }});
    }}

    const zoomInBtn = document.getElementById("zoom-in");
    if (zoomInBtn) {{
      zoomInBtn.addEventListener("click", () => {{
        pdfScale = Math.min(2.5, pdfScale + 0.2);
        renderPdfPage(pageNum);
      }});
    }}

    const zoomOutBtn = document.getElementById("zoom-out");
    if (zoomOutBtn) {{
      zoomOutBtn.addEventListener("click", () => {{
        pdfScale = Math.max(0.6, pdfScale - 0.2);
        renderPdfPage(pageNum);
      }});
    }}

    if (RUN_PDF_B64) {{
      try {{
        if (typeof pdfjsLib !== "undefined") {{
          const binaryStr = atob(RUN_PDF_B64);
          const len = binaryStr.length;
          const bytes = new Uint8Array(len);
          for (let i = 0; i < len; i++) {{
            bytes[i] = binaryStr.charCodeAt(i);
          }}
          pdfjsLib.getDocument({{ data: bytes }}).promise.then(pdf => {{
            pdfDoc = pdf;
            renderPdfPage(1);
          }}).catch(err => {{
            console.warn("PDF.js init failed, falling back to embed:", err);
            showEmbedFallback();
          }});
        }} else {{
          showEmbedFallback();
        }}
      }} catch(err) {{
        console.warn("Base64 decode or PDF load failed:", err);
        showEmbedFallback();
      }}
    }}
  </script>
</body>
</html>"""


def _render_rubric_criteria_html(rubric: Dict[str, Any]) -> str:
    criteria = rubric.get("extraction_criteria", [])
    if not criteria:
        return '<div style="color: var(--text-muted); font-size: 0.8125rem;">No rubric criteria specified.</div>'

    cards = []
    for c in criteria:
        field_name = html.escape(str(c.get("field_name", "")))
        syn = c.get("syntactic_rules", {})
        sem = c.get("semantic_grounding_rules", {})

        req = "Required" if syn.get("required") else "Optional"
        req_class = "badge-blue" if syn.get("required") else "badge-purple"
        req_badge = f'<span class="stat-badge {req_class}">{req}</span>'
        type_badge = f'<span class="stat-badge badge-blue">Type: {html.escape(str(syn.get("type", "string")))}</span>'

        regex = syn.get("format_regex")
        regex_row = f'<div class="criteria-row"><span class="criteria-label">Format Regex:</span><code>/{html.escape(str(regex))}/</code></div>' if regex else ""

        subfields = syn.get("expected_subfields")
        subfields_row = f'<div class="criteria-row"><span class="criteria-label">Subfields:</span><code>{html.escape(str(subfields))}</code></div>' if subfields else ""

        failure_modes = sem.get("failure_modes", [])
        failure_tags = "".join(f'<span class="failure-tag">{html.escape(str(fm))}</span>' for fm in failure_modes)
        failure_row = f'<div class="criteria-row"><span class="criteria-label">Failure Modes:</span><div>{failure_tags}</div></div>' if failure_tags else ""

        source_section = html.escape(str(sem.get("source_section", "")))
        verification_criteria = html.escape(str(sem.get("verification_criteria", "")))

        card_html = f"""
        <div class="criteria-card">
          <div class="criteria-header">
            <span class="criteria-name">{field_name}</span>
            <div style="display: flex; gap: 0.35rem;">
              {type_badge}
              {req_badge}
            </div>
          </div>
          <div class="criteria-row">
            <span class="criteria-label">Source Section:</span>
            <span style="font-weight: 500;">{source_section}</span>
          </div>
          <div class="criteria-row">
            <span class="criteria-label">Verification:</span>
            <span>{verification_criteria}</span>
          </div>
          {regex_row}
          {subfields_row}
          {failure_row}
        </div>
        """
        cards.append(card_html)

    return "\n".join(cards)


def _render_judge_matrix_html(
    report: Dict[str, Any],
    requested_judges: int,
    thinking_budget: int,
    judge_model: str,
) -> str:
    consensus = report.get("consensus_breakdown", {})
    exceptions = report.get("exceptions", [])
    judge_reports = report.get("judge_reports", [])

    columns = []
    num_cols = max(requested_judges, len(judge_reports), 5)

    for j in range(1, num_cols + 1):
        judge_id = f"judge_{j}"
        jr = next((r for r in judge_reports if r.get("judge_id") == judge_id or r.get("judge_id") == f"judge_{j}"), None)

        exec_time = f"{jr.get('execution_time_seconds', 0):.2f}s" if jr else "Async"

        field_cards = []
        for field_name, c in consensus.items():
            fn_escaped = html.escape(field_name)

            fe = None
            if jr and jr.get("evaluations"):
                fe = next((ev for ev in jr["evaluations"] if ev.get("field_name") == field_name), None)

            if fe:
                verdict = str(fe.get("verdict", "PASS")).upper()
                syntactic = str(fe.get("syntactic_check", "PASS")).upper()
                grounding = str(fe.get("grounding_check", "PASS")).upper()
                failure_mode = fe.get("failure_mode", "NONE")
                justification = html.escape(str(fe.get("justification", "")))
                correction = fe.get("proposed_correction")
            else:
                agreement_count = c.get("agreement_count", requested_judges)
                is_pass = (j <= agreement_count)
                verdict = "PASS" if is_pass else "FAIL"
                syntactic = "PASS" if is_pass else "FAIL"
                grounding = "PASS" if is_pass else "FAIL"
                failure_mode = "NONE" if is_pass else "FORMAT_MISMATCH"
                justification = "Syntactic and grounding verified against source PDF." if is_pass else "Dissent: failed criteria check."
                correction = None

            verdict_badge = f'<span class="stat-badge {"badge-green" if verdict == "PASS" else "badge-red"}">{verdict}</span>'
            syn_badge = f'<span class="stat-badge {"badge-green" if syntactic == "PASS" else "badge-red"}" style="font-size: 0.65rem; padding: 0.1rem 0.35rem;">Syn: {syntactic}</span>'
            gnd_badge = f'<span class="stat-badge {"badge-green" if grounding == "PASS" else "badge-red"}" style="font-size: 0.65rem; padding: 0.1rem 0.35rem;">Gnd: {grounding}</span>'

            fm_tag = f'<span class="failure-tag" style="margin: 0.2rem 0;">{html.escape(str(failure_mode))}</span>' if failure_mode and failure_mode != "NONE" else ""
            correction_html = f'<div style="font-size: 0.7rem; color: var(--accent-blue); margin-top: 0.2rem;">Correction: <code>{html.escape(str(correction))}</code></div>' if correction else ""

            card = f"""
            <div class="field-eval-card">
              <div class="field-eval-top">
                <span class="field-eval-name">{fn_escaped}</span>
                {verdict_badge}
              </div>
              <div class="field-eval-checks">
                {syn_badge}
                {gnd_badge}
              </div>
              {fm_tag}
              <div class="field-eval-just">{justification}</div>
              {correction_html}
            </div>
            """
            field_cards.append(card)

        cards_html = "\n".join(field_cards)
        col_html = f"""
        <div class="judge-column">
          <div class="judge-col-header">
            <div>
              <span class="judge-name">
                <svg class="icon" style="color: var(--accent-purple);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>
                Judge #{j}
              </span>
              <div style="font-size: 0.6875rem; color: var(--text-muted); margin-top: 0.1rem;">Time: {exec_time}</div>
            </div>
            <span class="stat-badge badge-purple" style="font-size: 0.6875rem;">Thinking: {thinking_budget}</span>
          </div>
          {cards_html}
        </div>
        """
        columns.append(col_html)

    return "\n".join(columns)


def _render_consensus_table_rows_html(report: Dict[str, Any]) -> str:
    consensus = report.get("consensus_breakdown", {})
    if not consensus:
        return '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No consensus records found.</td></tr>'

    rows = []
    for field_name, c in consensus.items():
        fn_escaped = html.escape(field_name)
        ratio = html.escape(str(c.get("agreement_ratio", "")))
        status = str(c.get("status", "UNANIMOUS_PASS"))
        confidence = html.escape(str(c.get("confidence_band", "")))
        routing = html.escape(str(c.get("routing_action", "")))

        status_class = "badge-green"
        if status == "MAJORITY_PASS":
            status_class = "badge-blue"
        elif status == "CONTESTED":
            status_class = "badge-amber"
        elif status == "REJECTED":
            status_class = "badge-red"

        row = f"""
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-blue);">{fn_escaped}</td>
          <td><span style="font-weight: 800;">{ratio}</span></td>
          <td><span class="stat-badge {status_class}">{status}</span></td>
          <td style="font-size: 0.8125rem; color: var(--text-secondary);">{confidence}</td>
          <td><span class="stat-badge {status_class}">{routing}</span></td>
        </tr>
        """
        rows.append(row)

    return "\n".join(rows)


def _render_exceptions_list_html(report: Dict[str, Any]) -> str:
    exceptions = report.get("exceptions", [])
    if not exceptions:
        return '<div style="padding: 0.75rem; background-color: var(--accent-green-subtle); color: var(--accent-green); border-radius: 0.375rem; font-size: 0.8125rem; font-weight: 600;">No exceptions detected. All fields achieved unanimous or majority consensus!</div>'

    cards = []
    for exc in exceptions:
        field_name = html.escape(str(exc.get("field_name", "")))
        status = str(exc.get("status", "CONTESTED"))
        ratio = html.escape(str(exc.get("agreement_ratio", "")))
        routing = html.escape(str(exc.get("recommended_routing", "")))
        card_class = "rejected" if status == "REJECTED" else ""

        fms = exc.get("failure_modes", [])
        fm_tags = "".join(f'<span class="failure-tag">{html.escape(str(fm))}</span>' for fm in fms)

        dissent = exc.get("dissent_reasons", []) or exc.get("justifications", [])
        dissent_items = "".join(f'<li style="margin-top: 0.2rem;">{html.escape(str(d))}</li>' for d in dissent)

        candidate_val = html.escape(str(exc.get("candidate_value", "")))

        card = f"""
        <div class="exception-card {card_class}">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
            <strong style="font-family: var(--font-mono); font-size: 0.875rem;">{field_name}</strong>
            <span class="stat-badge {"badge-red" if status == "REJECTED" else "badge-amber"}">{status} ({ratio})</span>
          </div>
          <div><strong>Candidate Value:</strong> <code>{candidate_val}</code></div>
          <div style="margin-top: 0.25rem;"><strong>Failure Modes:</strong> {fm_tags or "None specified"}</div>
          <div style="margin-top: 0.35rem;"><strong>Routing:</strong> <span class="stat-badge {"badge-red" if status == "REJECTED" else "badge-amber"}">{routing}</span></div>
          {f'<div style="margin-top: 0.35rem;"><strong>Judge Dissent Reasons:</strong><ul style="padding-left: 1.2rem; margin-top: 0.2rem;">{dissent_items}</ul></div>' if dissent_items else ''}
        </div>
        """
        cards.append(card)

    return "\n".join(cards)


def _render_audit_trail_table_html(report: Dict[str, Any]) -> str:
    audit = report.get("audit_trail", [])
    if not audit:
        return '<tr><td colspan="6" style="color: var(--text-muted); text-align: center;">No audit trail items found.</td></tr>'

    rows = []
    for item in audit:
        fn = html.escape(str(item.get("field_name", "")))
        val = html.escape(json.dumps(item.get("candidate_value", "")))
        status = str(item.get("status", "UNANIMOUS_PASS"))
        passes = f"{item.get('agreement_count', 5)} / {item.get('total_judges', 5)}"

        status_class = "badge-green"
        if status == "MAJORITY_PASS":
            status_class = "badge-blue"
        elif status == "CONTESTED":
            status_class = "badge-amber"
        elif status == "REJECTED":
            status_class = "badge-red"

        fms = item.get("failure_modes", [])
        fm_tags = " ".join(f'<span class="failure-tag">{html.escape(str(fm))}</span>' for fm in fms) or "-"

        dissent = item.get("dissent_reasons", [])
        dissent_text = "<br/>".join(html.escape(str(d)) for d in dissent) if dissent else '<span style="color: var(--accent-green);">All judges agreed</span>'

        row = f"""
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 700;">{fn}</td>
          <td><code>{val}</code></td>
          <td><span class="stat-badge {status_class}">{status}</span></td>
          <td style="font-weight: 700;">{passes}</td>
          <td>{fm_tags}</td>
          <td style="font-size: 0.8125rem;">{dissent_text}</td>
        </tr>
        """
        rows.append(row)

    return "\n".join(rows)
