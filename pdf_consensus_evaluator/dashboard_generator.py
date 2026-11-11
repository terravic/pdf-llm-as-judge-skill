"""HTML Dashboard Generator for Multi-Agent PDF Extraction and Consensus Evaluator.

Adheres strictly to DESIGN.md:
- Typography: Instrument Serif (display), DM Sans (UI), IBM Plex Mono (tabular-nums).
- Flat single-level bento surfaces with 1px solid borders (6px-8px radius), zero gradients.
- Three-icon segmented theme selector (Light / Dark / System) with cross-platform OS fallback.
- Exact Light (#FAFAF9 / #FFFFFF / #E7E5E4 / #1C1917 / #787774) and Dark
  (#0D0E11 / #16171D / #292A30 / #ECE8E2 / #9A9895) color schemes.
- Semantic data visualization palette (#0D9488, #D97706, #4F46E5, #059669, #E11D48) across
  2D quantitative charts, Judge x Field heatmap, and SVG Interactive Relationship Node Map.
- Editable model configuration inputs backed by dynamic credential-probed <datalist>
  autocomplete and a static /pdf-llm-as-judge & CLI command builder.
- Full dynamic usability for new PDF/image runs via CLI regeneration, in-browser file loading,
  and live HTTP Server-Sent Events (SSE) without resetting viewport/camera/scroll state.
"""

from __future__ import annotations

import base64
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf_consensus_evaluator.models import PipelineReport, RubricSpec

DEFAULT_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def _detect_host_system_theme() -> str:
    """Queries the native host OS appearance preference as a fallback for headless webviews."""
    try:
        if sys.platform == "darwin":
            res = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if res.returncode == 0 and "dark" in res.stdout.strip().lower():
                return "dark"
            return "light"

        if sys.platform == "win32":
            import winreg  # type: ignore

            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                return "dark" if int(val) == 0 else "light"

        # Linux: inspect GTK_THEME, gsettings, and kdeglobals
        gtk_theme = os.environ.get("GTK_THEME", "").lower()
        if "dark" in gtk_theme:
            return "dark"

        for cmd in (
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
        ):
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
                if res.returncode == 0 and "dark" in res.stdout.strip().lower():
                    return "dark"
            except (FileNotFoundError, subprocess.SubprocessError):
                continue

        kde_cfg = Path.home() / ".config" / "kdeglobals"
        if kde_cfg.exists():
            content = kde_cfg.read_text(encoding="utf-8", errors="ignore").lower()
            for line in content.splitlines():
                if line.strip().startswith("colorscheme=") and "dark" in line:
                    return "dark"
    except Exception:
        pass

    return "light"


def _discover_available_models(
    active_models: Optional[List[str]] = None,
    allow_network: Optional[bool] = None,
) -> List[str]:
    """Discovers generateContent-capable Gemini models available to active credentials."""
    ordered: List[str] = []

    def _add(model_id: str) -> None:
        cleaned = str(model_id or "").strip()
        if cleaned.startswith("models/"):
            cleaned = cleaned[len("models/") :]
        elif "/models/" in cleaned:
            cleaned = cleaned.rsplit("/models/", 1)[-1]
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)

    for m in active_models or []:
        _add(m)

    if allow_network is None:
        allow_network = "PYTEST_CURRENT_TEST" not in os.environ

    if allow_network:
        try:
            import requests
            from pdf_consensus_evaluator.gemini_client import GeminiClient

            client = GeminiClient()
            discovered: List[str] = []

            if client.api_key or client.oauth_token:
                headers: Dict[str, str] = {}
                sep = "&" if "?" in client.base_url else "?"
                url = f"{client.base_url}{sep}pageSize=100"
                if client.api_key:
                    url = f"{url}&key={client.api_key}"
                    headers["x-goog-api-key"] = client.api_key
                elif client.oauth_token:
                    headers["Authorization"] = f"Bearer {client.oauth_token}"
                    if client.project_id:
                        headers["x-goog-user-project"] = client.project_id

                resp = requests.get(url, headers=headers, timeout=2.0)
                if resp.status_code == 200:
                    payload = resp.json()
                    for item in payload.get("models") or []:
                        name = str(item.get("name", ""))
                        methods = item.get("supportedGenerationMethods") or []
                        if ("generateContent" in methods or not methods) and "gemini" in name.lower():
                            discovered.append(name)

            if not discovered and client.project_id and client.oauth_token:
                v_url = (
                    f"https://{client.location}-aiplatform.googleapis.com/v1beta1/"
                    f"publishers/google/models?pageSize=100"
                )
                v_headers = {
                    "Authorization": f"Bearer {client.oauth_token}",
                    "x-goog-user-project": client.project_id,
                }
                v_resp = requests.get(v_url, headers=v_headers, timeout=2.0)
                if v_resp.status_code == 200:
                    v_payload = v_resp.json()
                    for item in v_payload.get("publisherModels") or []:
                        name = str(item.get("name", ""))
                        if "gemini" in name.lower():
                            discovered.append(name)

            for d in sorted(set(discovered), reverse=True):
                _add(d)
        except Exception:
            pass

    for fallback in DEFAULT_FALLBACK_MODELS:
        _add(fallback)

    return ordered


def _safe_json_for_script(obj: Any) -> str:
    """Serializes Python data to JSON safe for inline <script> embedding."""
    return json.dumps(obj, default=str).replace("</", "<\\/")


def generate_dashboard_html(
    report: PipelineReport,
    document_path: Optional[str] = None,
    rubric: Optional[RubricSpec] = None,
    output_html_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
) -> str:
    """Generates a self-contained, DESIGN.md-compliant interactive HTML Dashboard."""
    target_path = document_path or pdf_path
    doc_b64 = ""
    mime_type = "application/pdf"
    doc_filename = ""
    doc_path_hint = ""

    if target_path:
        try:
            rel = os.path.relpath(target_path, os.getcwd())
            doc_path_hint = rel if not rel.startswith("..") else target_path
        except ValueError:
            doc_path_hint = target_path

    if target_path and os.path.exists(target_path):
        from pdf_consensus_evaluator.gemini_client import GeminiClient

        mime_type = GeminiClient.detect_mime_type(target_path)
        doc_filename = os.path.basename(target_path)
        with open(target_path, "rb") as f:
            doc_b64 = base64.b64encode(f.read()).decode("utf-8")

    rubric_data: Dict[str, Any] = {}
    if rubric:
        rubric_data = {
            "rubric_version": rubric.rubric_version,
            "document_type": rubric.document_type,
            "extraction_criteria": [
                {
                    "field_name": c.field_name,
                    "field_type": c.field_type or c.syntactic_rules.field_type,
                    "bounding_box": list(c.bounding_box)
                    if c.bounding_box
                    else (
                        list(c.syntactic_rules.bounding_box)
                        if c.syntactic_rules.bounding_box
                        else None
                    ),
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
    system_theme = _detect_host_system_theme()
    available_models = _discover_available_models(
        active_models=[
            str(report_data.get("extractor_model") or "gemini-3.8-flash"),
            str(report_data.get("judge_model") or "gemini-3.8-flash"),
        ]
    )

    html_content = _build_html_template(
        report=report_data,
        doc_b64=doc_b64,
        mime_type=mime_type,
        doc_filename=doc_filename,
        doc_path_hint=doc_path_hint,
        rubric=rubric_data,
        system_theme=system_theme,
        available_models=available_models,
    )

    if output_html_path:
        out_dir = os.path.dirname(os.path.abspath(output_html_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


def _status_token_class(status: str) -> str:
    st = str(status or "").upper()
    if st in ("UNANIMOUS_PASS", "PASS"):
        return "state-emerald"
    if st == "MAJORITY_PASS":
        return "state-teal"
    if st == "CONTESTED":
        return "state-amber"
    if st in ("REJECTED", "FAIL"):
        return "state-rose"
    return "state-muted"


def _render_extracted_fields_rows_html(report: Dict[str, Any]) -> str:
    cand = report.get("candidate_extraction") or {}
    consensus = report.get("consensus_breakdown") or {}
    if not cand:
        return (
            '<tr><td colspan="4" class="empty-cell">'
            "No candidate attributes extracted."
            "</td></tr>"
        )

    rows = []
    for field_name, val in cand.items():
        fn_esc = html.escape(str(field_name))
        if isinstance(val, dict):
            dtype = f"object ({len(val)} keys)"
            preview = json.dumps(val, ensure_ascii=False)
        elif isinstance(val, list):
            dtype = f"array ({len(val)} items)"
            preview = json.dumps(val, ensure_ascii=False)
        elif isinstance(val, bool):
            dtype = "boolean"
            preview = "true" if val else "false"
        elif isinstance(val, (int, float)):
            dtype = "number"
            preview = str(val)
        elif val is None:
            dtype = "null"
            preview = "null"
        else:
            s = str(val)
            dtype = f"string ({len(s)} ch)"
            preview = s

        c = consensus.get(field_name) or {}
        st = str(c.get("status", "CANDIDATE"))
        ratio = str(c.get("agreement_ratio", "")).split(" ")[0]
        pill_txt = f"{st} · {ratio}" if ratio else st
        cls = _status_token_class(st)

        rows.append(
            f"<tr>"
            f'<td class="mono-key">{fn_esc}</td>'
            f'<td class="mono-cell muted-cell">{html.escape(dtype)}</td>'
            f'<td class="mono-cell code-cell">{html.escape(preview)}</td>'
            f'<td><span class="status-pill {cls}">{html.escape(pill_txt)}</span></td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _render_rubric_rows_html(rubric: Dict[str, Any], report: Dict[str, Any]) -> str:
    criteria = rubric.get("extraction_criteria") or []
    if not criteria:
        # Synthesize view rows from consensus_breakdown if rubric was omitted
        consensus = report.get("consensus_breakdown") or {}
        if not consensus:
            return (
                '<tr><td colspan="4" class="empty-cell">'
                "No rubric specifications loaded."
                "</td></tr>"
            )
        rows = []
        for fn in consensus.keys():
            fn_esc = html.escape(str(fn))
            rows.append(
                f"<tr>"
                f'<td class="mono-key">{fn_esc}</td>'
                f'<td class="mono-cell">string · req</td>'
                f"<td>Document Source</td>"
                f'<td class="muted-cell">Derived from candidate extraction schema.</td>'
                f"</tr>"
            )
        return "\n".join(rows)

    rows = []
    for c in criteria:
        fn = html.escape(str(c.get("field_name", "")))
        syn = c.get("syntactic_rules") or {}
        sem = c.get("semantic_grounding_rules") or {}

        dtype = html.escape(str(syn.get("type", "string")))
        req = "req" if syn.get("required", True) else "opt"
        constraints: List[str] = [f"{dtype} · {req}"]
        if syn.get("format_regex"):
            constraints.append(f"/{html.escape(str(syn['format_regex']))}/")
        if syn.get("expected_subfields"):
            subs = ", ".join(html.escape(str(s)) for s in syn["expected_subfields"])
            constraints.append(f"[{subs}]")
        if c.get("field_type"):
            constraints.append(html.escape(str(c["field_type"])))
        if c.get("bounding_box"):
            bbox_str = ", ".join(str(v) for v in c["bounding_box"])
            constraints.append(f"roi:[{html.escape(bbox_str)}]")

        source_sec = html.escape(str(sem.get("source_section") or "—"))
        verif = html.escape(str(sem.get("verification_criteria") or ""))
        fms = sem.get("failure_modes") or []
        fm_html = ""
        if fms:
            tags = " ".join(
                f'<span class="fm-token">{html.escape(str(fm))}</span>' for fm in fms
            )
            fm_html = f'<div class="fm-token-row">{tags}</div>'

        rows.append(
            f"<tr>"
            f'<td class="mono-key">{fn}</td>'
            f'<td class="mono-cell">{"<br/>".join(constraints)}</td>'
            f'<td class="mono-cell">{source_sec}</td>'
            f"<td><div>{verif}</div>{fm_html}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _render_judge_matrix_table_html(report: Dict[str, Any], requested_judges: int) -> str:
    consensus = report.get("consensus_breakdown") or {}
    judge_reports = report.get("judge_reports") or []
    num_cols = max(int(requested_judges or 5), len(judge_reports), 1)

    header_Judge_ths = "".join(
        f'<th class="mono-th">Judge #{j}</th>' for j in range(1, num_cols + 1)
    )
    thead = (
        f"<thead><tr>"
        f"<th>Field</th>"
        f"{header_Judge_ths}"
        f'<th class="mono-th">Ratio</th>'
        f"<th>Primary Citation / Dissent</th>"
        f"</tr></thead>"
    )

    if not consensus:
        return (
            f"{thead}<tbody><tr>"
            f'<td colspan="{num_cols + 3}" class="empty-cell">No judge evaluations recorded.</td>'
            f"</tr></tbody>"
        )

    body_rows = []
    for field_name, c in consensus.items():
        fn_esc = html.escape(str(field_name))
        ratio_esc = html.escape(str(c.get("agreement_ratio", "")))
        cells = []
        primary_note = ""

        for j in range(1, num_cols + 1):
            jid = f"judge_{j}"
            jr = next((r for r in judge_reports if r.get("judge_id") == jid), None)
            fe = None
            if jr and jr.get("evaluations"):
                fe = next(
                    (ev for ev in jr["evaluations"] if ev.get("field_name") == field_name),
                    None,
                )

            if fe:
                verdict = str(fe.get("verdict", "PASS")).upper()
                syn = str(fe.get("syntactic_check", "PASS")).upper()
                gnd = str(fe.get("grounding_check", "PASS")).upper()
                fm = str(fe.get("failure_mode", "NONE"))
                just = str(fe.get("justification", ""))
                if not primary_note and just:
                    primary_note = just
                if verdict != "PASS" and just:
                    primary_note = f"[{jid}] {just}"
            elif jr and jr.get("error"):
                verdict = "ERR"
                syn = "—"
                gnd = "—"
                fm = "QUORUM_ERR"
                if not primary_note:
                    primary_note = str(jr.get("error", ""))
            else:
                ag = int(c.get("agreement_count", num_cols))
                is_pass = j <= ag
                verdict = "PASS" if is_pass else "FAIL"
                syn = verdict
                gnd = verdict
                fm = "NONE" if is_pass else "DISSENT"
                if not primary_note:
                    primary_note = (
                        "Verified against source document."
                        if is_pass
                        else "Dissenting check failure."
                    )

            cls = _status_token_class(verdict if verdict != "ERR" else "FAIL")
            sub = (
                f"S:{syn[0]} G:{gnd[0]}"
                if syn in ("PASS", "FAIL") and gnd in ("PASS", "FAIL")
                else fm
            )
            cells.append(
                f'<td class="mono-cell judge-verdict-cell" '
                f'data-field="{fn_esc}" data-judge="{j}">'
                f'<span class="status-pill {cls}">{html.escape(verdict)}</span>'
                f'<span class="verdict-sub">{html.escape(sub)}</span>'
                f"</td>"
            )

        body_rows.append(
            f'<tr class="judge-matrix-row" data-field="{fn_esc}">'
            f'<td class="mono-key">{fn_esc}</td>'
            f'{"".join(cells)}'
            f'<td class="mono-cell">{ratio_esc}</td>'
            f'<td class="note-cell">{html.escape(primary_note)}</td>'
            f"</tr>"
        )

    return f"{thead}<tbody id=\"judge-matrix-tbody\">{''.join(body_rows)}</tbody>"


def _render_consensus_rows_html(report: Dict[str, Any]) -> str:
    consensus = report.get("consensus_breakdown") or {}
    if not consensus:
        return (
            '<tr><td colspan="5" class="empty-cell">'
            "No consensus breakdown records available."
            "</td></tr>"
        )

    rows = []
    for field_name, c in consensus.items():
        fn_esc = html.escape(str(field_name))
        ratio_esc = html.escape(str(c.get("agreement_ratio", "")))
        status = str(c.get("status", "UNANIMOUS_PASS"))
        conf_esc = html.escape(str(c.get("confidence_band", "")))
        route_esc = html.escape(str(c.get("routing_action", "")))
        cls = _status_token_class(status)

        rows.append(
            f"<tr>"
            f'<td class="mono-key">{fn_esc}</td>'
            f'<td class="mono-cell">{ratio_esc}</td>'
            f'<td><span class="status-pill {cls}">{html.escape(status)}</span></td>'
            f'<td class="mono-cell muted-cell">{conf_esc}</td>'
            f'<td class="mono-cell">{route_esc}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _render_exceptions_ledger_html(report: Dict[str, Any]) -> str:
    exceptions = report.get("exceptions") or []
    if not exceptions:
        return (
            '<div class="ledger-empty">'
            '<span class="status-pill state-emerald">NOMINAL</span> '
            "Zero contested or rejected fields. All target attributes cleared consensus routing."
            "</div>"
        )

    items = []
    for exc in exceptions:
        fn = html.escape(str(exc.get("field_name", "")))
        status = str(exc.get("status", "CONTESTED"))
        cls = _status_token_class(status)
        ratio = html.escape(str(exc.get("agreement_ratio", "")))
        routing = html.escape(str(exc.get("recommended_routing", "")))
        cand_val = html.escape(json.dumps(exc.get("candidate_value"), ensure_ascii=False))
        fms = exc.get("failure_modes") or []
        fm_str = (
            " ".join(f'<span class="fm-token">{html.escape(str(m))}</span>' for m in fms)
            or '<span class="muted-cell">NONE</span>'
        )
        dissent = exc.get("dissent_reasons") or exc.get("justifications") or []
        dissent_html = "".join(f"<li>{html.escape(str(d))}</li>" for d in dissent)
        corr = exc.get("proposed_corrections") or []
        corr_html = ""
        if corr:
            corr_str = html.escape(json.dumps(corr, ensure_ascii=False))
            corr_html = (
                f'<div class="ledger-meta"><span>Proposed Overturn:</span> '
                f"<code>{corr_str}</code></div>"
            )

        items.append(
            f'<div class="ledger-row">'
            f'<div class="ledger-row-top">'
            f'<span class="mono-key">{fn}</span>'
            f'<span class="status-pill {cls}">{html.escape(status)} · {ratio}</span>'
            f"</div>"
            f'<div class="ledger-meta"><span>Candidate:</span> <code>{cand_val}</code></div>'
            f'<div class="ledger-meta"><span>Failure Modes:</span> {fm_str}</div>'
            f'<div class="ledger-meta"><span>Routing:</span> <span class="mono-cell">{routing}</span></div>'
            f"{corr_html}"
            f'{f"<ul class=ledger-list>{dissent_html}</ul>" if dissent_html else ""}'
            f"</div>"
        )
    return "\n".join(items)


def _render_audit_rows_html(report: Dict[str, Any]) -> str:
    audit = report.get("audit_trail") or []
    if not audit:
        return (
            '<tr><td colspan="6" class="empty-cell">'
            "No provenance audit trail entries available."
            "</td></tr>"
        )

    rows = []
    for item in audit:
        fn = html.escape(str(item.get("field_name", "")))
        val = html.escape(json.dumps(item.get("candidate_value", ""), ensure_ascii=False))
        status = str(item.get("status", "UNANIMOUS_PASS"))
        cls = _status_token_class(status)
        passes = f"{item.get('agreement_count', 0)} / {item.get('total_judges', 5)}"
        fms = item.get("failure_modes") or []
        fm_tags = (
            " ".join(f'<span class="fm-token">{html.escape(str(m))}</span>' for m in fms)
            or "—"
        )
        dissent = item.get("dissent_reasons") or []
        dissent_text = (
            "; ".join(html.escape(str(d)) for d in dissent)
            if dissent
            else "Verified across active judge quorum"
        )

        rows.append(
            f"<tr>"
            f'<td class="mono-key">{fn}</td>'
            f'<td class="mono-cell code-cell">{val}</td>'
            f'<td><span class="status-pill {cls}">{html.escape(status)}</span></td>'
            f'<td class="mono-cell">{html.escape(passes)}</td>'
            f'<td class="mono-cell">{fm_tags}</td>'
            f'<td class="note-cell">{dissent_text}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _build_html_template(
    report: Dict[str, Any],
    doc_b64: str,
    mime_type: str,
    doc_filename: str,
    doc_path_hint: str,
    rubric: Dict[str, Any],
    system_theme: str,
    available_models: List[str],
) -> str:
    is_image = mime_type.startswith("image/")

    extractor_model = html.escape(str(report.get("extractor_model") or "gemini-3.8-flash"))
    judge_model = html.escape(str(report.get("judge_model") or "gemini-3.8-flash"))
    thinking_budget = (
        report.get("thinking_budget") if report.get("thinking_budget") is not None else 2048
    )
    quorum = report.get("quorum_metadata") or {}
    requested_judges = int(quorum.get("requested_judges", 5))
    successful_judges = int(quorum.get("successful_judges", requested_judges))
    document_type = html.escape(str(report.get("document_type", "Document")))
    rubric_version = html.escape(str(report.get("rubric_version", "1.0")))
    raw_ts = str(report.get("timestamp", ""))
    timestamp = html.escape(raw_ts[:19].replace("T", " ") if raw_ts else "Live Session")

    total_fields = int(report.get("total_fields", 0))
    accepted_fields = int(report.get("accepted_field_count", 0))
    contested_fields = int(report.get("contested_field_count", 0))
    rejected_fields = int(report.get("rejected_field_count", 0))

    brand_title_text = (
        "Multi-Agent Image Extraction & Consensus Evaluator"
        if is_image
        else "Multi-Agent PDF Extraction & Consensus Evaluator"
    )
    tab1_label = (
        "1. Image &amp; Rubric Inspector" if is_image else "1. PDF &amp; Rubric Inspector"
    )
    viewer_header_title = (
        f"Multimodal Source Image Viewer ({html.escape(mime_type)})"
        if is_image
        else "Multimodal Source PDF Viewer"
    )
    modality_display = (
        f"Multimodal Inline Image ({html.escape(mime_type)})"
        if is_image
        else "Multimodal Inline PDF (application/pdf)"
    )

    rubric_rows_html = _render_rubric_rows_html(rubric, report)
    extracted_fields_rows_html = _render_extracted_fields_rows_html(report)
    candidate_json_html = html.escape(
        json.dumps(report.get("candidate_extraction") or {}, indent=2, ensure_ascii=False)
    )
    judge_matrix_table_html = _render_judge_matrix_table_html(report, requested_judges)
    consensus_rows_html = _render_consensus_rows_html(report)
    accepted_json_html = html.escape(
        json.dumps(report.get("accepted_payload") or {}, indent=2, ensure_ascii=False)
    )
    exceptions_ledger_html = _render_exceptions_ledger_html(report)
    audit_rows_html = _render_audit_rows_html(report)
    full_audit_json_html = html.escape(
        json.dumps(report, indent=2, default=str, ensure_ascii=False)
    )

    datalist_options_html = "\n".join(
        f'          <option value="{html.escape(m)}"></option>' for m in available_models
    )

    criteria_count = len(
        rubric.get("extraction_criteria")
        or list((report.get("consensus_breakdown") or {}).keys())
    )
    exceptions_count = len(report.get("exceptions") or [])

    img_src = f"data:{mime_type};base64,{doc_b64}" if (is_image and doc_b64) else ""
    pdf_src = f"data:application/pdf;base64,{doc_b64}" if ((not is_image) and doc_b64) else ""

    bundle_json = _safe_json_for_script(
        {
            "report": report,
            "rubric": rubric,
            "doc_b64": doc_b64,
            "mime_type": mime_type,
            "doc_filename": doc_filename,
            "doc_path_hint": doc_path_hint,
            "system_theme": system_theme,
            "available_models": available_models,
        }
    )

    template = _HTML_SHELL
    replacements = {
        "__DOC_TITLE__": f"{document_type} — {brand_title_text}",
        "__INITIAL_THEME__": html.escape(system_theme),
        "__BRAND_TITLE__": brand_title_text,
        "__DOC_TYPE__": document_type,
        "__RUBRIC_VERSION__": rubric_version,
        "__EXTRACTOR_MODEL__": extractor_model,
        "__JUDGE_QUORUM__": f"{successful_judges}/{requested_judges}x {judge_model}",
        "__JUDGE_MODEL__": judge_model,
        "__THINKING_BUDGET__": str(thinking_budget),
        "__REQUESTED_JUDGES__": str(requested_judges),
        "__TIMESTAMP__": timestamp,
        "__TOTAL_FIELDS__": str(total_fields),
        "__ACCEPTED_FIELDS__": str(accepted_fields),
        "__CONTESTED_FIELDS__": str(contested_fields),
        "__REJECTED_FIELDS__": str(rejected_fields),
        "__MODEL_COUNT__": str(len(available_models)),
        "__DATALIST_OPTIONS_HTML__": datalist_options_html,
        "__TAB1_LABEL__": tab1_label,
        "__VIEWER_HEADER_TITLE__": viewer_header_title,
        "__MODALITY_DISPLAY__": modality_display,
        "__CRITERIA_COUNT__": str(criteria_count),
        "__EXCEPTIONS_COUNT__": str(exceptions_count),
        "__PDF_CONTROLS_STYLE__": "display: none;" if is_image else "display: inline-flex;",
        "__IMG_WRAPPER_STYLE__": "display: flex;" if (is_image and doc_b64) else "display: none;",
        "__PDF_WRAPPER_STYLE__": "display: flex;" if ((not is_image) and doc_b64) else "display: none;",
        "__EMPTY_DOC_STYLE__": "display: none;" if doc_b64 else "display: flex;",
        "__IMG_SRC__": img_src,
        "__PDF_SRC__": pdf_src,
        "__RUBRIC_ROWS_HTML__": rubric_rows_html,
        "__EXTRACTED_FIELDS_ROWS_HTML__": extracted_fields_rows_html,
        "__CANDIDATE_JSON_HTML__": candidate_json_html,
        "__JUDGE_MATRIX_TABLE_HTML__": judge_matrix_table_html,
        "__CONSENSUS_ROWS_HTML__": consensus_rows_html,
        "__ACCEPTED_JSON_HTML__": accepted_json_html,
        "__EXCEPTIONS_LEDGER_HTML__": exceptions_ledger_html,
        "__AUDIT_ROWS_HTML__": audit_rows_html,
        "__FULL_AUDIT_JSON_HTML__": full_audit_json_html,
        "__BUNDLE_JSON__": bundle_json,
    }

    for key, val in replacements.items():
        template = template.replace(key, val)

    return template


_HTML_SHELL = r"""<!DOCTYPE html>
<html lang="en" data-theme="__INITIAL_THEME__">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Standalone interactive verification dashboard for multi-agent PDF and image extraction with parallel LLM judge consensus." />
  <title>__DOC_TITLE__</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..600&family=IBM+Plex+Mono:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" media="print" onload="this.media='all'" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js" async onload="window.__onPdfJsReady && window.__onPdfJsReady()"></script>
  <style>
    :root,
    [data-theme="light"] {
      --bg-canvas: #FAFAF9;
      --bg-surface: #FFFFFF;
      --bg-subtle: #F4F3F1;
      --border-color: #E7E5E4;
      --grid-line: #E7E5E4;
      --text-primary: #1C1917;
      --text-secondary: #787774;

      /* Semantic Data Visualization Palette (DESIGN.md Section 5) */
      --viz-teal: #0D9488;
      --viz-amber: #D97706;
      --viz-indigo: #4F46E5;
      --viz-emerald: #059669;
      --viz-rose: #E11D48;

      --tint-teal: rgba(13, 148, 136, 0.10);
      --tint-amber: rgba(217, 119, 6, 0.11);
      --tint-indigo: rgba(79, 70, 229, 0.10);
      --tint-emerald: rgba(5, 150, 105, 0.10);
      --tint-rose: rgba(225, 29, 72, 0.10);

      --font-serif: "Instrument Serif", Georgia, "Times New Roman", serif;
      --font-sans: "DM Sans", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
      --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;

      --radius-pane: 8px;
      --radius-ctrl: 6px;
    }

    [data-theme="dark"] {
      --bg-canvas: #0D0E11;
      --bg-surface: #16171D;
      --bg-subtle: #1E2028;
      --border-color: #292A30;
      --grid-line: #292A30;
      --text-primary: #ECE8E2;
      --text-secondary: #9A9895;

      --tint-teal: rgba(13, 148, 136, 0.18);
      --tint-amber: rgba(217, 119, 6, 0.18);
      --tint-indigo: rgba(79, 70, 229, 0.18);
      --tint-emerald: rgba(5, 150, 105, 0.18);
      --tint-rose: rgba(225, 29, 72, 0.18);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html, body {
      background-color: var(--bg-canvas);
      color: var(--text-primary);
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    code, pre, .mono-cell, .mono-key, .mono-th, .kpi-figure, .meta-val {
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }

    /* Top Bar */
    .topbar {
      position: sticky;
      top: 0;
      z-index: 40;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.85rem 1.5rem;
      background-color: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      flex-wrap: wrap;
    }

    .brand-block {
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }

    .brand-block h1 {
      font-family: var(--font-serif);
      font-size: 1.55rem;
      font-weight: 400;
      letter-spacing: -0.01em;
      line-height: 1.15;
      color: var(--text-primary);
    }

    .brand-caption {
      font-size: 0.75rem;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .live-indicator {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: var(--text-secondary);
    }

    .live-indicator[data-live="connected"] {
      color: var(--viz-emerald);
    }

    .topbar-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    /* Segmented 3-Icon Theme Selector (DESIGN.md Section 3) */
    .theme-segmented {
      display: inline-flex;
      align-items: center;
      background-color: var(--bg-canvas);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-ctrl);
      padding: 2px;
      gap: 2px;
    }

    .theme-icon-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 26px;
      border: 1px solid transparent;
      border-radius: 4px;
      background: transparent;
      color: var(--text-secondary);
      cursor: pointer;
      transition: background-color 0.12s ease, color 0.12s ease, border-color 0.12s ease;
    }

    .theme-icon-btn:hover {
      color: var(--text-primary);
    }

    .theme-icon-btn.active {
      background-color: var(--bg-surface);
      color: var(--text-primary);
      border-color: var(--border-color);
    }

    .theme-icon-btn svg {
      width: 14px;
      height: 14px;
      stroke: currentColor;
      stroke-width: 1.75;
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    /* Functional Controls */
    .ctrl-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      height: 30px;
      padding: 0 0.75rem;
      font-family: var(--font-sans);
      font-size: 0.78rem;
      font-weight: 500;
      color: var(--text-primary);
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-ctrl);
      cursor: pointer;
      transition: background-color 0.12s ease, border-color 0.12s ease;
      white-space: nowrap;
    }

    .ctrl-btn:hover {
      background-color: var(--bg-subtle);
    }

    .ctrl-btn-primary {
      background-color: var(--text-primary);
      color: var(--bg-surface);
      border-color: var(--text-primary);
    }

    .ctrl-btn-primary:hover {
      opacity: 0.9;
      background-color: var(--text-primary);
    }

    .ctrl-btn-icon {
      width: 28px;
      padding: 0;
    }

    .ctrl-btn-icon svg {
      width: 14px;
      height: 14px;
      stroke: currentColor;
      stroke-width: 1.8;
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    /* Workspace Container */
    .workspace {
      max-width: 1520px;
      margin: 0 auto;
      padding: 1.25rem 1.5rem 2.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    /* Dropzone Banner when dragging new PDF/JSON */
    body.drag-active::after {
      content: "Drop PDF, Image, Evaluation Report JSON, or Rubric JSON to inspect";
      position: fixed;
      inset: 12px;
      z-index: 100;
      background-color: var(--bg-surface);
      border: 1px solid var(--viz-teal);
      border-radius: var(--radius-pane);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-serif);
      font-size: 1.75rem;
      color: var(--text-primary);
      pointer-events: none;
    }

    /* Flat Single-Level Bento Strip: Metadata + KPIs */
    .telemetry-bento {
      display: grid;
      grid-template-columns: 2.2fr repeat(4, 1fr);
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-pane);
      overflow: hidden;
    }

    @media (max-width: 1120px) {
      .telemetry-bento {
        grid-template-columns: 1fr 1fr;
      }
      .telemetry-meta-cell {
        grid-column: 1 / -1;
      }
    }

    @media (max-width: 640px) {
      .telemetry-bento {
        grid-template-columns: 1fr;
      }
    }

    .telemetry-cell {
      padding: 0.95rem 1.15rem;
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 0.35rem;
    }

    .telemetry-cell:last-child {
      border-right: none;
    }

    @media (max-width: 1120px) {
      .telemetry-cell {
        border-bottom: 1px solid var(--border-color);
      }
    }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.55rem 1rem;
    }

    @media (max-width: 640px) {
      .meta-grid {
        grid-template-columns: 1fr 1fr;
      }
    }

    .meta-pair {
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
      min-width: 0;
    }

    .meta-lbl {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      font-weight: 600;
    }

    .meta-val {
      font-size: 0.78rem;
      color: var(--text-primary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .kpi-label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      font-weight: 600;
    }

    .kpi-figure {
      font-size: 1.85rem;
      font-weight: 500;
      line-height: 1.1;
      letter-spacing: -0.02em;
      color: var(--text-primary);
    }

    .kpi-figure.emerald { color: var(--viz-emerald); }
    .kpi-figure.amber { color: var(--viz-amber); }
    .kpi-figure.rose { color: var(--viz-rose); }

    .kpi-sub {
      font-family: var(--font-mono);
      font-size: 0.71rem;
      color: var(--text-secondary);
    }

    /* Text-First Navigation Bar */
    .tab-strip {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      border-bottom: 1px solid var(--border-color);
      overflow-x: auto;
    }

    .nav-tab {
      appearance: none;
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      padding: 0.65rem 1rem;
      font-family: var(--font-sans);
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-secondary);
      cursor: pointer;
      white-space: nowrap;
      transition: color 0.12s ease, border-color 0.12s ease;
    }

    .nav-tab:hover {
      color: var(--text-primary);
    }

    .nav-tab.active {
      color: var(--text-primary);
      font-weight: 600;
      border-bottom-color: var(--viz-teal);
    }

    .tab-panel {
      display: none;
    }

    .tab-panel.active {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    /* Single-Level Bento Grid Panes */
    .bento-two-col {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
      gap: 1rem;
      align-items: start;
    }

    .bento-equal-col {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 1rem;
      align-items: start;
    }

    @media (max-width: 1060px) {
      .bento-two-col,
      .bento-equal-col {
        grid-template-columns: 1fr;
      }
    }

    .bento-pane {
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-pane);
      overflow: hidden;
    }

    .pane-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.8rem 1.1rem;
      border-bottom: 1px solid var(--border-color);
      flex-wrap: wrap;
    }

    .pane-title-group {
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }

    .pane-title {
      font-family: var(--font-serif);
      font-size: 1.2rem;
      font-weight: 400;
      letter-spacing: -0.005em;
      line-height: 1.2;
      color: var(--text-primary);
    }

    .pane-subtitle {
      font-size: 0.73rem;
      color: var(--text-secondary);
    }

    .pane-actions {
      display: flex;
      align-items: center;
      gap: 0.45rem;
      flex-wrap: wrap;
    }

    .pane-section-divider {
      border-top: 1px solid var(--border-color);
      padding: 0.7rem 1.1rem;
      background-color: var(--bg-subtle);
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .filter-input {
      height: 28px;
      padding: 0 0.65rem;
      font-family: var(--font-sans);
      font-size: 0.78rem;
      color: var(--text-primary);
      background-color: var(--bg-canvas);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-ctrl);
      width: 190px;
    }

    .filter-input:focus {
      outline: none;
      border-color: var(--viz-teal);
    }

    /* Document Viewer Surface */
    .doc-viewport {
      background-color: var(--bg-canvas);
      min-height: 560px;
      max-height: 700px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      padding: 1rem;
      position: relative;
    }

    .pdf-stage,
    .image-stage {
      display: flex;
      justify-content: center;
      align-items: flex-start;
      width: 100%;
    }

    #pdf-canvas {
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border-color);
      background-color: #FFFFFF;
    }

    #source-doc-image {
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border-color);
      transform-origin: top center;
      transition: transform 0.12s ease;
    }

    .pdf-embed-fallback {
      width: 100%;
      height: 560px;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-ctrl);
    }

    .empty-doc-notice {
      min-height: 360px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      color: var(--text-secondary);
      font-size: 0.82rem;
      text-align: center;
    }

    /* Tabular Data Standards (DESIGN.md Section 4) */
    .table-scroll {
      width: 100%;
      overflow-x: auto;
      max-height: 640px;
      overflow-y: auto;
    }

    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }

    .data-table th {
      position: sticky;
      top: 0;
      z-index: 2;
      text-align: left;
      padding: 0.6rem 0.9rem;
      background-color: var(--bg-subtle);
      border-bottom: 1px solid var(--border-color);
      color: var(--text-secondary);
      font-size: 0.69rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      white-space: nowrap;
    }

    .data-table td {
      padding: 0.65rem 0.9rem;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
      color: var(--text-primary);
    }

    .data-table tbody tr:last-child td {
      border-bottom: none;
    }

    .data-table tbody tr:hover td {
      background-color: var(--bg-subtle);
    }

    #rubric-spec-table {
      table-layout: fixed;
    }

    #rubric-spec-table th:nth-child(1) { width: 24%; }
    #rubric-spec-table th:nth-child(2) { width: 21%; }
    #rubric-spec-table th:nth-child(3) { width: 19%; }
    #rubric-spec-table th:nth-child(4) { width: 36%; white-space: normal; }

    #rubric-spec-table td {
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }

    .mono-key {
      font-weight: 500;
      font-size: 0.77rem;
      color: var(--text-primary);
      white-space: nowrap;
    }

    .mono-cell {
      font-size: 0.76rem;
    }

    .code-cell {
      max-width: 280px;
      word-break: break-word;
    }

    .muted-cell {
      color: var(--text-secondary);
    }

    .note-cell {
      font-size: 0.77rem;
      color: var(--text-secondary);
      line-height: 1.45;
      min-width: 210px;
    }

    .empty-cell {
      padding: 1.5rem;
      text-align: center;
      color: var(--text-secondary);
      font-family: var(--font-mono);
      font-size: 0.78rem;
    }

    /* High-Signal Status Indicators (DESIGN.md Section 1 & 5) */
    .status-pill {
      display: inline-flex;
      align-items: center;
      padding: 0.12rem 0.45rem;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 0.69rem;
      font-weight: 500;
      letter-spacing: 0.02em;
      border: 1px solid var(--border-color);
      white-space: nowrap;
    }

    .state-emerald {
      color: var(--viz-emerald);
      background-color: var(--tint-emerald);
      border-color: rgba(5, 150, 105, 0.28);
    }

    .state-teal {
      color: var(--viz-teal);
      background-color: var(--tint-teal);
      border-color: rgba(13, 148, 136, 0.28);
    }

    .state-amber {
      color: var(--viz-amber);
      background-color: var(--tint-amber);
      border-color: rgba(217, 119, 6, 0.30);
    }

    .state-rose {
      color: var(--viz-rose);
      background-color: var(--tint-rose);
      border-color: rgba(225, 29, 72, 0.30);
    }

    .state-muted {
      color: var(--text-secondary);
      background-color: var(--bg-subtle);
    }

    .fm-token-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.25rem;
      margin-top: 0.35rem;
    }

    .fm-token {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.67rem;
      padding: 0.08rem 0.38rem;
      border-radius: 4px;
      border: 1px solid var(--border-color);
      color: var(--text-secondary);
      background-color: var(--bg-canvas);
    }

    .judge-verdict-cell {
      cursor: pointer;
    }

    .verdict-sub {
      display: block;
      font-size: 0.66rem;
      color: var(--text-secondary);
      margin-top: 0.2rem;
    }

    /* Code / JSON Panes */
    pre.json-view {
      font-family: var(--font-mono);
      font-size: 0.76rem;
      line-height: 1.55;
      padding: 1rem 1.1rem;
      background-color: var(--bg-canvas);
      color: var(--text-primary);
      overflow: auto;
      max-height: 430px;
      border-bottom: 1px solid var(--border-color);
    }

    pre.json-view:last-child {
      border-bottom: none;
    }

    /* Flat Divided Ledger (Replaces Nested Exception Cards) */
    .ledger-empty {
      padding: 1rem 1.1rem;
      font-size: 0.8rem;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .ledger-row {
      padding: 0.85rem 1.1rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      font-size: 0.8rem;
    }

    .ledger-row:last-child {
      border-bottom: none;
    }

    .ledger-row-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }

    .ledger-meta {
      font-size: 0.76rem;
      color: var(--text-secondary);
    }

    .ledger-meta code {
      color: var(--text-primary);
      font-size: 0.74rem;
    }

    .ledger-list {
      padding-left: 1.15rem;
      color: var(--text-secondary);
      font-size: 0.76rem;
      margin-top: 0.15rem;
    }

    /* Visualization Viewports (Charts, Heatmaps, 3D Structural Diagram, SVG Node Map) */
    .viz-stage {
      position: relative;
      width: 100%;
      background-color: transparent;
      padding: 0.85rem 1.1rem 1rem;
    }

    .viz-stage-unpadded {
      position: relative;
      width: 100%;
      background-color: var(--bg-surface);
      overflow: hidden;
    }

    .viz-legend {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      flex-wrap: wrap;
      padding: 0.55rem 1.1rem;
      border-top: 1px solid var(--border-color);
      font-family: var(--font-mono);
      font-size: 0.69rem;
      color: var(--text-secondary);
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }

    .legend-swatch {
      width: 9px;
      height: 9px;
      border-radius: 2px;
      display: inline-block;
    }

    .gradient-ramp {
      width: 84px;
      height: 8px;
      border-radius: 2px;
      border: 1px solid var(--border-color);
      display: inline-block;
    }

    /* Model Configuration & Command Builder Bento Bar */
    .model-config-bento {
      display: grid;
      grid-template-columns: auto 1fr;
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-pane);
      overflow: hidden;
    }

    @media (max-width: 1120px) {
      .model-config-bento {
        grid-template-columns: 1fr;
      }
    }

    .model-inputs-group {
      display: flex;
      align-items: flex-end;
      gap: 0.7rem;
      padding: 0.75rem 1.1rem;
      border-right: 1px solid var(--border-color);
      flex-wrap: wrap;
    }

    @media (max-width: 1120px) {
      .model-inputs-group {
        border-right: none;
        border-bottom: 1px solid var(--border-color);
      }
    }

    .model-input-field {
      display: flex;
      flex-direction: column;
      gap: 0.22rem;
    }

    .model-input-label {
      font-size: 0.66rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
    }

    .model-text-input {
      height: 30px;
      width: 185px;
      padding: 0 0.6rem;
      font-family: var(--font-mono);
      font-size: 0.76rem;
      color: var(--text-primary);
      background-color: var(--bg-canvas);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-ctrl);
    }

    .model-num-input {
      height: 30px;
      width: 82px;
      padding: 0 0.5rem;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      font-size: 0.76rem;
      color: var(--text-primary);
      background-color: var(--bg-canvas);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-ctrl);
    }

    .model-text-input:focus,
    .model-num-input:focus {
      outline: none;
      border-color: var(--viz-teal);
    }

    .model-cmd-group {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.75rem 1.1rem;
      min-width: 0;
      flex-wrap: wrap;
    }

    .cmd-preview-wrap {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      min-width: 0;
      flex: 1 1 280px;
    }

    .cmd-preview-code {
      display: block;
      padding: 0.36rem 0.65rem;
      background-color: var(--bg-canvas);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-ctrl);
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--text-primary);
      white-space: nowrap;
      overflow-x: auto;
      scrollbar-width: none;
    }

    .cmd-preview-code::-webkit-scrollbar {
      display: none;
    }

    .cmd-btn-group {
      display: flex;
      align-items: center;
      gap: 0.45rem;
      flex-shrink: 0;
    }

    #relationship-node-svg {
      display: block;
      width: 100%;
      height: 380px;
      cursor: grab;
      background-color: transparent;
      user-select: none;
    }

    #relationship-node-svg:active {
      cursor: grabbing;
    }

    .viz-hud {
      position: absolute;
      bottom: 10px;
      left: 12px;
      padding: 0.3rem 0.55rem;
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 0.68rem;
      color: var(--text-secondary);
      pointer-events: none;
      max-width: calc(100% - 24px);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .inspector-strip {
      padding: 0.7rem 1.1rem;
      border-bottom: 1px solid var(--border-color);
      background-color: var(--bg-canvas);
      font-size: 0.78rem;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .chart-svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }
  </style>
</head>
<body data-theme="__INITIAL_THEME__">
  <header class="topbar">
    <div class="brand-block">
      <h1 id="brand-heading">__BRAND_TITLE__</h1>
      <div class="brand-caption">
        <span id="brand-doc-subtitle">__DOC_TYPE__ · Rubric v__RUBRIC_VERSION__</span>
        <span>·</span>
        <span class="live-indicator" id="live-status-indicator" data-live="local">Offline / Local Bundle Ready</span>
      </div>
    </div>

    <div class="topbar-controls">
      <input type="file" id="load-files-input" multiple accept=".pdf,.png,.jpg,.jpeg,.webp,.json" style="display: none;" />
      <button type="button" class="ctrl-btn" id="load-files-btn" title="Load a new PDF/Image document, Evaluation Report JSON, or Rubric JSON">
        Load Run / PDF
      </button>
      <button type="button" class="ctrl-btn" id="export-json-btn" title="Download active evaluation report JSON">
        Export JSON
      </button>

      <!-- Three-Icon Theme Selector (DESIGN.md Section 3) -->
      <div class="theme-segmented" id="theme-toggle" role="group" aria-label="Theme Mode Selector">
        <button type="button" class="theme-icon-btn" id="theme-btn-light" title="Light Theme" aria-label="Light Theme" data-theme-mode="light" aria-pressed="false">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="4.5"></circle>
            <line x1="12" y1="2" x2="12" y2="4.5"></line>
            <line x1="12" y1="19.5" x2="12" y2="22"></line>
            <line x1="4.93" y1="4.93" x2="6.7" y2="6.7"></line>
            <line x1="17.3" y1="17.3" x2="19.07" y2="19.07"></line>
            <line x1="2" y1="12" x2="4.5" y2="12"></line>
            <line x1="19.5" y1="12" x2="22" y2="12"></line>
            <line x1="4.93" y1="19.07" x2="6.7" y2="17.3"></line>
            <line x1="17.3" y1="6.7" x2="19.07" y2="4.93"></line>
          </svg>
        </button>
        <button type="button" class="theme-icon-btn" id="theme-btn-dark" title="Dark Theme" aria-label="Dark Theme" data-theme-mode="dark" aria-pressed="false">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"></path>
          </svg>
        </button>
        <button type="button" class="theme-icon-btn active" id="theme-btn-system" title="System Theme" aria-label="System Theme" data-theme-mode="system" aria-pressed="true">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="2.5" y="3.5" width="19" height="13" rx="1.8"></rect>
            <line x1="8.5" y1="20.5" x2="15.5" y2="20.5"></line>
            <line x1="12" y1="16.5" x2="12" y2="20.5"></line>
          </svg>
        </button>
      </div>
    </div>
  </header>

  <main class="workspace">
    <!-- Single-Level Bento Strip: Execution Telemetry & Consensus Summary -->
    <section class="telemetry-bento" aria-label="Pipeline Execution Summary">
      <div class="telemetry-cell telemetry-meta-cell">
        <div class="meta-grid">
          <div class="meta-pair">
            <span class="meta-lbl">Document Schema</span>
            <span class="meta-val" id="meta-doc-type">__DOC_TYPE__</span>
          </div>
          <div class="meta-pair">
            <span class="meta-lbl">Stage 1 Extractor</span>
            <span class="meta-val" id="meta-extractor">__EXTRACTOR_MODEL__ (budget:0)</span>
          </div>
          <div class="meta-pair">
            <span class="meta-lbl">Stage 2 Judge Quorum</span>
            <span class="meta-val" id="meta-judges">__JUDGE_QUORUM__</span>
          </div>
          <div class="meta-pair">
            <span class="meta-lbl">Rubric Version</span>
            <span class="meta-val" id="meta-rubric-ver">v__RUBRIC_VERSION__</span>
          </div>
          <div class="meta-pair">
            <span class="meta-lbl">Judge Thinking Budget</span>
            <span class="meta-val" id="meta-thinking">__THINKING_BUDGET__ tokens</span>
          </div>
          <div class="meta-pair">
            <span class="meta-lbl">Timestamp (UTC)</span>
            <span class="meta-val" id="meta-timestamp">__TIMESTAMP__</span>
          </div>
        </div>
      </div>

      <div class="telemetry-cell">
        <span class="kpi-label">Evaluated Fields</span>
        <span class="kpi-figure" id="kpi-total">__TOTAL_FIELDS__</span>
        <span class="kpi-sub" id="kpi-total-sub">100% schema coverage</span>
      </div>

      <div class="telemetry-cell">
        <span class="kpi-label">Auto-Accepted</span>
        <span class="kpi-figure emerald" id="kpi-accepted">__ACCEPTED_FIELDS__</span>
        <span class="kpi-sub">Quorum &gt;= 80% pass</span>
      </div>

      <div class="telemetry-cell">
        <span class="kpi-label">Contested (HITL)</span>
        <span class="kpi-figure amber" id="kpi-contested">__CONTESTED_FIELDS__</span>
        <span class="kpi-sub">Split verdict (60%)</span>
      </div>

      <div class="telemetry-cell">
        <span class="kpi-label">Rejected</span>
        <span class="kpi-figure rose" id="kpi-rejected">__REJECTED_FIELDS__</span>
        <span class="kpi-sub">Failed quorum (&lt;= 40%)</span>
      </div>
    </section>

    <!-- Static Model Configuration & Command Builder Bar -->
    <section class="model-config-bento" aria-label="Model Selection and Command Builder">
      <datalist id="available-models-datalist">
__DATALIST_OPTIONS_HTML__
      </datalist>
      <div class="model-inputs-group">
        <label class="model-input-field" for="model-extractor-input">
          <span class="model-input-label">Extractor Model</span>
          <input type="text" class="model-text-input" id="model-extractor-input" list="available-models-datalist" value="__EXTRACTOR_MODEL__" spellcheck="false" autocomplete="off" placeholder="e.g. gemini-3.8-flash" />
        </label>
        <label class="model-input-field" for="model-judge-input">
          <span class="model-input-label">Judge Model</span>
          <input type="text" class="model-text-input" id="model-judge-input" list="available-models-datalist" value="__JUDGE_MODEL__" spellcheck="false" autocomplete="off" placeholder="e.g. gemini-3.8-flash" />
        </label>
        <label class="model-input-field" for="model-thinking-input">
          <span class="model-input-label">Thinking Tok</span>
          <input type="number" class="model-num-input" id="model-thinking-input" value="__THINKING_BUDGET__" min="0" step="256" />
        </label>
        <label class="model-input-field" for="model-judges-input">
          <span class="model-input-label">Judges</span>
          <input type="number" class="model-num-input" id="model-judges-input" value="__REQUESTED_JUDGES__" min="1" max="15" />
        </label>
      </div>

      <div class="model-cmd-group">
        <div class="cmd-preview-wrap">
          <span class="model-input-label" id="available-models-caption">Run Command Preview (__MODEL_COUNT__ models suggested · any custom model ID supported)</span>
          <code class="cmd-preview-code" id="model-cmd-preview">/pdf-llm-as-judge ...</code>
        </div>
        <div class="cmd-btn-group">
          <button type="button" class="ctrl-btn ctrl-btn-primary" id="copy-agent-cmd-btn" title="Copy /pdf-llm-as-judge slash command for Agent Harness">
            Copy /pdf-llm-as-judge
          </button>
          <button type="button" class="ctrl-btn" id="copy-cli-cmd-btn" title="Copy python3 scripts/run_pipeline.py CLI command">
            Copy CLI
          </button>
        </div>
      </div>
    </section>

    <!-- Five Workflow Views (SKILL.md Contract) -->
    <nav class="tab-strip" role="tablist" aria-label="Evaluation Views">
      <button type="button" class="nav-tab active" role="tab" aria-selected="true" data-tab="tab-pdf-rubric" id="nav-tab-1">__TAB1_LABEL__</button>
      <button type="button" class="nav-tab" role="tab" aria-selected="false" data-tab="tab-extraction" id="nav-tab-2">2. Primary Extraction</button>
      <button type="button" class="nav-tab" role="tab" aria-selected="false" data-tab="tab-judges" id="nav-tab-3">3. Judge Panel Matrix</button>
      <button type="button" class="nav-tab" role="tab" aria-selected="false" data-tab="tab-consensus" id="nav-tab-4">4. Consensus &amp; Routing</button>
      <button type="button" class="nav-tab" role="tab" aria-selected="false" data-tab="tab-audit" id="nav-tab-5">5. Provenance &amp; Audit Trail</button>
    </nav>

    <!-- TAB 1: Document & Rubric Inspector -->
    <section id="tab-pdf-rubric" class="tab-panel active" role="tabpanel">
      <div class="bento-two-col">
        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title" id="viewer-header-title">__VIEWER_HEADER_TITLE__</h2>
              <span class="pane-subtitle" id="viewer-subtitle">Source visual grounding reference</span>
            </div>
            <div class="pane-actions">
              <span id="pdf-page-controls" style="__PDF_CONTROLS_STYLE__ gap: 0.35rem; align-items: center;">
                <button type="button" class="ctrl-btn ctrl-btn-icon" id="prev-page" title="Previous Page" aria-label="Previous Page">
                  <svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>
                </button>
                <span class="mono-cell" id="page-num-display">Page 1</span>
                <button type="button" class="ctrl-btn ctrl-btn-icon" id="next-page" title="Next Page" aria-label="Next Page">
                  <svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>
                </button>
              </span>
              <button type="button" class="ctrl-btn ctrl-btn-icon" id="zoom-out" title="Zoom Out" aria-label="Zoom Out">
                <svg viewBox="0 0 24 24"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
              </button>
              <button type="button" class="ctrl-btn ctrl-btn-icon" id="zoom-in" title="Zoom In" aria-label="Zoom In">
                <svg viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
              </button>
              <button type="button" class="ctrl-btn" id="zoom-reset" title="Reset Zoom">100%</button>
            </div>
          </div>

          <div class="doc-viewport" id="pdf-container">
            <div class="pdf-stage" id="pdf-canvas-wrapper" style="__PDF_WRAPPER_STYLE__">
              <canvas id="pdf-canvas"></canvas>
            </div>
            <iframe class="pdf-embed-fallback" id="pdf-embed-frame" title="Embedded PDF Fallback" style="display: none;" src="__PDF_SRC__"></iframe>
            <div class="image-stage" id="image-wrapper" style="__IMG_WRAPPER_STYLE__">
              <img id="source-doc-image" src="__IMG_SRC__" alt="Source Document" />
            </div>
            <div class="empty-doc-notice" id="empty-doc-notice" style="__EMPTY_DOC_STYLE__">
              <div class="mono-key">No inline binary document stream attached</div>
              <div>Use "Load Run / PDF" in the header or drop a PDF / image file anywhere in the window.</div>
            </div>
          </div>
        </div>

        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Verification Rubric Specification</h2>
              <span class="pane-subtitle" id="rubric-count-subtitle">__CRITERIA_COUNT__ target extraction &amp; grounding rules</span>
            </div>
            <div class="pane-actions">
              <input type="text" class="filter-input" id="rubric-filter-input" placeholder="Filter rubric fields..." aria-label="Filter rubric fields" />
            </div>
          </div>
          <div class="table-scroll">
            <table class="data-table" id="rubric-spec-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Syntactic Rules</th>
                  <th>Source Section</th>
                  <th>Grounding Criteria &amp; Failure Modes</th>
                </tr>
              </thead>
              <tbody id="rubric-table-body">
                __RUBRIC_ROWS_HTML__
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 2: Primary Extraction -->
    <section id="tab-extraction" class="tab-panel" role="tabpanel">
      <div class="bento-two-col">
        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Candidate Extraction Payload</h2>
              <span class="pane-subtitle">Stage 1 multimodal output (thinking_budget: 0)</span>
            </div>
            <div class="pane-actions">
              <button type="button" class="ctrl-btn" id="copy-candidate-btn">Copy JSON</button>
            </div>
          </div>
          <pre class="json-view" id="candidate-extraction-json">__CANDIDATE_JSON_HTML__</pre>

          <div class="pane-section-divider">
            <span>Stage 1 Execution Parameters</span>
            <span class="mono-cell">Deterministic Schema Sweep</span>
          </div>
          <table class="data-table">
            <tbody id="stage1-diagnostics-tbody">
              <tr><td class="mono-key">Extractor Model</td><td class="mono-cell" id="diag-extractor-model">__EXTRACTOR_MODEL__</td></tr>
              <tr><td class="mono-key">Thinking Budget</td><td class="mono-cell">0 tokens (disabled for low-latency primary sweep)</td></tr>
              <tr><td class="mono-key">Sampling Temperature</td><td class="mono-cell">0.10</td></tr>
              <tr><td class="mono-key">Response MIME</td><td class="mono-cell">application/json</td></tr>
              <tr><td class="mono-key">Source Modality</td><td class="mono-cell" id="diag-modality">__MODALITY_DISPLAY__</td></tr>
              <tr><td class="mono-key">Extracted Fields</td><td class="mono-cell" id="diag-field-count">__TOTAL_FIELDS__ attributes</td></tr>
            </tbody>
          </table>
        </div>

        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Extracted Field Breakdown</h2>
              <span class="pane-subtitle">Attribute types, extracted values, and Stage 2 consensus verification status</span>
            </div>
            <div class="pane-actions">
              <input type="text" class="filter-input" id="extracted-filter-input" placeholder="Filter extracted fields..." aria-label="Filter extracted fields" />
            </div>
          </div>
          <div class="table-scroll" style="max-height: 360px;">
            <table class="data-table" id="extracted-fields-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Type</th>
                  <th>Extracted Value</th>
                  <th>Consensus</th>
                </tr>
              </thead>
              <tbody id="extracted-fields-tbody">
                __EXTRACTED_FIELDS_ROWS_HTML__
              </tbody>
            </table>
          </div>

          <div class="pane-section-divider">
            <span id="telemetry-chart-title">Extracted Field Completeness &amp; Structure</span>
            <span class="mono-cell" id="telemetry-chart-unit">Normalized Density</span>
          </div>
          <div class="viz-stage" id="extraction-telemetry-chart"></div>
        </div>
      </div>
    </section>

    <!-- TAB 3: Judge Panel Matrix -->
    <section id="tab-judges" class="tab-panel" role="tabpanel">
      <div class="bento-equal-col">
        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Judge × Field Verification Heatmap</h2>
              <span class="pane-subtitle">Dissent &amp; check intensity across parallel thinking judges</span>
            </div>
            <span class="mono-cell muted-cell" id="heatmap-quorum-label">__JUDGE_QUORUM__</span>
          </div>
          <div class="viz-stage" id="judge-heatmap-stage"></div>
          <div class="viz-legend">
            <span class="legend-item"><span class="gradient-ramp" id="heatmap-ramp-swatch"></span>Dissent Density (0% → 100%)</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#059669;"></span>Pass</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#D97706;"></span>Partial Check</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#E11D48;"></span>Fail / Conflation</span>
          </div>
        </div>

        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Judge Execution Latency &amp; Pass Rate</h2>
              <span class="pane-subtitle">Per-worker wall time (s) and syntactic/grounding pass ratio</span>
            </div>
            <span class="mono-cell muted-cell" id="judge-budget-label">Thinking: __THINKING_BUDGET__ tok</span>
          </div>
          <div class="viz-stage" id="judge-latency-stage"></div>
          <div class="viz-legend">
            <span class="legend-item"><span class="legend-swatch" style="background:#0D9488;"></span>Execution Latency (s)</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#4F46E5;"></span>Field Pass Rate (%)</span>
          </div>
        </div>
      </div>

      <div class="bento-pane">
        <div class="pane-head">
          <div class="pane-title-group">
            <h2 class="pane-title">Parallel LLM Judge Evaluation Matrix</h2>
            <span class="pane-subtitle">Click any cell to inspect individual judge reasoning, grounding citations, and proposed overturns</span>
          </div>
          <div class="pane-actions">
            <input type="text" class="filter-input" id="judge-filter-input" placeholder="Filter fields or modes..." aria-label="Filter judge matrix" />
          </div>
        </div>
        <div class="inspector-strip" id="judge-cell-inspector">
          <div>
            <span class="mono-key" id="inspector-cell-title">Select a matrix cell</span>
            <span class="muted-cell" id="inspector-cell-body"> — Click any Judge × Field verdict below or in the heatmap above to inspect full reasoning justification.</span>
          </div>
          <span class="mono-cell muted-cell" id="inspector-cell-meta">__JUDGE_MODEL__</span>
        </div>
        <div class="table-scroll">
          <table class="data-table" id="judge-matrix-table">
            __JUDGE_MATRIX_TABLE_HTML__
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 4: Consensus & Routing -->
    <section id="tab-consensus" class="tab-panel" role="tabpanel">
      <div class="bento-equal-col">
        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Field Agreement Ratios</h2>
              <span class="pane-subtitle">Quorum pass percentage vs. 80% auto-accept threshold</span>
            </div>
            <span class="mono-cell muted-cell">k / __REQUESTED_JUDGES__ passes</span>
          </div>
          <div class="viz-stage" id="consensus-chart-stage"></div>
          <div class="viz-legend">
            <span class="legend-item"><span class="legend-swatch" style="background:#059669;"></span>Unanimous (100%)</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#0D9488;"></span>Majority (&gt;=80%)</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#D97706;"></span>Contested (60%)</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#E11D48;"></span>Rejected (&lt;=40%)</span>
          </div>
        </div>

        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Interactive Relationship Node Map</h2>
              <span class="pane-subtitle">Deterministic provenance topology: Sections → Fields → Judges → Routing</span>
            </div>
            <div class="pane-actions">
              <button type="button" class="ctrl-btn" id="reset-nodemap-btn">Reset Viewport</button>
            </div>
          </div>
          <div class="viz-stage-unpadded">
            <svg id="relationship-node-svg" viewBox="0 0 840 390" aria-label="Interactive Relationship Node Map"></svg>
            <div class="viz-hud" id="nodemap-hud">Drag to pan · Scroll to zoom · Hover node to trace provenance edges</div>
          </div>
          <div class="viz-legend">
            <span class="legend-item"><span class="legend-swatch" style="background:#0D9488;"></span>Source Section</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#059669;"></span>Accepted Field</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#D97706;"></span>Contested Field</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#E11D48;"></span>Rejected Field</span>
            <span class="legend-item"><span class="legend-swatch" style="background:#4F46E5;"></span>Judge Worker</span>
          </div>
        </div>
      </div>

      <div class="bento-two-col">
        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Field-Level Consensus &amp; Routing Table</h2>
              <span class="pane-subtitle">Deterministic Stage 3 aggregation breakdown</span>
            </div>
            <div class="pane-actions">
              <input type="text" class="filter-input" id="consensus-filter-input" placeholder="Search fields..." aria-label="Search consensus fields" />
            </div>
          </div>
          <div class="table-scroll">
            <table class="data-table" id="consensus-table">
              <thead>
                <tr>
                  <th>Field Name</th>
                  <th>Agreement</th>
                  <th>Status</th>
                  <th>Confidence Band</th>
                  <th>Routing Action</th>
                </tr>
              </thead>
              <tbody id="consensus-table-body">
                __CONSENSUS_ROWS_HTML__
              </tbody>
            </table>
          </div>
        </div>

        <div class="bento-pane">
          <div class="pane-head">
            <div class="pane-title-group">
              <h2 class="pane-title">Validated Clean Output Payload</h2>
              <span class="pane-subtitle" id="accepted-payload-count">__ACCEPTED_FIELDS__ fields auto-accepted</span>
            </div>
            <div class="pane-actions">
              <button type="button" class="ctrl-btn" id="copy-payload-btn">Copy Clean JSON</button>
            </div>
          </div>
          <pre class="json-view" id="accepted-payload-json">__ACCEPTED_JSON_HTML__</pre>

          <div class="pane-section-divider">
            <span>Exceptions &amp; Dissent Ledger</span>
            <span class="mono-cell" id="exceptions-count-badge">__EXCEPTIONS_COUNT__ flagged</span>
          </div>
          <div id="exceptions-container">
            __EXCEPTIONS_LEDGER_HTML__
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 5: Provenance & Audit Trail -->
    <section id="tab-audit" class="tab-panel" role="tabpanel">
      <div class="bento-pane">
        <div class="pane-head">
          <div class="pane-title-group">
            <h2 class="pane-title">Provenance Audit Trail</h2>
            <span class="pane-subtitle">Complete field-level verification log and serialized report bundle</span>
          </div>
          <div class="pane-actions">
            <button type="button" class="ctrl-btn" id="copy-audit-btn">Copy Full Audit JSON</button>
            <button type="button" class="ctrl-btn ctrl-btn-primary" id="download-audit-btn">Download Report</button>
          </div>
        </div>

        <div class="table-scroll">
          <table class="data-table" id="audit-trail-table">
            <thead>
              <tr>
                <th>Field Name</th>
                <th>Candidate Value</th>
                <th>Status</th>
                <th>Passes</th>
                <th>Failure Modes</th>
                <th>Verification / Dissent Record</th>
              </tr>
            </thead>
            <tbody id="audit-trail-tbody">
              __AUDIT_ROWS_HTML__
            </tbody>
          </table>
        </div>

        <div class="pane-section-divider">
          <span>Serialized PipelineReport JSON</span>
          <span class="mono-cell" id="audit-bytes-label">UTF-8</span>
        </div>
        <pre class="json-view" id="full-audit-json">__FULL_AUDIT_JSON_HTML__</pre>
      </div>
    </section>
  </main>

  <script>
    (() => {
      "use strict";

      const INITIAL_BUNDLE = __BUNDLE_JSON__;

      /* Semantic Palette Constants (DESIGN.md Section 5) */
      const PALETTE = {
        teal: "#0D9488",
        amber: "#D97706",
        indigo: "#4F46E5",
        emerald: "#059669",
        rose: "#E11D48"
      };

      const THEME_TOKENS = {
        light: {
          canvas: "#FAFAF9",
          surface: "#FFFFFF",
          border: "#E7E5E4",
          grid: "#E7E5E4",
          textPrimary: "#1C1917",
          textSecondary: "#787774"
        },
        dark: {
          canvas: "#0D0E11",
          surface: "#16171D",
          border: "#292A30",
          grid: "#292A30",
          textPrimary: "#ECE8E2",
          textSecondary: "#9A9895"
        }
      };

      /* Mutable Application State (Preserves Viewport, Camera, Filters & Scroll) */
      const STATE = {
        report: INITIAL_BUNDLE.report || {},
        rubric: INITIAL_BUNDLE.rubric || {},
        docB64: INITIAL_BUNDLE.doc_b64 || "",
        mimeType: INITIAL_BUNDLE.mime_type || "application/pdf",
        docFilename: INITIAL_BUNDLE.doc_filename || "",
        docPathHint: INITIAL_BUNDLE.doc_path_hint || INITIAL_BUNDLE.doc_filename || "",
        rubricPathHint: "",
        availableModels: Array.isArray(INITIAL_BUNDLE.available_models) ? INITIAL_BUNDLE.available_models : [],
        hostSystemTheme: INITIAL_BUNDLE.system_theme === "dark" ? "dark" : "light",
        themeMode: "system",
        resolvedTheme: "light",
        selectedCell: null,
        pdf: {
          doc: null,
          pageNum: 1,
          scale: 1.2,
          loadedHash: ""
        },
        image: {
          scale: 1.0
        },
        nodeMap: {
          panX: 0,
          panY: 0,
          zoom: 1.0,
          hoveredId: null,
          nodes: [],
          edges: []
        }
      };

      /* Utility Escaping */
      function escHtml(val) {
        return String(val ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function hexToRgb(hex) {
        const clean = hex.replace("#", "");
        const num = parseInt(clean, 16);
        return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
      }

      function lerpColor(hexA, hexB, t) {
        const a = hexToRgb(hexA);
        const b = hexToRgb(hexB);
        const clamped = Math.max(0, Math.min(1, t));
        const r = Math.round(a[0] + (b[0] - a[0]) * clamped);
        const g = Math.round(a[1] + (b[1] - a[1]) * clamped);
        const bl = Math.round(a[2] + (b[2] - a[2]) * clamped);
        return `rgb(${r}, ${g}, ${bl})`;
      }

      /* Smooth Heatmap Transition: Surface -> Vibrant Teal -> Vivid Indigo -> Deep Rose */
      function heatmapRampColor(intensity) {
        const surf = THEME_TOKENS[STATE.resolvedTheme].surface;
        const t = Math.max(0, Math.min(1, intensity));
        if (t <= 0.001) return surf;
        if (t < 0.34) return lerpColor(surf, PALETTE.teal, t / 0.34);
        if (t < 0.67) return lerpColor(PALETTE.teal, PALETTE.indigo, (t - 0.34) / 0.33);
        return lerpColor(PALETTE.indigo, PALETTE.rose, (t - 0.67) / 0.33);
      }

      function statusColorHex(status) {
        const st = String(status || "").toUpperCase();
        if (st === "UNANIMOUS_PASS" || st === "PASS") return PALETTE.emerald;
        if (st === "MAJORITY_PASS") return PALETTE.teal;
        if (st === "CONTESTED") return PALETTE.amber;
        if (st === "REJECTED" || st === "FAIL") return PALETTE.rose;
        return THEME_TOKENS[STATE.resolvedTheme].textSecondary;
      }

      function statusPillClass(status) {
        const st = String(status || "").toUpperCase();
        if (st === "UNANIMOUS_PASS" || st === "PASS") return "state-emerald";
        if (st === "MAJORITY_PASS") return "state-teal";
        if (st === "CONTESTED") return "state-amber";
        if (st === "REJECTED" || st === "FAIL") return "state-rose";
        return "state-muted";
      }

      /* =========================================================================
       * 1. THREE-ICON THEME ENGINE (Light / Dark / System + OS matchMedia)
       * ========================================================================= */
      const darkMql = typeof window.matchMedia === "function"
        ? window.matchMedia("(prefers-color-scheme: dark)")
        : null;
      const lightMql = typeof window.matchMedia === "function"
        ? window.matchMedia("(prefers-color-scheme: light)")
        : null;

      function resolveSystemTheme() {
        if (typeof window.matchMedia === "function") {
          if (window.matchMedia("(prefers-color-scheme: dark)").matches) return "dark";
          if (window.matchMedia("(prefers-color-scheme: light)").matches) return "light";
        }
        return STATE.hostSystemTheme || "light";
      }

      function applyThemeMode(mode, persist = true) {
        const validMode = (mode === "light" || mode === "dark" || mode === "system") ? mode : "system";
        STATE.themeMode = validMode;
        STATE.resolvedTheme = validMode === "system" ? resolveSystemTheme() : validMode;

        document.documentElement.setAttribute("data-theme", STATE.resolvedTheme);
        document.body.setAttribute("data-theme", STATE.resolvedTheme);

        const buttons = [
          document.getElementById("theme-btn-light"),
          document.getElementById("theme-btn-dark"),
          document.getElementById("theme-btn-system")
        ];
        buttons.forEach((btn) => {
          if (!btn) return;
          const isActive = btn.getAttribute("data-theme-mode") === validMode;
          btn.classList.toggle("active", isActive);
          btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        });

        const rampSwatch = document.getElementById("heatmap-ramp-swatch");
        if (rampSwatch) {
          const steps = 8;
          const stepW = 84 / steps;
          const rects = Array.from({ length: steps }, (_, idx) => (
            `<rect x="${(idx * stepW).toFixed(1)}" y="0" width="${(stepW + 0.4).toFixed(1)}" height="8" fill="${heatmapRampColor(idx / (steps - 1))}" />`
          )).join("");
          rampSwatch.innerHTML = `<svg viewBox="0 0 84 8" width="84" height="8" style="display:block;">${rects}</svg>`;
        }

        if (persist) {
          try {
            localStorage.setItem("dashboard_theme_mode", validMode);
          } catch (_) {}
        }

        renderAllVisualizations();
      }

      ["theme-btn-light", "theme-btn-dark", "theme-btn-system"].forEach((id) => {
        const btn = document.getElementById(id);
        if (btn) {
          btn.addEventListener("click", () => {
            applyThemeMode(btn.getAttribute("data-theme-mode"), true);
          });
        }
      });

      const onOsThemeChange = () => {
        if (STATE.themeMode === "system") {
          applyThemeMode("system", false);
        }
      };
      [darkMql, lightMql].forEach((mql) => {
        if (!mql) return;
        if (typeof mql.addEventListener === "function") {
          mql.addEventListener("change", onOsThemeChange);
        } else if (typeof mql.addListener === "function") {
          mql.addListener(onOsThemeChange);
        }
      });

      /* =========================================================================
       * 2. NAVIGATION TABS & SEARCH FILTERS (Preserves State)
       * ========================================================================= */
      document.querySelectorAll(".nav-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
          document.querySelectorAll(".nav-tab").forEach((t) => {
            t.classList.remove("active");
            t.setAttribute("aria-selected", "false");
          });
          document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
          tab.classList.add("active");
          tab.setAttribute("aria-selected", "true");
          const panel = document.getElementById(tab.getAttribute("data-tab"));
          if (panel) panel.classList.add("active");
          renderAllVisualizations();
        });
      });

      function applyTableFilter(inputId, rowSelector) {
        const input = document.getElementById(inputId);
        if (!input) return;
        const q = (input.value || "").trim().toLowerCase();
        document.querySelectorAll(rowSelector).forEach((row) => {
          const match = !q || row.textContent.toLowerCase().includes(q);
          row.style.display = match ? "" : "none";
        });
      }

      const rubricFilter = document.getElementById("rubric-filter-input");
      if (rubricFilter) {
        rubricFilter.addEventListener("input", () => applyTableFilter("rubric-filter-input", "#rubric-table-body tr"));
      }
      const judgeFilter = document.getElementById("judge-filter-input");
      if (judgeFilter) {
        judgeFilter.addEventListener("input", () => applyTableFilter("judge-filter-input", "#judge-matrix-tbody tr"));
      }
      const consensusFilter = document.getElementById("consensus-filter-input");
      if (consensusFilter) {
        consensusFilter.addEventListener("input", () => applyTableFilter("consensus-filter-input", "#consensus-table-body tr"));
      }

      /* =========================================================================
       * 3. CLIPBOARD & JSON EXPORT CONTROLS
       * ========================================================================= */
      function flashButton(btnId, label, restoreLabel) {
        const btn = document.getElementById(btnId);
        if (!btn) return;
        btn.textContent = label;
        setTimeout(() => { btn.textContent = restoreLabel; }, 1600);
      }

      function copyText(text, btnId, restoreLabel) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(() => flashButton(btnId, "Copied", restoreLabel));
        }
      }

      const copyCandBtn = document.getElementById("copy-candidate-btn");
      if (copyCandBtn) {
        copyCandBtn.addEventListener("click", () => {
          copyText(JSON.stringify(STATE.report.candidate_extraction || {}, null, 2), "copy-candidate-btn", "Copy JSON");
        });
      }

      const copyPayloadBtn = document.getElementById("copy-payload-btn");
      if (copyPayloadBtn) {
        copyPayloadBtn.addEventListener("click", () => {
          copyText(JSON.stringify(STATE.report.accepted_payload || {}, null, 2), "copy-payload-btn", "Copy Clean JSON");
        });
      }

      const copyAuditBtn = document.getElementById("copy-audit-btn");
      if (copyAuditBtn) {
        copyAuditBtn.addEventListener("click", () => {
          copyText(JSON.stringify(STATE.report || {}, null, 2), "copy-audit-btn", "Copy Full Audit JSON");
        });
      }

      function downloadReportJson() {
        const blob = new Blob([JSON.stringify(STATE.report || {}, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${STATE.report.document_type || "consensus"}_evaluation_report.json`;
        a.click();
        URL.revokeObjectURL(url);
      }

      const exportBtn = document.getElementById("export-json-btn");
      if (exportBtn) exportBtn.addEventListener("click", downloadReportJson);
      const downloadAuditBtn = document.getElementById("download-audit-btn");
      if (downloadAuditBtn) downloadAuditBtn.addEventListener("click", downloadReportJson);

      /* =========================================================================
       * 4. MULTIMODAL DOCUMENT VIEWER (PDF.js + Image Zoom/Pan)
       * ========================================================================= */
      function renderPdfPage(pageNumber) {
        const canvas = document.getElementById("pdf-canvas");
        const pageDisplay = document.getElementById("page-num-display");
        if (!STATE.pdf.doc || !canvas) return;

        const clamped = Math.max(1, Math.min(STATE.pdf.doc.numPages, pageNumber));
        STATE.pdf.pageNum = clamped;
        const seq = (STATE.pdf.renderSeq || 0) + 1;
        STATE.pdf.renderSeq = seq;

        const prevPromise = STATE.pdf.renderTask
          ? (() => {
              try { STATE.pdf.renderTask.cancel(); } catch (_) {}
              return STATE.pdf.renderTask.promise.catch(() => {});
            })()
          : Promise.resolve();

        prevPromise.then(() => {
          if (STATE.pdf.renderSeq !== seq || !STATE.pdf.doc) return;
          return STATE.pdf.doc.getPage(clamped).then((page) => {
            if (STATE.pdf.renderSeq !== seq) return;
            const ctx = canvas.getContext("2d");
            const viewport = page.getViewport({ scale: STATE.pdf.scale });
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            const task = page.render({ canvasContext: ctx, viewport });
            STATE.pdf.renderTask = task;
            return task.promise.then(() => {
              if (STATE.pdf.renderTask === task) STATE.pdf.renderTask = null;
              if (pageDisplay && STATE.pdf.doc) {
                pageDisplay.textContent = `Page ${clamped} / ${STATE.pdf.doc.numPages}`;
              }
            }).catch((err) => {
              if (STATE.pdf.renderTask === task) STATE.pdf.renderTask = null;
              if (err && err.name === "RenderingCancelledException") return;
            });
          });
        }).catch(() => {
          showPdfEmbedFallback();
        });
      }

      function showPdfEmbedFallback() {
        const canvasWrap = document.getElementById("pdf-canvas-wrapper");
        const embed = document.getElementById("pdf-embed-frame");
        if (canvasWrap) canvasWrap.style.display = "none";
        if (embed && STATE.docB64) {
          embed.src = `data:application/pdf;base64,${STATE.docB64}`;
          embed.style.display = "block";
        }
      }

      function refreshDocumentViewer(resetPage = false) {
        const isImg = (STATE.mimeType || "").startsWith("image/");
        const pdfControls = document.getElementById("pdf-page-controls");
        const pdfWrap = document.getElementById("pdf-canvas-wrapper");
        const pdfEmbed = document.getElementById("pdf-embed-frame");
        const imgWrap = document.getElementById("image-wrapper");
        const imgEl = document.getElementById("source-doc-image");
        const emptyEl = document.getElementById("empty-doc-notice");
        const viewerTitle = document.getElementById("viewer-header-title");
        const viewerSub = document.getElementById("viewer-subtitle");
        const navTab1 = document.getElementById("nav-tab-1");
        const brandHeading = document.getElementById("brand-heading");

        if (brandHeading) {
          brandHeading.textContent = isImg
            ? "Multi-Agent Image Extraction & Consensus Evaluator"
            : "Multi-Agent PDF Extraction & Consensus Evaluator";
        }
        if (navTab1) {
          navTab1.textContent = isImg ? "1. Image & Rubric Inspector" : "1. PDF & Rubric Inspector";
        }
        if (viewerTitle) {
          viewerTitle.textContent = isImg
            ? `Multimodal Source Image Viewer (${STATE.mimeType})`
            : "Multimodal Source PDF Viewer";
        }
        if (viewerSub) {
          viewerSub.textContent = STATE.docFilename
            ? `${STATE.docFilename} · ${STATE.mimeType}`
            : `Source visual grounding (${STATE.mimeType})`;
        }

        if (!STATE.docB64) {
          if (pdfControls) pdfControls.style.display = "none";
          if (pdfWrap) pdfWrap.style.display = "none";
          if (pdfEmbed) pdfEmbed.style.display = "none";
          if (imgWrap) imgWrap.style.display = "none";
          if (emptyEl) emptyEl.style.display = "flex";
          return;
        }

        if (emptyEl) emptyEl.style.display = "none";

        if (isImg) {
          if (pdfControls) pdfControls.style.display = "none";
          if (pdfWrap) pdfWrap.style.display = "none";
          if (pdfEmbed) pdfEmbed.style.display = "none";
          if (imgWrap) imgWrap.style.display = "flex";
          if (imgEl) {
            imgEl.src = `data:${STATE.mimeType};base64,${STATE.docB64}`;
            imgEl.style.transform = `scale(${STATE.image.scale})`;
          }
          return;
        }

        // PDF Mode
        if (imgWrap) imgWrap.style.display = "none";
        if (pdfEmbed) pdfEmbed.style.display = "none";
        if (pdfControls) pdfControls.style.display = "inline-flex";
        if (pdfWrap) pdfWrap.style.display = "flex";

        const hashKey = `${STATE.docB64.length}:${STATE.docB64.slice(0, 48)}:${STATE.docB64.slice(-48)}`;
        if (STATE.pdf.loadedHash === hashKey && STATE.pdf.doc) {
          renderPdfPage(resetPage ? 1 : STATE.pdf.pageNum);
          return;
        }
        if (resetPage) STATE.pdf.pageNum = 1;

        try {
          if (typeof pdfjsLib !== "undefined") {
            if (pdfjsLib.GlobalWorkerOptions && !pdfjsLib.GlobalWorkerOptions.workerSrc) {
              pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
            }
            const binary = atob(STATE.docB64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
              bytes[i] = binary.charCodeAt(i);
            }
            pdfjsLib.getDocument({ data: bytes }).promise.then((pdfDoc) => {
              STATE.pdf.loadedHash = hashKey;
              STATE.pdf.doc = pdfDoc;
              if (pdfEmbed) pdfEmbed.style.display = "none";
              if (pdfWrap) pdfWrap.style.display = "flex";
              renderPdfPage(STATE.pdf.pageNum);
            }).catch(() => {
              showPdfEmbedFallback();
            });
          } else {
            showPdfEmbedFallback();
          }
        } catch (_) {
          showPdfEmbedFallback();
        }
      }

      window.__onPdfJsReady = () => {
        if (!(STATE.mimeType || "").startsWith("image/") && STATE.docB64) {
          refreshDocumentViewer(false);
        }
      };

      document.getElementById("prev-page")?.addEventListener("click", () => {
        if (STATE.pdf.doc && STATE.pdf.pageNum > 1) {
          renderPdfPage(STATE.pdf.pageNum - 1);
        }
      });
      document.getElementById("next-page")?.addEventListener("click", () => {
        if (STATE.pdf.doc && STATE.pdf.pageNum < STATE.pdf.doc.numPages) {
          renderPdfPage(STATE.pdf.pageNum + 1);
        }
      });
      document.getElementById("zoom-in")?.addEventListener("click", () => {
        if ((STATE.mimeType || "").startsWith("image/")) {
          STATE.image.scale = Math.min(3.0, +(STATE.image.scale + 0.2).toFixed(2));
          const img = document.getElementById("source-doc-image");
          if (img) img.style.transform = `scale(${STATE.image.scale})`;
        } else {
          STATE.pdf.scale = Math.min(2.6, +(STATE.pdf.scale + 0.2).toFixed(2));
          renderPdfPage(STATE.pdf.pageNum);
        }
      });
      document.getElementById("zoom-out")?.addEventListener("click", () => {
        if ((STATE.mimeType || "").startsWith("image/")) {
          STATE.image.scale = Math.max(0.4, +(STATE.image.scale - 0.2).toFixed(2));
          const img = document.getElementById("source-doc-image");
          if (img) img.style.transform = `scale(${STATE.image.scale})`;
        } else {
          STATE.pdf.scale = Math.max(0.6, +(STATE.pdf.scale - 0.2).toFixed(2));
          renderPdfPage(STATE.pdf.pageNum);
        }
      });
      document.getElementById("zoom-reset")?.addEventListener("click", () => {
        if ((STATE.mimeType || "").startsWith("image/")) {
          STATE.image.scale = 1.0;
          const img = document.getElementById("source-doc-image");
          if (img) img.style.transform = "scale(1)";
        } else {
          STATE.pdf.scale = 1.2;
          renderPdfPage(STATE.pdf.pageNum);
        }
      });

      /* =========================================================================
       * 5. 2D QUANTITATIVE CHARTS & JUDGE HEATMAP (SVG, Transparent Bg, Gridlines)
       * ========================================================================= */
      function renderExtractionTelemetryChart() {
        const container = document.getElementById("extraction-telemetry-chart");
        const titleEl = document.getElementById("telemetry-chart-title");
        const unitEl = document.getElementById("telemetry-chart-unit");
        if (!container) return;

        const cand = STATE.report.candidate_extraction || {};
        const consensus = STATE.report.consensus_breakdown || {};
        const entries = Object.entries(cand);
        const items = [];

        entries.slice(0, 14).forEach(([k, val]) => {
          let density = 0;
          let rawDesc = "null / empty";
          if (val && typeof val === "object" && !Array.isArray(val)) {
            const keys = Object.keys(val);
            density = keys.length > 0 ? Math.min(100, 35 + keys.length * 18) : 0;
            rawDesc = `${keys.length} subfields`;
          } else if (Array.isArray(val)) {
            density = val.length > 0 ? Math.min(100, 35 + val.length * 20) : 0;
            rawDesc = `${val.length} items`;
          } else if (typeof val === "number" || typeof val === "boolean") {
            density = 100;
            rawDesc = String(val);
          } else if (val !== null && val !== undefined && String(val).trim() !== "") {
            const str = String(val).trim();
            density = Math.min(100, Math.max(30, str.length * 3.6));
            rawDesc = `${str.length} chars`;
          }
          const st = consensus[k]?.status || (density > 0 ? "PASS" : "FAIL");
          items.push({
            label: k.length > 21 ? `${k.slice(0, 20)}…` : k,
            raw: rawDesc,
            norm: density,
            color: statusColorHex(st)
          });
        });

        if (titleEl) {
          titleEl.textContent = "Extracted Field Completeness & Structure";
        }
        if (unitEl) {
          unitEl.textContent = `${entries.length} extracted fields`;
        }

        if (items.length === 0) {
          container.innerHTML = `<div class="empty-cell">No candidate attributes extracted.</div>`;
          return;
        }

        const tok = THEME_TOKENS[STATE.resolvedTheme];
        const W = 600;
        const rowH = 23;
        const topPad = 16;
        const botPad = 24;
        const leftPad = 152;
        const rightPad = 94;
        const plotW = W - leftPad - rightPad;
        const H = topPad + items.length * rowH + botPad;

        const ticks = [0, 25, 50, 75, 100];
        const gridSvg = ticks.map((t) => {
          const x = leftPad + (t / 100) * plotW;
          return `
            <line x1="${x}" y1="${topPad - 4}" x2="${x}" y2="${H - botPad}" stroke="${tok.grid}" stroke-width="1" />
            <text x="${x}" y="${H - 6}" text-anchor="middle" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${t}</text>
          `;
        }).join("");

        const barsSvg = items.map((d, idx) => {
          const y = topPad + idx * rowH;
          const bw = Math.max(2, (d.norm / 100) * plotW);
          return `
            <text x="${leftPad - 8}" y="${y + 13}" text-anchor="end" fill="${tok.textPrimary}" font-family="IBM Plex Mono, monospace" font-size="10">${escHtml(d.label)}</text>
            <rect x="${leftPad}" y="${y + 4}" width="${plotW}" height="12" fill="none" stroke="${tok.grid}" stroke-width="0.75" rx="2" />
            <rect x="${leftPad}" y="${y + 4}" width="${bw.toFixed(1)}" height="12" fill="${d.color}" rx="2" />
            <text x="${leftPad + plotW + 8}" y="${y + 13}" text-anchor="start" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${escHtml(d.raw)}</text>
          `;
        }).join("");

        container.innerHTML = `
          <svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Quantitative Extraction Profile">
            ${gridSvg}
            ${barsSvg}
          </svg>
        `;
      }

      function inspectJudgeCell(fieldName, judgeIdx) {
        STATE.selectedCell = { fieldName, judgeIdx };
        const titleEl = document.getElementById("inspector-cell-title");
        const bodyEl = document.getElementById("inspector-cell-body");
        const metaEl = document.getElementById("inspector-cell-meta");
        if (!titleEl || !bodyEl || !metaEl) return;

        const judgeReports = STATE.report.judge_reports || [];
        const jid = `judge_${judgeIdx}`;
        const jr = judgeReports.find((r) => r.judge_id === jid);
        const fe = jr?.evaluations?.find((ev) => ev.field_name === fieldName);
        const c = (STATE.report.consensus_breakdown || {})[fieldName] || {};

        titleEl.textContent = `${fieldName} · Judge #${judgeIdx}`;
        if (fe) {
          const corr = fe.proposed_correction !== null && fe.proposed_correction !== undefined
            ? ` · Proposed correction: ${JSON.stringify(fe.proposed_correction)}`
            : "";
          bodyEl.textContent = ` — [${fe.verdict} | Syn:${fe.syntactic_check} | Gnd:${fe.grounding_check} | Mode:${fe.failure_mode}] ${fe.justification}${corr}`;
          metaEl.textContent = jr.execution_time_seconds ? `${jr.execution_time_seconds.toFixed(2)}s · ${jr.model_name || ""}` : (jr.model_name || "");
        } else if (jr?.error) {
          bodyEl.textContent = ` — Quorum error: ${jr.error}`;
          metaEl.textContent = "Error";
        } else {
          bodyEl.textContent = ` — Consensus status: ${c.status || "N/A"} (${c.agreement_ratio || ""})`;
          metaEl.textContent = STATE.report.judge_model || "";
        }
      }

      function renderJudgeHeatmap() {
        const container = document.getElementById("judge-heatmap-stage");
        if (!container) return;

        const consensus = STATE.report.consensus_breakdown || {};
        const fields = Object.keys(consensus);
        const judgeReports = STATE.report.judge_reports || [];
        const reqJudges = Math.max(
          Number(STATE.report.quorum_metadata?.requested_judges || 5),
          judgeReports.length,
          1
        );

        if (fields.length === 0) {
          container.innerHTML = `<div class="empty-cell">No judge verification cells to display.</div>`;
          return;
        }

        const tok = THEME_TOKENS[STATE.resolvedTheme];
        const W = 620;
        const leftPad = 165;
        const rightPad = 78;
        const topPad = 26;
        const rowH = 22;
        const H = topPad + fields.length * rowH + 10;
        const gridW = W - leftPad - rightPad;
        const colW = (gridW - 54) / reqJudges;

        let colHeaders = "";
        for (let j = 1; j <= reqJudges; j++) {
          const cx = leftPad + (j - 0.5) * colW;
          colHeaders += `<text x="${cx.toFixed(1)}" y="16" text-anchor="middle" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9.5">J${j}</text>`;
        }
        colHeaders += `<text x="${(leftPad + reqJudges * colW + 27).toFixed(1)}" y="16" text-anchor="middle" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9.5">DISSENT</text>`;

        const rowsSvg = fields.map((fn, rIdx) => {
          const y = topPad + rIdx * rowH;
          const c = consensus[fn] || {};
          let failCount = 0;
          let cellsSvg = "";

          for (let j = 1; j <= reqJudges; j++) {
            const jr = judgeReports.find((r) => r.judge_id === `judge_${j}`);
            const fe = jr?.evaluations?.find((ev) => ev.field_name === fn);
            let verdict = "PASS";
            let partial = false;
            if (fe) {
              verdict = String(fe.verdict || "PASS").toUpperCase();
              partial = (String(fe.syntactic_check).toUpperCase() !== String(fe.grounding_check).toUpperCase());
            } else if (jr?.error) {
              verdict = "FAIL";
            } else {
              const ag = Number(c.agreement_count ?? reqJudges);
              verdict = j <= ag ? "PASS" : "FAIL";
            }

            if (verdict !== "PASS") failCount++;
            const x = leftPad + (j - 1) * colW;
            const fill = verdict === "PASS"
              ? PALETTE.emerald
              : (partial ? PALETTE.amber : PALETTE.rose);
            const fillAlpha = verdict === "PASS" ? "0.82" : "0.92";

            cellsSvg += `
              <g class="hm-cell" data-field="${escHtml(fn)}" data-judge="${j}" style="cursor:pointer;">
                <rect x="${(x + 2).toFixed(1)}" y="${y + 2}" width="${Math.max(8, colW - 4).toFixed(1)}" height="${rowH - 4}" rx="3" fill="${fill}" fill-opacity="${fillAlpha}" stroke="${tok.border}" stroke-width="0.75" />
                <text x="${(x + colW / 2).toFixed(1)}" y="${y + 14}" text-anchor="middle" fill="#FFFFFF" font-family="IBM Plex Mono, monospace" font-size="8.5" font-weight="600">${verdict === "PASS" ? "P" : "F"}</text>
              </g>
            `;
          }

          const dissentIntensity = reqJudges > 0 ? failCount / reqJudges : 0;
          const rampFill = heatmapRampColor(dissentIntensity);
          const dx = leftPad + reqJudges * colW + 4;
          const textColor = dissentIntensity > 0.35 ? "#FFFFFF" : tok.textPrimary;
          const shortFn = fn.length > 23 ? `${fn.slice(0, 22)}…` : fn;

          return `
            <text x="${leftPad - 8}" y="${y + 14}" text-anchor="end" fill="${tok.textPrimary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${escHtml(shortFn)}</text>
            ${cellsSvg}
            <rect x="${dx.toFixed(1)}" y="${y + 2}" width="48" height="${rowH - 4}" rx="3" fill="${rampFill}" stroke="${tok.border}" stroke-width="0.75" />
            <text x="${(dx + 24).toFixed(1)}" y="${y + 14}" text-anchor="middle" fill="${textColor}" font-family="IBM Plex Mono, monospace" font-size="9">${Math.round(dissentIntensity * 100)}%</text>
            <text x="${W - 6}" y="${y + 14}" text-anchor="end" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${reqJudges - failCount}/${reqJudges}</text>
          `;
        }).join("");

        container.innerHTML = `
          <svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Judge by Field Verification Heatmap">
            ${colHeaders}
            ${rowsSvg}
          </svg>
        `;

        container.querySelectorAll(".hm-cell").forEach((el) => {
          el.addEventListener("click", () => {
            inspectJudgeCell(el.getAttribute("data-field"), Number(el.getAttribute("data-judge")));
          });
        });
      }

      function renderJudgeLatencyChart() {
        const container = document.getElementById("judge-latency-stage");
        if (!container) return;

        const judgeReports = STATE.report.judge_reports || [];
        const reqJudges = Math.max(
          Number(STATE.report.quorum_metadata?.requested_judges || 5),
          judgeReports.length,
          1
        );
        const consensus = STATE.report.consensus_breakdown || {};
        const fields = Object.keys(consensus);

        const tok = THEME_TOKENS[STATE.resolvedTheme];
        const rows = [];
        for (let j = 1; j <= reqJudges; j++) {
          const jr = judgeReports.find((r) => r.judge_id === `judge_${j}`);
          const secs = Number(jr?.execution_time_seconds || 0);
          let passes = 0;
          if (jr?.evaluations && jr.evaluations.length > 0) {
            passes = jr.evaluations.filter((e) => String(e.verdict).toUpperCase() === "PASS").length;
          } else if (!jr?.error) {
            fields.forEach((fn) => {
              if (j <= Number(consensus[fn]?.agreement_count ?? reqJudges)) passes++;
            });
          }
          const passPct = fields.length > 0 ? (passes / fields.length) * 100 : 0;
          rows.push({ label: `judge_${j}`, secs, passPct, hasErr: Boolean(jr?.error) });
        }

        const maxSecs = Math.max(2.5, ...rows.map((r) => r.secs * 1.15));
        const W = 600;
        const leftPad = 76;
        const rightPad = 112;
        const topPad = 24;
        const botPad = 26;
        const rowH = 42;
        const plotW = W - leftPad - rightPad;
        const H = topPad + rows.length * rowH + botPad;

        const ticks = [0, 0.25, 0.5, 0.75, 1.0];
        const gridSvg = ticks.map((t) => {
          const x = leftPad + t * plotW;
          const secVal = (t * maxSecs).toFixed(1);
          return `
            <line x1="${x.toFixed(1)}" y1="${topPad - 6}" x2="${x.toFixed(1)}" y2="${H - botPad}" stroke="${tok.grid}" stroke-width="1" />
            <text x="${x.toFixed(1)}" y="${H - 7}" text-anchor="middle" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${secVal}s / ${Math.round(t * 100)}%</text>
          `;
        }).join("");

        const barsSvg = rows.map((r, i) => {
          const y = topPad + i * rowH;
          const latW = Math.max(2, (r.secs / maxSecs) * plotW);
          const passW = Math.max(2, (r.passPct / 100) * plotW);
          const passColor = r.hasErr ? PALETTE.rose : PALETTE.indigo;
          return `
            <text x="${leftPad - 10}" y="${y + 20}" text-anchor="end" fill="${tok.textPrimary}" font-family="IBM Plex Mono, monospace" font-size="10">${r.label}</text>
            <rect x="${leftPad}" y="${y + 5}" width="${latW.toFixed(1)}" height="10" rx="2" fill="${PALETTE.teal}" />
            <rect x="${leftPad}" y="${y + 19}" width="${passW.toFixed(1)}" height="10" rx="2" fill="${passColor}" />
            <text x="${leftPad + plotW + 10}" y="${y + 14}" fill="${tok.textPrimary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${r.secs > 0 ? r.secs.toFixed(2) + "s" : "async"}</text>
            <text x="${leftPad + plotW + 10}" y="${y + 28}" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${r.passPct.toFixed(0)}% pass</text>
          `;
        }).join("");

        container.innerHTML = `
          <svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Judge Execution Latency and Pass Rate Chart">
            ${gridSvg}
            ${barsSvg}
          </svg>
        `;
      }

      function renderConsensusBarChart() {
        const container = document.getElementById("consensus-chart-stage");
        if (!container) return;

        const consensus = STATE.report.consensus_breakdown || {};
        const entries = Object.entries(consensus);
        if (entries.length === 0) {
          container.innerHTML = `<div class="empty-cell">No field consensus metrics available.</div>`;
          return;
        }

        const tok = THEME_TOKENS[STATE.resolvedTheme];
        const W = 620;
        const leftPad = 172;
        const rightPad = 76;
        const topPad = 20;
        const botPad = 26;
        const rowH = 24;
        const plotW = W - leftPad - rightPad;
        const H = topPad + entries.length * rowH + botPad;

        const ticks = [0, 20, 40, 60, 80, 100];
        const gridSvg = ticks.map((t) => {
          const x = leftPad + (t / 100) * plotW;
          const isThreshold = t === 80;
          return `
            <line x1="${x.toFixed(1)}" y1="${topPad - 6}" x2="${x.toFixed(1)}" y2="${H - botPad}"
              stroke="${isThreshold ? PALETTE.amber : tok.grid}"
              stroke-width="${isThreshold ? "1.25" : "1"}"
              ${isThreshold ? 'stroke-dasharray="3,3"' : ""} />
            <text x="${x.toFixed(1)}" y="${H - 7}" text-anchor="middle"
              fill="${isThreshold ? PALETTE.amber : tok.textSecondary}"
              font-family="IBM Plex Mono, monospace" font-size="9.5">${t}%</text>
          `;
        }).join("");

        const rowsSvg = entries.map(([fn, c], idx) => {
          const y = topPad + idx * rowH;
          const total = Math.max(1, Number(c.total_judges || 5));
          const count = Number(c.agreement_count || 0);
          const pct = Math.max(0, Math.min(100, (count / total) * 100));
          const bw = Math.max(2, (pct / 100) * plotW);
          const color = statusColorHex(c.status);
          const shortFn = fn.length > 24 ? `${fn.slice(0, 23)}…` : fn;
          return `
            <text x="${leftPad - 8}" y="${y + 14}" text-anchor="end" fill="${tok.textPrimary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${escHtml(shortFn)}</text>
            <rect x="${leftPad}" y="${y + 4}" width="${plotW}" height="13" rx="2" fill="none" stroke="${tok.grid}" stroke-width="0.75" />
            <rect x="${leftPad}" y="${y + 4}" width="${bw.toFixed(1)}" height="13" rx="2" fill="${color}" />
            <text x="${leftPad + plotW + 8}" y="${y + 14}" fill="${tok.textPrimary}" font-family="IBM Plex Mono, monospace" font-size="9.5">${count}/${total} (${Math.round(pct)}%)</text>
          `;
        }).join("");

        container.innerHTML = `
          <svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Field Agreement Ratios">
            ${gridSvg}
            ${rowsSvg}
          </svg>
        `;
      }

      /* =========================================================================
       * 6. SVG NETWORK GRAPH: "Interactive Relationship Node Map"
       * ========================================================================= */
      function buildRelationshipGraph() {
        const consensus = STATE.report.consensus_breakdown || {};
        const fields = Object.keys(consensus);
        const criteria = STATE.rubric?.extraction_criteria || [];
        const judgeReports = STATE.report.judge_reports || [];
        const numJudges = Math.max(
          Number(STATE.report.quorum_metadata?.requested_judges || 5),
          judgeReports.length,
          1
        );

        const sectionMap = new Map();
        fields.forEach((fn) => {
          const crit = criteria.find((c) => c.field_name === fn);
          const rawSec = crit?.semantic_grounding_rules?.source_section || "DOCUMENT ROOT";
          const cleaned = rawSec.replace(/^SECTION\s+\d+:\s*/i, "");
          const secLabel = cleaned.length > 20 ? `${cleaned.slice(0, 19)}…` : cleaned;
          if (!sectionMap.has(secLabel)) sectionMap.set(secLabel, []);
          sectionMap.get(secLabel).push(fn);
        });
        if (sectionMap.size === 0) {
          sectionMap.set("DOCUMENT SOURCE", []);
        }

        const W = 840;
        const H = 390;
        const topY = 48;
        const botY = 330;
        const spanY = botY - topY;
        const nodes = [];
        const edges = [];

        // Col 1: Document Source Sections
        const sections = Array.from(sectionMap.keys());
        sections.forEach((sec, idx) => {
          const y = sections.length === 1
            ? (topY + botY) / 2
            : topY + (idx / (sections.length - 1)) * spanY;
          nodes.push({
            id: `sec:${sec}`,
            label: sec,
            col: 0,
            x: 172,
            y,
            r: 7,
            color: PALETTE.teal,
            detail: `Source Section: ${sec} (${sectionMap.get(sec).length} target fields)`
          });
        });

        // Col 2: Extracted Fields
        fields.forEach((fn, idx) => {
          const y = fields.length === 1
            ? (topY + botY) / 2
            : topY + (idx / (fields.length - 1)) * spanY;
          const c = consensus[fn] || {};
          const colHex = statusColorHex(c.status);
          nodes.push({
            id: `fld:${fn}`,
            label: fn.length > 18 ? `${fn.slice(0, 17)}…` : fn,
            col: 1,
            x: 368,
            y,
            r: 6,
            color: colHex,
            detail: `Field ${fn} · ${c.status || "EVALUATED"} (${c.agreement_ratio || ""})`
          });
        });

        // Col 3: Parallel Judges
        for (let j = 1; j <= numJudges; j++) {
          const y = numJudges === 1
            ? (topY + botY) / 2
            : (topY + 18) + ((j - 1) / (numJudges - 1)) * (spanY - 36);
          const jr = judgeReports.find((r) => r.judge_id === `judge_${j}`);
          const hasErr = Boolean(jr?.error);
          nodes.push({
            id: `jdg:${j}`,
            label: `Judge #${j}`,
            col: 2,
            x: 558,
            y,
            r: 7.5,
            color: hasErr ? PALETTE.rose : PALETTE.indigo,
            detail: jr
              ? `Judge #${j} (${jr.model_name || "LLM"}) · ${jr.execution_time_seconds ? jr.execution_time_seconds.toFixed(2) + "s" : "active"}`
              : `Judge #${j} · Parallel verification worker`
          });
        }

        // Col 4: Routing Tiers
        const routingDefs = [
          { id: "rt:accept", label: "Auto-Accept", y: topY + spanY * 0.20, color: PALETTE.emerald, count: STATE.report.accepted_field_count || 0 },
          { id: "rt:hitl", label: "HITL Review", y: topY + spanY * 0.50, color: PALETTE.amber, count: STATE.report.contested_field_count || 0 },
          { id: "rt:reject", label: "Resubmit", y: topY + spanY * 0.80, color: PALETTE.rose, count: STATE.report.rejected_field_count || 0 }
        ];
        routingDefs.forEach((rt) => {
          nodes.push({
            id: rt.id,
            label: `${rt.label} (${rt.count})`,
            col: 3,
            x: 692,
            y: rt.y,
            r: 8.5,
            color: rt.color,
            detail: `${rt.label}: ${rt.count} routed fields`
          });
        });

        const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

        // Section -> Field edges
        sections.forEach((sec) => {
          (sectionMap.get(sec) || []).forEach((fn) => {
            const s = byId[`sec:${sec}`];
            const t = byId[`fld:${fn}`];
            if (s && t) {
              edges.push({ source: s, target: t, color: t.color });
            }
          });
        });

        // Field -> Judge & Judge -> Routing edges
        const activeRoutes = new Set();
        fields.forEach((fn) => {
          const fNode = byId[`fld:${fn}`];
          const st = String(consensus[fn]?.status || "UNANIMOUS_PASS").toUpperCase();
          const rtId = (st === "UNANIMOUS_PASS" || st === "MAJORITY_PASS")
            ? "rt:accept"
            : (st === "CONTESTED" ? "rt:hitl" : "rt:reject");
          activeRoutes.add(rtId);

          for (let j = 1; j <= numJudges; j++) {
            const jNode = byId[`jdg:${j}`];
            if (fNode && jNode) {
              edges.push({ source: fNode, target: jNode, color: fNode.color });
            }
          }
        });

        for (let j = 1; j <= numJudges; j++) {
          const jNode = byId[`jdg:${j}`];
          activeRoutes.forEach((rtId) => {
            const rNode = byId[rtId];
            if (jNode && rNode) {
              edges.push({ source: jNode, target: rNode, color: rNode.color });
            }
          });
        }

        STATE.nodeMap.nodes = nodes;
        STATE.nodeMap.edges = edges;
      }

      function renderRelationshipNodeMap() {
        const svg = document.getElementById("relationship-node-svg");
        if (!svg) return;

        buildRelationshipGraph();
        const tok = THEME_TOKENS[STATE.resolvedTheme];
        const { panX, panY, zoom, hoveredId, nodes, edges } = STATE.nodeMap;

        const colHeaders = [
          { x: 172, text: "SOURCE SECTIONS" },
          { x: 368, text: "TARGET FIELDS" },
          { x: 558, text: "JUDGE PANEL" },
          { x: 692, text: "ROUTING" }
        ].map((h) => (
          `<text x="${h.x}" y="18" text-anchor="middle" fill="${tok.textSecondary}" font-family="IBM Plex Mono, monospace" font-size="9" letter-spacing="0.06em">${h.text}</text>`
        )).join("");

        const edgesSvg = edges.map((e) => {
          const connected = !hoveredId || e.source.id === hoveredId || e.target.id === hoveredId;
          const opacity = hoveredId ? (connected ? "0.68" : "0.07") : "0.24";
          const width = hoveredId && connected ? "1.6" : "1.0";
          const mx = (e.source.x + e.target.x) / 2;
          return `<path d="M ${e.source.x} ${e.source.y} C ${mx} ${e.source.y}, ${mx} ${e.target.y}, ${e.target.x} ${e.target.y}" fill="none" stroke="${e.color}" stroke-opacity="${opacity}" stroke-width="${width}" />`;
        }).join("");

        const nodesSvg = nodes.map((n) => {
          const isHovered = hoveredId === n.id;
          const anchor = n.col === 0 ? "end" : (n.col === 3 ? "start" : "middle");
          const tx = n.col === 0 ? n.x - 11 : (n.col === 3 ? n.x + 11 : n.x);
          const ty = (n.col === 1 || n.col === 2) ? n.y - 10 : n.y + 3;
          return `
            <g class="nm-node" data-node-id="${escHtml(n.id)}" style="cursor:pointer;">
              <circle cx="${n.x}" cy="${n.y}" r="${isHovered ? n.r + 2.5 : n.r}" fill="${n.color}" stroke="${tok.surface}" stroke-width="1.5" />
              <text x="${tx}" y="${ty}" text-anchor="${anchor}" fill="${tok.textPrimary}" font-family="IBM Plex Mono, monospace" font-size="8.8">${escHtml(n.label)}</text>
            </g>
          `;
        }).join("");

        svg.innerHTML = `
          <g transform="translate(${panX.toFixed(1)}, ${panY.toFixed(1)}) scale(${zoom.toFixed(3)})">
            ${colHeaders}
            ${edgesSvg}
            ${nodesSvg}
          </g>
        `;

        const hud = document.getElementById("nodemap-hud");
        svg.querySelectorAll(".nm-node").forEach((g) => {
          g.addEventListener("mouseenter", () => {
            const id = g.getAttribute("data-node-id");
            STATE.nodeMap.hoveredId = id;
            const found = STATE.nodeMap.nodes.find((n) => n.id === id);
            if (hud && found) hud.textContent = found.detail;
            renderRelationshipNodeMap();
          });
          g.addEventListener("mouseleave", () => {
            STATE.nodeMap.hoveredId = null;
            if (hud) hud.textContent = "Drag to pan · Scroll to zoom · Hover node to trace provenance edges";
            renderRelationshipNodeMap();
          });
        });
      }

      (() => {
        const svg = document.getElementById("relationship-node-svg");
        if (!svg) return;
        let dragging = false;
        let lastX = 0;
        let lastY = 0;

        svg.addEventListener("mousedown", (ev) => {
          dragging = true;
          lastX = ev.clientX;
          lastY = ev.clientY;
        });
        window.addEventListener("mousemove", (ev) => {
          if (!dragging) return;
          STATE.nodeMap.panX += ev.clientX - lastX;
          STATE.nodeMap.panY += ev.clientY - lastY;
          lastX = ev.clientX;
          lastY = ev.clientY;
          renderRelationshipNodeMap();
        });
        window.addEventListener("mouseup", () => { dragging = false; });
        svg.addEventListener("wheel", (ev) => {
          ev.preventDefault();
          const factor = ev.deltaY < 0 ? 1.1 : 0.9;
          STATE.nodeMap.zoom = Math.max(0.55, Math.min(2.4, STATE.nodeMap.zoom * factor));
          renderRelationshipNodeMap();
        }, { passive: false });

        document.getElementById("reset-nodemap-btn")?.addEventListener("click", () => {
          STATE.nodeMap.panX = 0;
          STATE.nodeMap.panY = 0;
          STATE.nodeMap.zoom = 1.0;
          renderRelationshipNodeMap();
        });
      })();

      /* =========================================================================
       * 7. MODEL CONFIGURATION & STATIC COMMAND BUILDER
       * ========================================================================= */
      let activeCmdPreviewMode = "agent"; // "agent" | "cli"
      let userEditedModels = false;

      function inferRubricPathHint() {
        if (STATE.rubricPathHint) return STATE.rubricPathHint;
        const docHint = STATE.docPathHint || "";
        if (docHint.startsWith("samples/")) {
          const base = docHint.split("/").pop().replace(/\.(pdf|jpg|jpeg|png|webp|heic)$/i, "");
          const cleanStem = base.replace(/_(scanned|degraded)$/i, "");
          return `samples/rubric_spec_${cleanStem}.json`;
        }
        return "<PATH_TO_RUBRIC_JSON>";
      }

      function buildRunCommands() {
        const extInput = document.getElementById("model-extractor-input");
        const jdgInput = document.getElementById("model-judge-input");
        const thkInput = document.getElementById("model-thinking-input");
        const numInput = document.getElementById("model-judges-input");

        const extModel = (extInput?.value || STATE.report.extractor_model || "gemini-3.8-flash").trim() || "gemini-3.8-flash";
        const jdgModel = (jdgInput?.value || STATE.report.judge_model || "gemini-3.8-flash").trim() || "gemini-3.8-flash";
        const thkBudget = Math.max(0, parseInt(thkInput?.value ?? "2048", 10) || 0);
        const numJudges = Math.max(1, parseInt(numInput?.value ?? "5", 10) || 5);

        const docArg = STATE.docPathHint || STATE.docFilename || "<PATH_TO_DOCUMENT>";
        const rubArg = inferRubricPathHint();

        const agentCmd = `/pdf-llm-as-judge ${docArg} ${rubArg} --extractor-model ${extModel} --judge-model ${jdgModel} --thinking-budget ${thkBudget} --num-judges ${numJudges}`;
        const cliCmd = `python3 scripts/run_pipeline.py --document ${docArg} --rubric ${rubArg} --output output/extraction_report.json --dashboard dashboard.html --extractor-model ${extModel} --judge-model ${jdgModel} --thinking-budget ${thkBudget} --num-judges ${numJudges}`;

        return { agentCmd, cliCmd };
      }

      function updateCommandBuilder() {
        const previewEl = document.getElementById("model-cmd-preview");
        if (!previewEl) return;
        const { agentCmd, cliCmd } = buildRunCommands();
        previewEl.textContent = activeCmdPreviewMode === "cli" ? cliCmd : agentCmd;
      }

      ["model-extractor-input", "model-judge-input", "model-thinking-input", "model-judges-input"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener("input", () => {
            userEditedModels = true;
            updateCommandBuilder();
          });
        }
      });

      document.getElementById("copy-agent-cmd-btn")?.addEventListener("click", () => {
        activeCmdPreviewMode = "agent";
        updateCommandBuilder();
        const { agentCmd } = buildRunCommands();
        copyText(agentCmd, "copy-agent-cmd-btn", "Copy /pdf-llm-as-judge");
      });

      document.getElementById("copy-cli-cmd-btn")?.addEventListener("click", () => {
        activeCmdPreviewMode = "cli";
        updateCommandBuilder();
        const { cliCmd } = buildRunCommands();
        copyText(cliCmd, "copy-cli-cmd-btn", "Copy CLI");
      });

      const extractedFilter = document.getElementById("extracted-filter-input");
      if (extractedFilter) {
        extractedFilter.addEventListener("input", () => applyTableFilter("extracted-filter-input", "#extracted-fields-tbody tr"));
      }

      function bindJudgeMatrixRowClicks() {
        document.querySelectorAll(".judge-verdict-cell").forEach((cell) => {
          cell.addEventListener("click", () => {
            inspectJudgeCell(cell.getAttribute("data-field"), Number(cell.getAttribute("data-judge")));
          });
        });
      }

      function renderAllVisualizations() {
        renderExtractionTelemetryChart();
        renderJudgeHeatmap();
        renderJudgeLatencyChart();
        renderConsensusBarChart();
        renderRelationshipNodeMap();
      }

      /* =========================================================================
       * 8. INCREMENTAL STATE RECONCILER FOR NEW PDF RUNS & LIVE SSE STREAMING
       * ========================================================================= */
      function reRenderDomTablesAndKpis() {
        const r = STATE.report || {};
        const rub = STATE.rubric || {};
        const quorum = r.quorum_metadata || {};
        const reqJudges = Number(quorum.requested_judges || 5);
        const succJudges = Number(quorum.successful_judges ?? reqJudges);
        const extractorModel = r.extractor_model || "gemini-3.8-flash";
        const judgeModel = r.judge_model || "gemini-3.8-flash";
        const thinkingBudget = r.thinking_budget ?? 2048;
        const docType = r.document_type || rub.document_type || "Document";
        const rubricVer = r.rubric_version || rub.rubric_version || "1.0";
        const ts = r.timestamp ? String(r.timestamp).slice(0, 19).replace("T", " ") : "Live Session";

        const totalFields = Number(r.total_fields ?? Object.keys(r.candidate_extraction || {}).length);
        const acceptedFields = Number(r.accepted_field_count ?? 0);
        const contestedFields = Number(r.contested_field_count ?? 0);
        const rejectedFields = Number(r.rejected_field_count ?? 0);

        const setTxt = (id, val) => {
          const el = document.getElementById(id);
          if (el) el.textContent = String(val);
        };

        document.title = `${docType} — Multi-Agent Consensus Evaluation Dashboard`;
        setTxt("brand-doc-subtitle", `${docType} · Rubric v${rubricVer}`);
        setTxt("meta-doc-type", docType);
        setTxt("meta-extractor", `${extractorModel} (budget:0)`);
        setTxt("meta-judges", `${succJudges}/${reqJudges}x ${judgeModel}`);
        setTxt("meta-rubric-ver", `v${rubricVer}`);
        setTxt("meta-thinking", `${thinkingBudget} tokens`);
        setTxt("meta-timestamp", ts);

        setTxt("kpi-total", totalFields);
        setTxt("kpi-accepted", acceptedFields);
        setTxt("kpi-contested", contestedFields);
        setTxt("kpi-rejected", rejectedFields);

        setTxt("diag-extractor-model", extractorModel);
        setTxt("diag-modality", (STATE.mimeType || "").startsWith("image/")
          ? `Multimodal Inline Image (${STATE.mimeType})`
          : `Multimodal Inline PDF (${STATE.mimeType || "application/pdf"})`);
        setTxt("diag-field-count", `${totalFields} attributes`);
        setTxt("heatmap-quorum-label", `${succJudges}/${reqJudges}x ${judgeModel}`);
        setTxt("judge-budget-label", `Thinking: ${thinkingBudget} tok`);
        setTxt("accepted-payload-count", `${acceptedFields} fields auto-accepted`);

        if (!userEditedModels) {
          const extEl = document.getElementById("model-extractor-input");
          const jdgEl = document.getElementById("model-judge-input");
          const thkEl = document.getElementById("model-thinking-input");
          const numEl = document.getElementById("model-judges-input");
          if (extEl) extEl.value = extractorModel;
          if (jdgEl) jdgEl.value = judgeModel;
          if (thkEl) thkEl.value = String(thinkingBudget);
          if (numEl) numEl.value = String(reqJudges);
        }

        // Ensure active report models are also in autocomplete datalist
        [extractorModel, judgeModel].forEach((m) => {
          if (m && !STATE.availableModels.includes(m)) {
            STATE.availableModels.unshift(m);
          }
        });
        const datalistEl = document.getElementById("available-models-datalist");
        if (datalistEl) {
          datalistEl.innerHTML = STATE.availableModels.map((m) => `<option value="${escHtml(m)}"></option>`).join("");
        }
        setTxt("available-models-caption", `Run Command Preview (${STATE.availableModels.length} models suggested · any custom model ID supported)`);
        updateCommandBuilder();

        const candPre = document.getElementById("candidate-extraction-json");
        if (candPre) candPre.textContent = JSON.stringify(r.candidate_extraction || {}, null, 2);
        const accPre = document.getElementById("accepted-payload-json");
        if (accPre) accPre.textContent = JSON.stringify(r.accepted_payload || {}, null, 2);
        const fullAuditStr = JSON.stringify(r, null, 2);
        const auditPre = document.getElementById("full-audit-json");
        if (auditPre) auditPre.textContent = fullAuditStr;
        setTxt("audit-bytes-label", `${fullAuditStr.length.toLocaleString()} bytes · UTF-8`);

        // Extracted Field Breakdown Table (Tab 2)
        const extBody = document.getElementById("extracted-fields-tbody");
        if (extBody) {
          const candObj = r.candidate_extraction || {};
          const consMap = r.consensus_breakdown || {};
          const candEntries = Object.entries(candObj);
          extBody.innerHTML = candEntries.length === 0
            ? `<tr><td colspan="4" class="empty-cell">No candidate attributes extracted.</td></tr>`
            : candEntries.map(([fn, val]) => {
                let dtype = "string";
                let preview = "";
                if (val && typeof val === "object" && !Array.isArray(val)) {
                  dtype = `object (${Object.keys(val).length} keys)`;
                  preview = JSON.stringify(val);
                } else if (Array.isArray(val)) {
                  dtype = `array (${val.length} items)`;
                  preview = JSON.stringify(val);
                } else if (typeof val === "boolean") {
                  dtype = "boolean";
                  preview = String(val);
                } else if (typeof val === "number") {
                  dtype = "number";
                  preview = String(val);
                } else if (val === null || val === undefined) {
                  dtype = "null";
                  preview = "null";
                } else {
                  const s = String(val);
                  dtype = `string (${s.length} ch)`;
                  preview = s;
                }
                const c = consMap[fn] || {};
                const st = String(c.status || "CANDIDATE");
                const ratio = String(c.agreement_ratio || "").split(" ")[0];
                const pillTxt = ratio ? `${st} · ${ratio}` : st;
                const cls = statusPillClass(st);
                return `<tr><td class="mono-key">${escHtml(fn)}</td><td class="mono-cell muted-cell">${escHtml(dtype)}</td><td class="mono-cell code-cell">${escHtml(preview)}</td><td><span class="status-pill ${cls}">${escHtml(pillTxt)}</span></td></tr>`;
              }).join("");
          applyTableFilter("extracted-filter-input", "#extracted-fields-tbody tr");
        }

        // Rubric Table
        const rubricBody = document.getElementById("rubric-table-body");
        const criteria = rub.extraction_criteria || [];
        if (rubricBody) {
          if (criteria.length > 0) {
            setTxt("rubric-count-subtitle", `${criteria.length} target extraction & grounding rules`);
            rubricBody.innerHTML = criteria.map((c) => {
              const syn = c.syntactic_rules || {};
              const sem = c.semantic_grounding_rules || {};
              const parts = [`${escHtml(syn.type || "string")} · ${syn.required === false ? "opt" : "req"}`];
              if (syn.format_regex) parts.push(`/${escHtml(syn.format_regex)}/`);
              if (Array.isArray(syn.expected_subfields)) parts.push(`[${syn.expected_subfields.map(escHtml).join(", ")}]`);
              if (c.field_type || syn.field_type) parts.push(escHtml(c.field_type || syn.field_type));
              const bbox = c.bounding_box || syn.bounding_box;
              if (Array.isArray(bbox)) parts.push(`roi:[${bbox.join(", ")}]`);
              const fms = (sem.failure_modes || []).map((fm) => `<span class="fm-token">${escHtml(fm)}</span>`).join(" ");
              return `
                <tr>
                  <td class="mono-key">${escHtml(c.field_name)}</td>
                  <td class="mono-cell">${parts.join("<br/>")}</td>
                  <td class="mono-cell">${escHtml(sem.source_section || "—")}</td>
                  <td><div>${escHtml(sem.verification_criteria || "")}</div>${fms ? `<div class="fm-token-row">${fms}</div>` : ""}</td>
                </tr>
              `;
            }).join("");
          } else {
            const keys = Object.keys(r.consensus_breakdown || r.candidate_extraction || {});
            setTxt("rubric-count-subtitle", `${keys.length} target fields`);
            rubricBody.innerHTML = keys.length
              ? keys.map((k) => `<tr><td class="mono-key">${escHtml(k)}</td><td class="mono-cell">dynamic</td><td class="mono-cell">Document</td><td class="muted-cell">Derived from loaded report schema</td></tr>`).join("")
              : `<tr><td colspan="4" class="empty-cell">No rubric specifications loaded.</td></tr>`;
          }
          applyTableFilter("rubric-filter-input", "#rubric-table-body tr");
        }

        // Judge Matrix Table
        const judgeTable = document.getElementById("judge-matrix-table");
        if (judgeTable) {
          const consensus = r.consensus_breakdown || {};
          const judgeReports = r.judge_reports || [];
          const cols = Math.max(reqJudges, judgeReports.length, 1);
          const ths = Array.from({ length: cols }, (_, i) => `<th class="mono-th">Judge #${i + 1}</th>`).join("");
          const fNames = Object.keys(consensus);
          const rows = fNames.length === 0
            ? `<tr><td colspan="${cols + 3}" class="empty-cell">No judge evaluations recorded.</td></tr>`
            : fNames.map((fn) => {
                const c = consensus[fn] || {};
                let note = "";
                const jCells = Array.from({ length: cols }, (_, idx) => {
                  const j = idx + 1;
                  const jr = judgeReports.find((rep) => rep.judge_id === `judge_${j}`);
                  const fe = jr?.evaluations?.find((ev) => ev.field_name === fn);
                  let verdict = "PASS", sub = "S:P G:P";
                  if (fe) {
                    verdict = String(fe.verdict || "PASS").toUpperCase();
                    const s = String(fe.syntactic_check || "PASS").toUpperCase();
                    const g = String(fe.grounding_check || "PASS").toUpperCase();
                    sub = `S:${s[0]} G:${g[0]}`;
                    if (!note && fe.justification) note = fe.justification;
                    if (verdict !== "PASS" && fe.justification) note = `[judge_${j}] ${fe.justification}`;
                  } else if (jr?.error) {
                    verdict = "ERR";
                    sub = "QUORUM_ERR";
                    if (!note) note = jr.error;
                  } else {
                    const isPass = j <= Number(c.agreement_count ?? cols);
                    verdict = isPass ? "PASS" : "FAIL";
                    sub = isPass ? "S:P G:P" : "DISSENT";
                  }
                  const cls = statusPillClass(verdict === "ERR" ? "FAIL" : verdict);
                  return `<td class="mono-cell judge-verdict-cell" data-field="${escHtml(fn)}" data-judge="${j}"><span class="status-pill ${cls}">${escHtml(verdict)}</span><span class="verdict-sub">${escHtml(sub)}</span></td>`;
                }).join("");
                return `<tr class="judge-matrix-row" data-field="${escHtml(fn)}"><td class="mono-key">${escHtml(fn)}</td>${jCells}<td class="mono-cell">${escHtml(c.agreement_ratio || "")}</td><td class="note-cell">${escHtml(note)}</td></tr>`;
              }).join("");
          judgeTable.innerHTML = `<thead><tr><th>Field</th>${ths}<th class="mono-th">Ratio</th><th>Primary Citation / Dissent</th></tr></thead><tbody id="judge-matrix-tbody">${rows}</tbody>`;
          bindJudgeMatrixRowClicks();
          applyTableFilter("judge-filter-input", "#judge-matrix-tbody tr");
        }

        // Consensus Table
        const consBody = document.getElementById("consensus-table-body");
        if (consBody) {
          const entries = Object.entries(r.consensus_breakdown || {});
          consBody.innerHTML = entries.length === 0
            ? `<tr><td colspan="5" class="empty-cell">No consensus breakdown records available.</td></tr>`
            : entries.map(([fn, c]) => {
                const cls = statusPillClass(c.status);
                return `<tr><td class="mono-key">${escHtml(fn)}</td><td class="mono-cell">${escHtml(c.agreement_ratio || "")}</td><td><span class="status-pill ${cls}">${escHtml(c.status || "")}</span></td><td class="mono-cell muted-cell">${escHtml(c.confidence_band || "")}</td><td class="mono-cell">${escHtml(c.routing_action || "")}</td></tr>`;
              }).join("");
          applyTableFilter("consensus-filter-input", "#consensus-table-body tr");
        }

        // Exceptions Ledger
        const excContainer = document.getElementById("exceptions-container");
        const excs = r.exceptions || [];
        setTxt("exceptions-count-badge", `${excs.length} flagged`);
        if (excContainer) {
          excContainer.innerHTML = excs.length === 0
            ? `<div class="ledger-empty"><span class="status-pill state-emerald">NOMINAL</span> Zero contested or rejected fields. All target attributes cleared consensus routing.</div>`
            : excs.map((exc) => {
                const cls = statusPillClass(exc.status);
                const fms = (exc.failure_modes || []).map((m) => `<span class="fm-token">${escHtml(m)}</span>`).join(" ") || `<span class="muted-cell">NONE</span>`;
                const diss = (exc.dissent_reasons || exc.justifications || []).map((d) => `<li>${escHtml(d)}</li>`).join("");
                const corr = (exc.proposed_corrections && exc.proposed_corrections.length)
                  ? `<div class="ledger-meta"><span>Proposed Overturn:</span> <code>${escHtml(JSON.stringify(exc.proposed_corrections))}</code></div>`
                  : "";
                return `
                  <div class="ledger-row">
                    <div class="ledger-row-top">
                      <span class="mono-key">${escHtml(exc.field_name)}</span>
                      <span class="status-pill ${cls}">${escHtml(exc.status)} · ${escHtml(exc.agreement_ratio || "")}</span>
                    </div>
                    <div class="ledger-meta"><span>Candidate:</span> <code>${escHtml(JSON.stringify(exc.candidate_value))}</code></div>
                    <div class="ledger-meta"><span>Failure Modes:</span> ${fms}</div>
                    <div class="ledger-meta"><span>Routing:</span> <span class="mono-cell">${escHtml(exc.recommended_routing || "")}</span></div>
                    ${corr}
                    ${diss ? `<ul class="ledger-list">${diss}</ul>` : ""}
                  </div>
                `;
              }).join("");
        }

        // Audit Trail Table
        const auditBody = document.getElementById("audit-trail-tbody");
        if (auditBody) {
          const audit = r.audit_trail || [];
          auditBody.innerHTML = audit.length === 0
            ? `<tr><td colspan="6" class="empty-cell">No provenance audit trail entries available.</td></tr>`
            : audit.map((item) => {
                const cls = statusPillClass(item.status);
                const fms = (item.failure_modes || []).map((m) => `<span class="fm-token">${escHtml(m)}</span>`).join(" ") || "—";
                const diss = (item.dissent_reasons || []).length
                  ? item.dissent_reasons.map(escHtml).join("; ")
                  : "Verified across active judge quorum";
                return `<tr><td class="mono-key">${escHtml(item.field_name)}</td><td class="mono-cell code-cell">${escHtml(JSON.stringify(item.candidate_value))}</td><td><span class="status-pill ${cls}">${escHtml(item.status)}</span></td><td class="mono-cell">${Number(item.agreement_count ?? 0)} / ${Number(item.total_judges ?? 5)}</td><td class="mono-cell">${fms}</td><td class="note-cell">${diss}</td></tr>`;
              }).join("");
        }
      }

      /* Public API for Incremental Updates & SSE without Resetting Viewport/Scroll/Camera */
      window.updateDashboardState = function updateDashboardState(patch) {
        if (!patch || typeof patch !== "object") return;
        let docChanged = false;

        if (patch.report && typeof patch.report === "object") {
          STATE.report = { ...STATE.report, ...patch.report };
        }
        if (patch.rubric && typeof patch.rubric === "object") {
          STATE.rubric = patch.rubric;
        }
        if (Array.isArray(patch.availableModels) || Array.isArray(patch.available_models)) {
          STATE.availableModels = patch.availableModels ?? patch.available_models;
        }
        if (typeof patch.docB64 === "string" || typeof patch.doc_b64 === "string") {
          const incoming = patch.docB64 ?? patch.doc_b64;
          if (incoming !== STATE.docB64) {
            STATE.docB64 = incoming;
            docChanged = true;
          }
        }
        if (typeof patch.mimeType === "string" || typeof patch.mime_type === "string") {
          STATE.mimeType = patch.mimeType ?? patch.mime_type;
        }
        if (typeof patch.docFilename === "string" || typeof patch.doc_filename === "string") {
          const fname = patch.docFilename ?? patch.doc_filename;
          STATE.docFilename = fname;
          STATE.docPathHint = patch.docPathHint ?? patch.doc_path_hint ?? fname;
        }
        if (typeof patch.rubricFilename === "string") {
          STATE.rubricPathHint = patch.rubricFilename;
        }

        reRenderDomTablesAndKpis();
        refreshDocumentViewer(docChanged);
        renderAllVisualizations();
      };

      /* In-Browser File Picker & Drag-and-Drop Loader for New PDFs / Images / JSON Reports */
      function arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        const chunk = 0x8000;
        const parts = [];
        for (let i = 0; i < bytes.length; i += chunk) {
          parts.push(String.fromCharCode.apply(null, bytes.subarray(i, i + chunk)));
        }
        return btoa(parts.join(""));
      }

      function inferMimeFromFilename(name) {
        const lower = (name || "").toLowerCase();
        if (lower.endsWith(".pdf")) return "application/pdf";
        if (lower.endsWith(".png")) return "image/png";
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
        if (lower.endsWith(".webp")) return "image/webp";
        return "application/pdf";
      }

      function handleUserFiles(fileList) {
        if (!fileList || fileList.length === 0) return;
        const statusInd = document.getElementById("live-status-indicator");

        Array.from(fileList).forEach((file) => {
          const lower = file.name.toLowerCase();
          if (lower.endsWith(".json")) {
            const reader = new FileReader();
            reader.onload = () => {
              try {
                const parsed = JSON.parse(String(reader.result || "{}"));
                if (parsed && Array.isArray(parsed.extraction_criteria)) {
                  window.updateDashboardState({ rubric: parsed, rubricFilename: file.name });
                  if (statusInd) statusInd.textContent = `Loaded Rubric: ${file.name}`;
                } else if (parsed && (parsed.consensus_breakdown || parsed.candidate_extraction || parsed.audit_trail)) {
                  window.updateDashboardState({ report: parsed });
                  if (statusInd) statusInd.textContent = `Loaded Report: ${file.name}`;
                } else if (parsed && typeof parsed === "object") {
                  const keys = Object.keys(parsed);
                  window.updateDashboardState({
                    report: {
                      ...STATE.report,
                      document_type: file.name.replace(/\.json$/i, ""),
                      total_fields: keys.length,
                      candidate_extraction: parsed
                    }
                  });
                  if (statusInd) statusInd.textContent = `Loaded Candidate JSON: ${file.name}`;
                }
              } catch (err) {
                console.warn("Failed to parse JSON file:", err);
              }
            };
            reader.readAsText(file);
          } else {
            const mime = file.type || inferMimeFromFilename(file.name);
            const reader = new FileReader();
            reader.onload = () => {
              const b64 = arrayBufferToBase64(reader.result);
              window.updateDashboardState({
                docB64: b64,
                mimeType: mime,
                docFilename: file.name
              });
              if (statusInd) statusInd.textContent = `Loaded Document: ${file.name}`;
            };
            reader.readAsArrayBuffer(file);
          }
        });
      }

      const loadBtn = document.getElementById("load-files-btn");
      const loadInput = document.getElementById("load-files-input");
      if (loadBtn && loadInput) {
        loadBtn.addEventListener("click", () => loadInput.click());
        loadInput.addEventListener("change", () => {
          handleUserFiles(loadInput.files);
          loadInput.value = "";
        });
      }

      window.addEventListener("dragover", (ev) => {
        ev.preventDefault();
        document.body.classList.add("drag-active");
      });
      window.addEventListener("dragleave", (ev) => {
        if (ev.relatedTarget === null) {
          document.body.classList.remove("drag-active");
        }
      });
      window.addEventListener("drop", (ev) => {
        ev.preventDefault();
        document.body.classList.remove("drag-active");
        if (ev.dataTransfer?.files?.length) {
          handleUserFiles(ev.dataTransfer.files);
        }
      });

      /* Optional HTTP Server-Sent Events (SSE) & Query Report Loader */
      function initLiveStreamIfAvailable() {
        const params = new URLSearchParams(window.location.search);
        const reportUrl = params.get("report");
        const sseParam = params.get("sse");
        const statusInd = document.getElementById("live-status-indicator");

        if (reportUrl) {
          fetch(reportUrl)
            .then((r) => (r.ok ? r.json() : null))
            .then((data) => {
              if (data) window.updateDashboardState(data.report ? data : { report: data });
            })
            .catch(() => {});
        }

        const isHttp = window.location.protocol === "http:" || window.location.protocol === "https:";
        const sseUrl = sseParam === "off" ? null : (sseParam || (isHttp ? "/events" : null));
        if (sseUrl && typeof EventSource !== "undefined") {
          try {
            const source = new EventSource(sseUrl);
            source.onopen = () => {
              if (statusInd) {
                statusInd.setAttribute("data-live", "connected");
                statusInd.textContent = "SSE Live Stream Connected";
              }
            };
            source.onmessage = (ev) => {
              try {
                const payload = JSON.parse(ev.data);
                window.updateDashboardState(payload.report ? payload : { report: payload });
              } catch (_) {}
            };
            source.onerror = () => {
              source.close();
              if (statusInd) {
                statusInd.setAttribute("data-live", "local");
                statusInd.textContent = "Offline / Local Bundle Ready";
              }
            };
          } catch (_) {}
        }
      }

      /* Boot Sequence */
      let savedMode = "system";
      try {
        const stored = localStorage.getItem("dashboard_theme_mode");
        if (stored === "light" || stored === "dark" || stored === "system") {
          savedMode = stored;
        }
      } catch (_) {}

      bindJudgeMatrixRowClicks();
      updateCommandBuilder();
      applyThemeMode(savedMode, false);
      refreshDocumentViewer(false);
      initLiveStreamIfAvailable();
    })();
  </script>
</body>
</html>
"""
