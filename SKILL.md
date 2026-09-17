---
name: pdf-llm-as-judge
description: >-
  Extracts target data fields from PDF documents into structured JSON and validates each field using a parallel panel of five thinking LLM judges against a verification rubric, producing field-level consensus scores, routing decisions, audit trails, and interactive UI dashboards.
---

# Multi-Agent Document/PDF Extraction and Consensus Evaluator

Use this skill when processing PDF documents or image files (such as JPEG, PNG, or WebP scans, intake forms, identity documents, invoices, or medical reports)—including documents containing structured form layouts, fill-in-the-blank fields, checkboxes, and handwritten text (cursive or print)—that require verified extraction with provenance and consensus scoring across an ensemble of independent thinking LLM judges.

---

## Workflow Overview

The skill coordinates a three-stage pipeline:

1. **Stage 1: Primary Multimodal Extraction**
   - High-throughput multimodal model (`gemini-3.8-flash`) parses the PDF document or image file with thinking tokens disabled (`thinking_budget: 0`).
   - Multimodal perception natively processes complex form grids, checkboxes, filled-in blank lines, and handwritten notes/cursive alongside typed vector text.
   - When handwritten text is detected on forms, the extractor leverages template subtraction (mentally subtracting rigid comb dividers and baseline ticks) and stroke ink color/contrast (such as blue or black pen ink vs. pre-printed template ink) to distinguish genuine pen strokes from touching or overlapping printed form lines, comb tick marks, boxes, and template labels.
   - Generates an initial candidate JSON object adhering to the target field schema.

2. **Stage 2: Parallel 5x Thinking Judge Panel**
   - Spawns 5 concurrent, isolated instances of `gemini-3.6-flash` with thinking budget enabled (`thinking_budget: 2048+`).
   - Each judge independently cross-references the candidate extraction against both the raw visual document/image and the verification rubric (syntactic rules and semantic grounding criteria).
   - Judges execute an **Adversarial Forensic Protocol** within their thinking process to falsify potential template-ink conflation errors (e.g. baseline comb ticks or vertical dividers misread as character stems). Judges inspect visual features, overturn conflated characters, and return verdicts, failure modes (including `TEMPLATE_INK_CONFLATION`), visual citations, and proposed corrections.

3. **Stage 3: Consensus Aggregation and Routing**
   - Deterministic post-processing computes field agreement ratios (k / 5 passes).
   - Classifies fields into confidence tiers:
     - 5/5: `UNANIMOUS_PASS` (Auto-Accept)
     - 4/5: `MAJORITY_PASS` (Auto-Accept with Warning)
     - 3/5: `CONTESTED` (Route to Human-in-the-Loop)
     - <= 2/5: `REJECTED` (Flag for Resubmission / Rerun)
   - Emits accepted data, exception reports with dissent justifications, an audit trail, and an interactive UI Dashboard.

### Handwritten Text & Overlapping Form Element Guidelines

When processing forms containing handwritten text (in PDF or image format), especially structured forms with segmented boxes, comb dividers, or baseline tick marks:

#### 1. Extractor Template Subtraction
- **Visual Subtraction**: When extracting from segmented boxes or comb-tick fields, mentally subtract rigid pre-printed template marks before recognizing glyphs.
- **Ink Color & Contrast**: Use pen ink characteristics (natural curvature, ink saturation, pressure taper, blue/black ballpoint/gel pigment) to isolate handwriting from touching or overlapping printed form elements.

#### 2. Adversarial Forensic Judge Protocol (Thinking Process)
Judges must utilize their 2048+ token thinking budget to execute this 4-step adversarial protocol to falsify potential OCR/extraction errors caused by template-ink conflation:
1. **Template Baseline Mapping**: Identify repeating pre-printed elements (box borders, baseline ticks, comb dividers) and calculate their periodic intervals and grid spacing.
2. **Forensic Stroke Isolation**: Mentally subtract all pre-printed template lines. For every character where an ascender, descender, or vertical stem aligns with a known template tick interval, perform a stroke continuity test: does the stroke exhibit genuine pen ink characteristics (curvature, taper, saturation, pressure variations), or does it directly align with and rely on the rigid, pre-printed template mark?
3. **Conflation Falsification Check**: Specifically test whether the candidate character is an optical artifact of a simpler glyph touching a printed mark:
   - Is an alleged `'a'` actually an `'o'` touching a baseline tick mark?
   - Is an alleged `'4'` actually a `'C'` or `'L'` touching a vertical comb tick?
   - Is an alleged `'d'` or `'q'` actually an `'o'` intersecting a vertical divider?
   - Is an alleged `'H'` actually two separate characters or an `'I'`/`'1'` touching a vertical cell wall?
   - Is an alleged `'B'` or `'8'` an open loop (such as `'3'` or `'C'`) touching a cell boundary?
4. **Verdict Determination & True Ink Overturn**: If any character in the candidate extraction relies on a pre-printed form mark to justify its classification, the judge MUST fail the grounding check (`FAIL`), fail the verdict (`FAIL`), assign `failure_mode: "TEMPLATE_INK_CONFLATION"`, detail the falsification in `justification`, and output the true ink-only value in `proposed_correction`.

#### 3. Preservation of Other Document Types
This adversarial protocol and template subtraction specifically target handwritten entries on structured forms and comb layouts. It does not modify or degrade the skill's ability to process other document types (such as standard typed PDFs, digital vector documents, molecular lab reports, or image scans without handwriting), which continue to be extracted and verified with standard high precision.

---

## UI Dashboard Support

The skill generates a self-contained interactive UI dashboard (`ui/index.html`) upon executing the pipeline for the given input document and rubric specification. The dashboard renders inside browser and iframe environments.

Features (Organized across 5 tabs from left to right):
1. **Document & Rubric Inspector**: Visual multi-page PDF or image viewer with zoom/pan alongside rubric specifications.
2. **Primary Extraction**: Candidate extraction viewer with thinking token diagnostics and model parameters.
3. **Judge Panel Matrix**: 5-column parallel judge grid displaying reasoning tokens, field verdicts, and citations.
4. **Consensus & Routing**: Summary statistics cards, field agreement ratios (k / 5), confidence bands, and validated output payload.
5. **Provenance & Audit Trail**: Complete JSON audit trail viewer with one-click clipboard copy and report download.
- Dark and Light mode toggle button with persistent preferences.

---

## Agent Instructions for Executing This Skill

When a user requests document extraction and verification:

### Step 1: Authentication & Environment
The pipeline natively supports both Cloud ADC and AI Studio:
- **Cloud ADC (Default / Recommended)**: If the user is authenticated via Application Default Credentials (`gcloud auth application-default login` or active `gcloud` session), the pipeline automatically resolves credentials and routes requests. **Do NOT prompt the user for an API key** if ADC or `gcloud` is available.
- **AI Studio (Alternative)**: If `GEMINI_API_KEY` is present in the environment or passed via `--api-key`, the client will use the AI Studio endpoint.

### Step 2: Verify Input Files
Check that the required input files exist:
- Target document/image (e.g., PDF, JPEG, PNG, WebP)
- Target Rubric Specification JSON (e.g., [samples/rubric_spec_healthcare_patient_intake_form.json](samples/rubric_spec_healthcare_patient_intake_form.json) or [samples/rubric_spec_cancer_screening_lab_report.json](samples/rubric_spec_cancer_screening_lab_report.json))

If a custom rubric is not supplied, inspect the document and construct a rubric specification adhering to the format defined in the samples.

### Step 3: Execute the Pipeline and Generate Dashboard
Run the pipeline runner script specifying JSON output and UI dashboard output. The CLI accepts `--document`, `--image`, or `--pdf` for the input file path. Custom models can be passed via `--extractor-model` and `--judge-model`:

```bash
python3 scripts/run_pipeline.py \
  --document <PATH_TO_DOCUMENT_OR_IMAGE> \
  --rubric <PATH_TO_RUBRIC_JSON> \
  --output <PATH_TO_OUTPUT_JSON> \
  --dashboard <PATH_TO_DASHBOARD_HTML> \
  --extractor-model gemini-3.8-flash \
  --judge-model gemini-3.6-flash
```

Example using the sample files:

```bash
python3 scripts/run_pipeline.py \
  --document samples/healthcare_patient_intake_form.pdf \
  --rubric samples/rubric_spec_healthcare_patient_intake_form.json \
  --output output/extraction_report.json \
  --dashboard ui/index.html
```

If evaluating an already extracted candidate JSON, use the candidate evaluation script:

```bash
python3 scripts/evaluate_candidate.py \
  --document <PATH_TO_DOCUMENT_OR_IMAGE> \
  --rubric <PATH_TO_RUBRIC_JSON> \
  --candidate-json <PATH_TO_CANDIDATE_JSON> \
  --output <PATH_TO_OUTPUT_JSON> \
  --dashboard ui/index.html
```

### Step 4: Emit Dashboard Artifact for Agent Harness
When running within an agent harness or framework:
1. **Interactive Artifact Delivery**: After the pipeline generates the standalone dashboard HTML file (e.g., `ui/index.html` or `output/<stem>_dashboard.html`), the agent can emit this HTML as a user-facing artifact named `dashboard.html`.
2. **Direct Visual Display**: This enables the harness to render the interactive HTML dashboard directly in preview panes or web views without copying files.

### Step 5: Interpret Output and Present Summary
Read the generated output JSON and present a clear summary:
1. **Accepted Fields**: List all fields that attained unanimous or majority consensus along with their extracted values.
2. **Contested Fields (HITL)**: Highlight fields that received split verdicts (e.g., 3/5 passes). Present the candidate value, the specific dissent reasons from the judges, and any proposed corrections for user confirmation.
3. **Rejected Fields**: Highlight fields that failed verification (<= 2/5 passes), cite the detected failure modes (e.g., `FORMAT_MISMATCH`, `HALLUCINATION`, `FACILITY_ADDRESS_CONFUSED_AS_PATIENT`), and list the judge justifications.
4. **UI Dashboard Link**: Provide a clickable markdown link to the generated dashboard file (e.g., [ui/index.html](ui/index.html) or [output/extraction_report_dashboard.html](output/extraction_report_dashboard.html)).

---

## File References

- Architecture Diagram: [assets/pipeline_workflow_dashboard.png](assets/pipeline_workflow_dashboard.png)
- Dashboard UI: [ui/index.html](ui/index.html)
- Dashboard Generator: [pdf_consensus_evaluator/dashboard_generator.py](pdf_consensus_evaluator/dashboard_generator.py)
- Pipeline Runner: [scripts/run_pipeline.py](scripts/run_pipeline.py)
- Candidate Evaluator: [scripts/evaluate_candidate.py](scripts/evaluate_candidate.py)
- Core Pipeline Orchestrator: [pdf_consensus_evaluator/pipeline.py](pdf_consensus_evaluator/pipeline.py)
- Sample PDF: [samples/healthcare_patient_intake_form.pdf](samples/healthcare_patient_intake_form.pdf)
- Sample Rubric Spec: [samples/rubric_spec_healthcare_patient_intake_form.json](samples/rubric_spec_healthcare_patient_intake_form.json)
