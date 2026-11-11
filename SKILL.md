---
name: pdf-llm-as-judge
description: >-
  Extracts target data fields from PDF documents and images into structured JSON and validates each field using a parallel panel of five thinking LLM judges against a verification rubric, producing field-level consensus scores, routing decisions, audit trails, and an interactive HTML dashboard.
---

# Multi-Agent Document/PDF Extraction and Consensus Evaluator

Use this skill when processing PDF documents or image files (such as JPEG, PNG, or WebP scans, intake forms, identity documents, invoices, or medical reports), including documents containing structured form layouts, fill-in-the-blank fields, checkboxes, and handwritten text (cursive or print), that require verified extraction with provenance and consensus scoring across an ensemble of independent thinking LLM judges.

---

## Workflow Overview

The skill coordinates a three-stage pipeline:

1. **Stage 1: Primary Multimodal Extraction**
   - Multimodal model (`gemini-3.8-flash`) parses the PDF document or image file with thinking tokens disabled (`thinking_budget: 0`).
   - Multimodal perception processes form grids, checkboxes, filled-in blank lines, and handwritten notes/cursive alongside typed vector text.
   - When handwritten text is detected on forms, the extractor applies template subtraction (mentally subtracting rigid comb dividers and baseline ticks) and stroke ink color/contrast (such as blue or black pen ink versus pre-printed template ink) to distinguish genuine pen strokes from touching or overlapping printed form lines, comb tick marks, boxes, and template labels.
   - Generates an initial candidate JSON object adhering to the target field schema.

2. **Stage 2: Parallel 5x Thinking Judge Panel**
   - Spawns 5 concurrent, isolated instances of `gemini-3.8-flash` with thinking budget enabled (`thinking_budget: 2048+`).
   - Each judge independently cross-references the candidate extraction against both the raw visual document/image and the verification rubric (syntactic rules and semantic grounding criteria).
   - Judges execute an **Adversarial Forensic Protocol** within their thinking process to falsify potential template-ink conflation errors (e.g., baseline comb ticks or vertical dividers misread as character stems). Judges inspect visual features, overturn conflated characters, and return verdicts, failure modes (including `TEMPLATE_INK_CONFLATION`), visual citations, and proposed corrections.

3. **Stage 3: Consensus Aggregation and Routing**
   - Deterministic post-processing computes field agreement ratios (`k / 5` passes).
   - Classifies fields into confidence tiers:
     - `5/5`: `UNANIMOUS_PASS` (Auto-Accept)
     - `4/5`: `MAJORITY_PASS` (Auto-Accept with Warning)
     - `3/5`: `CONTESTED` (Route to Human-in-the-Loop)
     - `<= 2/5`: `REJECTED` (Flag for Resubmission / Rerun)
   - Emits accepted data, exception reports with dissent justifications, an audit trail, and an interactive HTML dashboard (`dashboard.html`).

### Handwritten Text and Overlapping Form Element Guidelines

When processing forms containing handwritten text (in PDF or image format), especially structured forms with segmented boxes, comb dividers, or baseline tick marks:

#### 1. Extractor Template Subtraction
- **Visual Subtraction**: When extracting from segmented boxes or comb-tick fields, subtract rigid pre-printed template marks before recognizing glyphs.
- **Ink Color and Contrast**: Use pen ink characteristics (natural curvature, ink saturation, pressure taper, blue/black ballpoint/gel pigment) to isolate handwriting from touching or overlapping printed form elements.

#### 2. Adversarial Forensic Judge Protocol (Thinking Process)
Judges use their `2048+` token thinking budget to execute a 4-step adversarial protocol to falsify potential OCR/extraction errors caused by template-ink conflation:
1. **Template Baseline Mapping**: Identify repeating pre-printed elements (box borders, baseline ticks, comb dividers) and calculate their periodic intervals and grid spacing.
2. **Forensic Stroke Isolation**: Subtract pre-printed template lines. For every character where an ascender, descender, or vertical stem aligns with a known template tick interval, perform a stroke continuity test: determine whether the stroke exhibits pen ink characteristics (curvature, taper, saturation, pressure variations) or aligns with the rigid pre-printed template mark.
3. **Conflation Falsification Check**: Test whether the candidate character is an optical artifact of a simpler glyph touching a printed mark:
   - Is an alleged `'a'` actually an `'o'` touching a baseline tick mark?
   - Is an alleged `'4'` actually a `'C'` or `'L'` touching a vertical comb tick?
   - Is an alleged `'d'` or `'q'` actually an `'o'` intersecting a vertical divider?
   - Is an alleged `'H'` actually two separate characters or an `'I'`/`'1'` touching a vertical cell wall?
   - Is an alleged `'B'` or `'8'` an open loop (such as `'3'` or `'C'`) touching a cell boundary?
4. **Verdict Determination and True Ink Overturn**: If any character in the candidate extraction relies on a pre-printed form mark to justify its classification, fail the grounding check (`FAIL`), fail the verdict (`FAIL`), assign `failure_mode: "TEMPLATE_INK_CONFLATION"`, detail the falsification in `justification`, and output the true ink-only value in `proposed_correction`.

#### 3. Region-of-Interest Cropping, Morphological Line Suppression, and Comb Auto-Detection
When processing pre-printed forms containing segmented character comb boxes with baseline tick marks or dividers (e.g., ID numbers, postal codes, account numbers):
- **Architectural Division**:
  - **Stage 1 (Primary Extractor)**: Operates with thinking tokens disabled (`thinking_budget: 0`) as a fast multimodal sweep across the document or image.
  - **Stage 2 (Judge Panel)**: Operates with thinking tokens enabled (`thinking_budget: 2048+`) combined with image preprocessing for forensic verification.
- **Problem**: Full-page downsampling causes small comb boxes (`<1.5%` of total page area) to blend baseline tick marks with handwritten pen strokes, leading to false optical conflation (e.g., `'C'` touching a tick misclassified as `'4'`, or `'o'` touching a tick misclassified as `'a'`).
- **Dual-Engine Filtering (`utils/comb_filter.py`)**:
  - `crop_field_roi`: Crops the field region based on normalized coordinates `[ymin, xmin, ymax, xmax]` with 3% contextual padding, preserving native scan resolution without downsampling artifacts.
  - `suppress_vertical_ticks`: Isolates and bleaches thin vertical tick marks and dividers (`<= 2px`) to background white (`255`) using morphological structuring elements and vectorized run-length analysis, while preserving curved character pen strokes (`>= 3px`) for both black ballpoint/gel pen ink and colored ink.
  - `suppress_comb_lines`: Combines HSV color space thresholding for colored ink with morphological line suppression for black ballpoint/gel pen ink.
  - `auto_detect_comb_rois`: Uses horizontal and vertical projection profiles to locate segmented comb box rows when `bounding_box` is omitted from the rubric specification.
- **Rubric Specification Support**:
  - Rubrics configure comb fields with `field_type: "segmented_comb_box"` and optional `bounding_box: [ymin, xmin, ymax, xmax]`.
- **Forensic Comb Audit and Overrule (Stage 2 Judge Panel)**:
  - Judges audit candidate extractions against the high-resolution, morphologically cleaned crop, falsify template-ink conflations (e.g., `'4'` -> `'C'`, `'a'` -> `'o'`), and overrule candidate values with `failure_mode: "TEMPLATE_INK_CONFLATION"` and true ink corrections.

#### 4. Preservation of Other Document Types
This adversarial protocol, comb region cropping, and template subtraction target handwritten entries on structured forms and comb layouts without altering processing of standard typed PDFs, digital vector documents, molecular lab reports, or image scans without handwriting.

---

## UI Dashboard Support

The skill generates a self-contained interactive HTML dashboard (`dashboard.html`) upon executing the pipeline for an input document and rubric specification.

Features (organized across 5 tabs from left to right):
1. **Document & Rubric Inspector**: Multi-page PDF or image viewer with zoom/page controls alongside rubric specifications.
2. **Primary Extraction**: Candidate extraction viewer with thinking token diagnostics, model parameters, `Extracted Field Breakdown` table, and completeness chart.
3. **Judge Panel Matrix**: `Judge x Field Verification Heatmap`, latency chart, and parallel judge matrix displaying reasoning justifications, field verdicts, and citations.
4. **Consensus & Routing**: Summary statistics, field agreement ratios (`k / 5`), `Interactive Relationship Node Map`, confidence bands, and validated output payload.
5. **Provenance & Audit Trail**: Complete JSON audit trail viewer with clipboard copy and report download.
- **Model Configuration & Command Builder**: Editable Extractor and Judge model text inputs backed by dynamically discovered `<datalist>` suggestions, generating copyable `/pdf-llm-as-judge` and CLI commands.
- Light, Dark, and System mode theme selector with persistent preferences and live OS appearance tracking.

---

## Execution Steps

### Step 1: Authentication and Environment
The pipeline supports Application Default Credentials (ADC) and API key authentication:
- **Application Default Credentials (ADC)**: When ADC credentials exist in the environment (`gcloud auth application-default login`), the client resolves credentials automatically without requiring an API key.
- **API Key**: If `GEMINI_API_KEY` is set in the environment or passed via `--api-key`, the client uses the API key endpoint.

### Step 2: Verify Input Files
Confirm that the required input files exist relative to the project root:
- Target document or image (e.g., PDF, JPEG, PNG, WebP)
- Target Rubric Specification JSON (e.g., [samples/rubric_spec_healthcare_patient_intake_form.json](samples/rubric_spec_healthcare_patient_intake_form.json) or [samples/rubric_spec_cancer_screening_lab_report.json](samples/rubric_spec_cancer_screening_lab_report.json))

If a custom rubric is not supplied, inspect the document and construct a rubric specification matching the schema in `samples/`.

### Step 3: Execute the Pipeline and Generate Dashboard
Run the pipeline runner script specifying JSON output and HTML dashboard output. Custom models can be passed via `--extractor-model` and `--judge-model`:

```bash
python3 scripts/run_pipeline.py \
  --document <PATH_TO_DOCUMENT_OR_IMAGE> \
  --rubric <PATH_TO_RUBRIC_JSON> \
  --output <PATH_TO_OUTPUT_JSON> \
  --dashboard dashboard.html \
  --extractor-model gemini-3.8-flash \
  --judge-model gemini-3.8-flash
```

Example using sample files:

```bash
python3 scripts/run_pipeline.py \
  --document samples/healthcare_patient_intake_form.pdf \
  --rubric samples/rubric_spec_healthcare_patient_intake_form.json \
  --output output/extraction_report.json \
  --dashboard dashboard.html
```

To evaluate an existing candidate JSON without re-running Stage 1:

```bash
python3 scripts/evaluate_candidate.py \
  --document <PATH_TO_DOCUMENT_OR_IMAGE> \
  --rubric <PATH_TO_RUBRIC_JSON> \
  --candidate-json <PATH_TO_CANDIDATE_JSON> \
  --output <PATH_TO_OUTPUT_JSON> \
  --dashboard dashboard.html
```

### Step 4: Output Summary
Present the structured evaluation results from the output JSON:
1. **Accepted Fields**: Fields attaining unanimous (`5/5`) or majority (`4/5`) consensus and their verified values.
2. **Contested Fields (HITL)**: Fields receiving split verdicts (`3/5`), including dissent justifications and proposed corrections.
3. **Rejected Fields**: Fields failing verification (`<= 2/5`), detected failure modes, and judge justifications.
4. **Dashboard Reference**: Link to [dashboard.html](dashboard.html).

---

## File References

- Architecture Diagram: [assets/pipeline_workflow_dashboard.png](assets/pipeline_workflow_dashboard.png)
- Dashboard UI: [dashboard.html](dashboard.html)
- Design Specification: [DESIGN.md](DESIGN.md)
- Dashboard Generator: [pdf_consensus_evaluator/dashboard_generator.py](pdf_consensus_evaluator/dashboard_generator.py)
- Pipeline Runner: [scripts/run_pipeline.py](scripts/run_pipeline.py)
- Candidate Evaluator: [scripts/evaluate_candidate.py](scripts/evaluate_candidate.py)
- Core Pipeline Orchestrator: [pdf_consensus_evaluator/pipeline.py](pdf_consensus_evaluator/pipeline.py)
- Comb Line Filtering Utility: [utils/comb_filter.py](utils/comb_filter.py)
- Sample PDF: [samples/healthcare_patient_intake_form.pdf](samples/healthcare_patient_intake_form.pdf)
- Sample Rubric Spec: [samples/rubric_spec_healthcare_patient_intake_form.json](samples/rubric_spec_healthcare_patient_intake_form.json)
