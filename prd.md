# Product Requirements Document (PRD)

## Multi-Agent Document Extraction and Consensus Evaluator

- **Status**: Current Architecture Baseline
- **Document Version**: 1.2.0
- **Target Systems**: Document Processing Workflows, Multimodal AI Systems, Clinical and Form Data Pipelines

---

## 1. Executive Summary & Objective

Automated document extraction in regulated, operational, and clinical domains demands high extraction fidelity. Traditional single-pass extraction systems and standard optical character recognition (OCR) engines frequently suffer from silent transcription errors, hallucinated field values, field transpositions, and optical conflation between pre-printed form templates and handwritten strokes.

The Multi-Agent Document Extraction and Consensus Evaluator implements a multi-stage validation framework:
1. Primary extraction across complex multi-page PDF documents and high-resolution image scans (e.g., JPEG, PNG, WebP).
2. Independent parallel evaluation across a panel of five thinking-enabled LLM judges executing adversarial forensic validation protocols.
3. Deterministic mathematical consensus scoring and routing based on field-level agreement ratios.
4. An interactive HTML5 inspection dashboard providing auditability and provenance visualization.

---

## 2. Core Architectural Design

The pipeline consists of three sequential, decoupled stages:

```
[Source Document (PDF / Image) + Target Rubric Specification]
                              |
                              v
+--------------------------------------------------------------+
| Stage 1: Primary Multimodal Extractor                        |
| - Engine: Gemini 3.8 Flash                                   |
| - Configuration: thinking_budget = 0, temperature = 0.1      |
| - Fast full-document sweep across typed text, forms & ink    |
| - Output: candidate_extraction.json                          |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| Stage 2: Parallel 5x Judge Panel + Image Preprocessing       |
| - Engine: 5x Gemini 3.8 Flash (Concurrent Asyncio Tasks)     |
| - Configuration: thinking_budget >= 2048, temperature = 0.7  |
| - Preprocessing: Morphological tick suppression & ROI crop   |
| - Adversarial Forensic Audit: Blind ink tracing & overturns  |
| - Output: 5 independent JudgeReport structures               |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| Stage 3: Deterministic Consensus & Routing Engine            |
| - Input: 5x JudgeReports + Candidate Extraction              |
| - Deterministic vote aggregation (k / 5 passes)              |
| - Consensus classification: UNANIMOUS, MAJORITY, CONTESTED,  |
|   REJECTED                                                   |
| - Output: PipelineReport JSON + Interactive Dashboard HTML   |
+--------------------------------------------------------------+
```

### 2.1 Stage 1: Primary Multimodal Extractor
- **Role**: High-throughput preliminary extraction pass.
- **Thinking Budget Constraint**: Strictly configured with `thinking_budget: 0`. Stage 1 performs a fast multimodal sweep without consuming reasoning tokens.
- **Input Scope**: Multi-page PDF documents and raw image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`).
- **Processing Capabilities**: Parses vector text, visual form layouts, table matrices, checkboxes, and handwritten ink.
- **Schema Grounding**: Converts extraction criteria in the verification rubric into structured JSON generation constraints.

### 2.2 Stage 2: Parallel 5x Thinking Judge Panel
- **Role**: Rigorous, multi-perspective adversarial evaluation and visual verification.
- **Thinking Budget Constraint**: Configured with `thinking_budget: 2048` or higher to enable chain-of-thought verification.
- **Model Standard**: `gemini-3.8-flash`.
- **Parallel Execution**: Dispatches 5 concurrent, isolated judge evaluations using non-blocking asynchronous tasks (`asyncio.gather`), with independent random seeds/temperatures to ensure sample diversity.
- **Dual Verification Criteria**:
  - *Syntactic Verification*: Type matching, regex format constraints, token counts, and required field completeness.
  - *Semantic Grounding*: Direct visual cross-referencing against the source document pages to eliminate hallucinations, address mix-ups, and field transpositions.
- **Adversarial Forensic Protocol**: On fields with segmented comb boxes or baseline tick marks, judges execute a four-step falsification protocol:
  1. *Blind Independent Tracing*: Traces ink strokes in the visual crop prior to inspecting candidate values to avoid anchoring bias.
  2. *Grid Alignment & Tick Mapping*: Maps periodic intervals of pre-printed template borders and baseline ticks.
  3. *Conflation Falsification*: Directly tests for common optical conflations (e.g., curved `'C'` touching a tick mark misread as `'4'`; oval `'o'` touching a divider misread as `'a'`).
  4. *Adversarial Overturn*: Overturns conflated candidate characters with `failure_mode: "TEMPLATE_INK_CONFLATION"` and outputs the true ink transcription.

### 2.3 Stage 3: Deterministic Consensus & Routing Engine
- **Role**: Purely mathematical aggregation without secondary model calls.
- **Routing Decision Matrix**:

| Agreement Ratio (k / 5) | Consensus Tier | Operational Routing Action |
| :--- | :--- | :--- |
| 5 / 5 | `UNANIMOUS_PASS` | Automated Acceptance |
| 4 / 5 | `MAJORITY_PASS` | Automated Acceptance with Logged Dissent |
| 3 / 5 | `CONTESTED` | Route to Human-in-the-Loop (HITL) Queue |
| <= 2 / 5 | `REJECTED` | Mark as Extraction Failure / Flag for Resubmission |

- **Degraded Quorum Support**: Dynamically adjusts agreement thresholds when a judge times out or encounters network degradation (e.g., 4/4 unanimous, 3/4 majority).
- **Phantom Dissent Reconciliation**: Automatically reconciles minor formatting equivalents (e.g., date normalization from `'2 3 1944'` to `'02/03/1944'`) where semantic intent is verified.

---

## 3. Surgical ROI Cropping & Morphological Preprocessing

### 3.1 Problem Statement: Optical Conflation in Pre-Printed Comb Boxes
When high-resolution documents containing character comb boxes (occupying <1.5% of total page area) are downsampled for multimodal ingestion, rigid pre-printed baseline tick marks and dividers blend with curved handwritten pen strokes. This produces recurrent transcription errors:
- Open curved `'C'` or `'c'` touching a baseline tick is misclassified as `'4'`.
- Oval `'o'` or `'0'` touching a vertical divider is misclassified as `'a'`, `'d'`, or `'q'`.
- Isolated vertical strokes `'1'` or `'I'` touching dividers are misclassified as `'H'`.

### 3.2 Preprocessing Engine Architecture (`utils/comb_filter.py`)
1. **High-Resolution Surgical Cropping (`crop_field_roi`)**:
   - Isolates the bounding box `[ymin, xmin, ymax, xmax]` directly from native resolution source scans.
   - Applies 3% contextual boundary padding (minimum 15 pixels) to prevent boundary clipping of ascenders, descenders, and character loops.

2. **Morphological Vertical Line Suppression (`suppress_vertical_ticks`)**:
   - Analyzes stroke geometry: pre-printed tick marks are strictly linear and $\le 2\text{px}$ wide, whereas handwritten ballpoint/gel pen strokes exhibit thickness $\ge 3\text{px}$ and continuous curvature.
   - Applies morphological vertical structuring kernels and vectorized column run-length suppression.
   - Bleaches isolated thin vertical ticks to background white (`255`) while preserving curved and thick pen strokes for both black ballpoint ink and colored ink.

3. **Color Space Filtering (`suppress_comb_lines`)**:
   - For colored pen inks (blue, violet), isolates low-saturation template gray lines ($s < 28$, $v > 45$) in HSV color space and removes them, preserving ink saturation channels.

4. **Automated Comb Box Localization (`auto_detect_comb_rois`)**:
   - Computes horizontal and vertical pixel projection profiles across image scan margins.
   - Automatically identifies repeating periodic comb cell geometries when explicit bounding boxes are omitted from the rubric.

---

## 4. Data Models & Interface Contracts

### 4.1 Rubric Specification Schema (`RubricSpec`)
```json
{
  "document_type": "string",
  "rubric_version": "string",
  "target_schema": {
    "field_name": "string",
    "type": "string | integer | number | boolean | array"
  },
  "extraction_criteria": [
    {
      "field_name": "string",
      "field_type": "standard | segmented_comb_box",
      "bounding_box": [0.12, 0.45, 0.16, 0.88],
      "syntactic_rules": {
        "required": true,
        "type": "string",
        "min_tokens": 1,
        "format_regex": "^[A-Z0-9]{6,10}$",
        "masking_allowed": false
      },
      "semantic_grounding": {
        "page_anchor": "string",
        "visual_cues": ["string"],
        "critical_distinction": "string"
      }
    }
  ]
}
```

### 4.2 Failure Mode Taxonomy
- `NONE`: Verification successful.
- `FORMAT_MISMATCH`: Fails regex, type, or token count requirements.
- `MISSING_VALUE`: Required field omitted from extraction.
- `HALLUCINATION`: Extracted value does not exist in the source document.
- `WRONG_ENTITY`: Extracted data from an adjacent or unrelated entity.
- `FACILITY_ADDRESS_CONFUSED_AS_PATIENT`: Specific address transposition.
- `DIGIT_TRANSPOSITION`: Numerical order inverted.
- `TEMPLATE_INK_CONFLATION`: Character identity contaminated by pre-printed comb dividers or ticks.
- `ILLEGIBLE_HANDWRITING`: Visual stroke ambiguity unresolvable by forensic inspection.
- `CHECKBOX_MISINTERPRETED`: Incorrect binary or categorical checkbox selection.

---

## 5. User Interface & Dashboard Requirements

The system outputs a self-contained HTML5 single-page application (`ui/index.html`):
- **Tab 1: Document & Rubric Inspector**: Side-by-side view with interactive PDF/image rendering (zoom, pan, page navigation) and rubric rule inspection.
- **Tab 2: Primary Extraction**: Displays Stage 1 extraction results, execution duration, and model parameters.
- **Tab 3: Judge Panel Matrix**: 5-column grid detailing individual judge verdicts, reasoning token diagnostics, visual citations, and proposed corrections.
- **Tab 4: Consensus & Routing**: Visual summary metrics, field confidence tiers, agreement ratios, and sanitized verified payload.
- **Tab 5: Provenance & Audit Trail**: Full structured JSON audit record with clipboard copy and export options.
- **Styling**: Clean CSS layout supporting both Light and Dark display themes with local storage state persistence. Zero external JavaScript runtime build requirements.

---

## 6. Verification & Quality Assurance Standards

1. **Automated Test Coverage**:
   - `tests/test_consensus.py`: Verifies consensus math, tie-breaking, degraded quorums, and phantom dissent resolution.
   - `tests/test_comb_filter.py`: Validates bounding box cropping, morphological vertical line suppression for black and colored inks, comb auto-detection, and judge adversarial overturning.
   - `tests/test_stage1_and_judge_prompts.py`: Verifies prompt construction, thinking budget settings, and failure mode taxonomies.
   - `tests/test_synthetic_discrepancies.py`: Verifies rejection of injected invalid SSNs, clinic address confusions, and date transposition errors.
   - `tests/test_pipeline_e2e.py`: Executes end-to-end processing across digital vector PDFs, degraded image scans, and synthetic medical records.

2. **Performance Constraints**:
   - Stage 1 execution: low latency primary sweep.
   - Stage 2 execution: concurrent parallel execution within 90-second timeout.
   - Post-processing consensus: deterministic execution in < 100ms.
