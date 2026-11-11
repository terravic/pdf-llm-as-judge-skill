# Product Requirements Document (PRD)

## Multi-Agent Document Extraction and Consensus Evaluator

- **Status**: Current Architecture Baseline
- **Document Version**: 2.0.0
- **Target Systems**: Document Processing Workflows, Multimodal Verification Pipelines, Clinical and Operational Intake Systems

---

## 1. Objective and Scope

Automated document extraction across clinical, financial, and operational domains requires structured validation to detect transcription errors, ungrounded values, field transpositions, and optical conflation between pre-printed form templates and handwritten ink strokes.

The Multi-Agent Document Extraction and Consensus Evaluator provides a three-stage extraction and verification pipeline paired with a standalone HTML inspection dashboard (`dashboard.html`):
1. Primary multimodal extraction across multi-page PDF documents and image scans (JPEG, PNG, WebP, HEIC).
2. Independent parallel evaluation across five thinking-enabled LLM judges executing syntactic checks, visual semantic grounding, and adversarial forensic comb-box audits.
3. Deterministic consensus scoring and routing based on field-level judge agreement ratios.
4. Self-contained interactive HTML inspection (`dashboard.html`) compliant with `DESIGN.md` specifications, pre-loaded with workspace evaluation data and reusable across new document runs.

---

## 2. Core Pipeline Architecture

The system executes three sequential, decoupled stages:

```
[Source Document (PDF / Image) + Verification Rubric Specification]
                               |
                               v
+--------------------------------------------------------------+
| Stage 1: Primary Multimodal Extractor                        |
| - Default Model: gemini-3.8-flash (configurable)             |
| - Configuration: thinking_budget = 0, temperature = 0.1      |
| - Full-document sweep across typed text, forms, and ink      |
| - Output: candidate_extraction JSON object                   |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| Stage 2: Parallel 5x Judge Panel + ROI Preprocessing         |
| - Default Model: 5x gemini-3.8-flash (asyncio.gather)        |
| - Configuration: thinking_budget >= 2048, temperature ~ 0.7  |
| - Preprocessing: Morphological tick suppression & ROI crop   |
| - Adversarial Forensic Audit: Blind ink tracing & overturns  |
| - Output: 5 independent JudgeReport structures               |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| Stage 3: Deterministic Consensus & Routing Engine            |
| - Input: JudgeReports + Candidate Extraction + RubricSpec    |
| - Deterministic vote aggregation (k / N passes)              |
| - Consensus classification: UNANIMOUS_PASS, MAJORITY_PASS,   |
|   CONTESTED, REJECTED                                        |
| - Output: PipelineReport JSON + Standalone dashboard.html    |
+--------------------------------------------------------------+
```

### 2.1 Stage 1: Primary Multimodal Extractor (`pdf_consensus_evaluator/stage1_extractor.py`)
- **Role**: High-throughput preliminary extraction pass.
- **Thinking Budget Constraint**: Configured with `thinking_budget: 0` so Stage 1 performs a fast multimodal sweep without consuming reasoning tokens.
- **Supported Formats**: Multi-page PDF documents (`.pdf`) and image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`, `.heif`, `.gif`).
- **Form and Handwriting Parsing**: Extracts vector text, tabular grids, checkboxes, strikethrough corrections, and handwritten entries while subtracting pre-printed template borders and baseline ticks.
- **Targeted Comb Crop Pass**: On image inputs with rubric fields marked `field_type: "segmented_comb_box"` and normalized `bounding_box` coordinates, crops and filters the region prior to slot-by-slot transcription.

### 2.2 Stage 2: Parallel Thinking Judge Panel (`pdf_consensus_evaluator/stage2_judge_panel.py`)
- **Role**: Multi-perspective syntactic and visual grounding verification.
- **Thinking Budget Constraint**: Configured with `thinking_budget: 2048` by default (configurable via `--thinking-budget`).
- **Parallel Execution**: Dispatches `N` (default `5`) concurrent, isolated judge evaluations via `asyncio.gather` with staggered temperatures (`0.60` to `0.80`) for sample diversity and per-judge execution timeouts (`90.0s` default).
- **Verification Checks**:
  - *Syntactic Check*: Validates data types, required presence, minimum token counts, masking rules, expected subfields, and regular expression patterns.
  - *Semantic Grounding Check*: Cross-references extracted values against visual source sections, page anchors, checkbox states, and handwritten strokes.
- **Adversarial Forensic Protocol**: On segmented comb boxes or tick-marked fields, judges execute a four-step falsification sequence:
  1. *Blind Independent Tracing*: Traces ink strokes prior to anchoring on the candidate value.
  2. *Grid Alignment and Tick Mapping*: Identifies periodic pre-printed cell borders and baseline ticks.
  3. *Conflation Falsification*: Tests whether characters such as `'4'`, `'a'`, `'d'`, `'H'`, or `'8'` result from `'C'`, `'o'`, `'1'`, or `'3'` touching a template line.
  4. *Overturn and Correction*: Sets `verdict: "FAIL"`, assigns `failure_mode: "TEMPLATE_INK_CONFLATION"`, documents the falsification, and supplies `proposed_correction`.

### 2.3 Stage 3: Deterministic Consensus and Routing Engine (`pdf_consensus_evaluator/stage3_consensus.py`)
- **Role**: Mathematical vote aggregation without secondary model calls.
- **Routing Decision Matrix (5-Judge Quorum)**:

| Agreement Ratio (k / 5) | Consensus Tier | Confidence Band | Operational Routing Action |
| :--- | :--- | :--- | :--- |
| 5 / 5 | `UNANIMOUS_PASS` | Very High Confidence (>=98%) | `Auto-Accept` |
| 4 / 5 | `MAJORITY_PASS` | High Confidence (90-95%) | `Auto-Accept with Warning` |
| 3 / 5 | `CONTESTED` | Split Verdict (approx 60%) | `Route to Human-in-the-Loop (HITL)` |
| <= 2 / 5 | `REJECTED` | Low Confidence / Failure | `Flag for Extraction Failure / Resubmission` |

- **Degraded Quorum Scaling**: Dynamically scales pass ratios (`min_pass_ratio = 0.8`, `contested_min_ratio = 0.5`) when individual judges time out or return transport errors.
- **Phantom Dissent Reconciliation**: Normalizes semantically identical formatting variations (such as `'2/3/1944'` versus `'02/03/1944'`) when a judge marks `FAIL` while proposing an equivalent value.

---

## 3. Region-of-Interest Cropping and Morphological Preprocessing

### 3.1 Optical Conflation in Segmented Comb Boxes
When full-page scans containing small character comb boxes (`<1.5%` of page area) are downsampled during multimodal ingestion, thin pre-printed baseline ticks and vertical dividers merge visually with adjacent pen strokes:
- Open curved `'C'` or `'c'` touching a baseline tick is misread as `'4'`.
- Oval `'o'` or `'0'` touching a vertical divider is misread as `'a'`, `'d'`, or `'q'`.
- Vertical strokes `'1'` or `'I'` adjacent to cell walls are misread as `'H'`.

### 3.2 Preprocessing Engine (`utils/comb_filter.py`)
1. **Region-of-Interest Cropping (`crop_field_roi`)**:
   - Extracts normalized `[ymin, xmin, ymax, xmax]` coordinates directly from native-resolution images with contextual boundary padding (`15px` default or `3%` ratio) so ascenders and descenders are preserved.
2. **Morphological Vertical Tick Suppression (`suppress_vertical_ticks`)**:
   - Distinguishes thin linear template ticks (`<= 2px` width) from thicker curved pen strokes (`>= 3px`) using morphological opening kernels (OpenCV when installed, with a vectorized NumPy + Pillow fallback) and bleaches isolated vertical ticks to white (`255`).
3. **Color Space Comb Filtering (`suppress_comb_lines`)**:
   - Separates saturated colored pen inks in HSV space (`s > 28`, `v < 225`) from low-saturation gray template lines before applying vertical tick suppression.
4. **Automated Comb Localization (`auto_detect_comb_rois`)**:
   - Computes horizontal and vertical pixel density projections to locate candidate comb-box regions when explicit `bounding_box` coordinates are omitted from the rubric.

---

## 4. Data Contracts and Failure Taxonomy

### 4.1 Rubric Specification Schema (`RubricSpec`)
```json
{
  "rubric_version": "2.0",
  "document_type": "Healthcare_Patient_Intake_Form",
  "extraction_criteria": [
    {
      "field_name": "insurance_member_id",
      "field_type": "segmented_comb_box",
      "bounding_box": [0.12, 0.45, 0.16, 0.88],
      "syntactic_rules": {
        "required": true,
        "type": "string",
        "min_tokens": 1,
        "format_regex": "^[A-Z]{3}-\\d{9}-\\d{2}$",
        "masking_allowed": false,
        "expected_subfields": null
      },
      "semantic_grounding_rules": {
        "source_section": "SECTION 2: COVERAGE & INSURANCE",
        "verification_criteria": "Must match the primary subscriber member identifier verbatim.",
        "failure_modes": [
          "TEMPLATE_INK_CONFLATION",
          "DIGIT_TRANSPOSITION",
          "FORMAT_MISMATCH"
        ]
      }
    }
  ]
}
```

### 4.2 Failure Mode Taxonomy (`FailureMode`)
- `NONE`: Syntactic and visual grounding checks passed.
- `FORMAT_MISMATCH`: Value violates type, regex, masking, or token constraints.
- `MISSING_VALUE`: Required field omitted or null.
- `HALLUCINATION`: Extracted value absent from the source document.
- `WRONG_ENTITY`: Value extracted from an unrelated person, clinician, or organization.
- `FACILITY_ADDRESS_CONFUSED_AS_PATIENT`: Clinic or facility address extracted instead of patient residential address.
- `DIGIT_TRANSPOSITION`: Characters or digits inverted relative to the source document.
- `SECONDARY_CODE_EXTRACTED_AS_PRIMARY`: Secondary diagnosis or origin extracted as primary.
- `PREFERRED_NAME_CONFUSED_WITH_LEGAL`: Preferred nickname extracted instead of legal name.
- `TEMPLATE_INK_CONFLATION`: Pre-printed comb divider or tick mark conflated with handwritten pen ink.
- `ILLEGIBLE_HANDWRITING`: Ambiguous handwriting that cannot be resolved from visual inspection.
- `CHECKBOX_MISINTERPRETED`: Selected vs. unselected checkbox or radio control misread.
- `HANDWRITTEN_CORRECTION_IGNORED`: Crossed-out entry extracted instead of adjacent handwritten correction.
- `UNREADABLE_SOURCE`: Degraded scan or complete judge quorum failure.

---

## 5. Interactive Dashboard Requirements (`DESIGN.md` Alignment)

### 5.1 Singular Output and Dynamic Runtime (`dashboard.html`)
- **Single Root Artifact**: `pdf_consensus_evaluator/dashboard_generator.py` emits a singular self-contained `dashboard.html` at the repository root (unless overridden via `--dashboard`), pre-loaded with empirical workspace evaluation data.
- **Multi-Document Usability**: Supports evaluating and inspecting new documents without code changes through:
  1. CLI regeneration (`scripts/run_pipeline.py` and `scripts/evaluate_candidate.py`).
  2. In-browser `Load Run / PDF` file picker and window-wide drag-and-drop (`FileReader`) for `.json` reports, `.pdf` files, and image scans.
  3. Programmatic state updates via `window.updateDashboardState(bundle)`, `?report=<url>` query loading, and live HTTP Server-Sent Events (`/events`) that preserve scroll position, active tab, filter queries, and node-map pan/zoom state.

### 5.2 Visual Hierarchy, Theme Controls, and Color Tokens
- **Typography**: Pairs `Instrument Serif` for display titles, `DM Sans` for UI labels and prose, and `IBM Plex Mono` (`font-variant-numeric: tabular-nums`) for all numerical metrics, field keys, verdicts, and code payloads.
- **Flat Bento Architecture**: Uses single-level bento panes with solid background tokens and `1px solid` borders (`6px` to `8px` radius) without CSS color gradients or nested card containers.
- **Segmented Theme Selector (`#theme-toggle`)**: Provides Light (`#theme-btn-light`), Dark (`#theme-btn-dark`), and System (`#theme-btn-system`) mode buttons with `localStorage` persistence, live `window.matchMedia('(prefers-color-scheme: dark)')` tracking, and host environment fallback (`system_theme`).
- **Surface Tokens**:
  - Light Mode: `#FAFAF9` page background, `#FFFFFF` pane surface, `#E7E5E4` border, `#1C1917` primary text, `#787774` secondary text.
  - Dark Mode: `#0D0E11` page background, `#16171D` pane surface, `#292A30` border, `#ECE8E2` primary text, `#9A9895` secondary text.
- **Data Visualization Palette**: `#0D9488` (Vibrant Teal), `#D97706` (Energetic Amber), `#4F46E5` (Vivid Indigo), `#059669` (Functional Emerald), and `#E11D48` (Deep Rose) rendered on transparent plot surfaces with `#E7E5E4` / `#292A30` axis gridlines.

### 5.3 Model Configuration and Command Builder
- Provides an interactive configuration bar inside `dashboard.html` while keeping the file strictly static (no local backend server required):
  - Editable text inputs for **Stage 1 Extractor Model** (`#model-extractor-input`) and **Stage 2 Judge Model** (`#model-judge-input`) backed by `<datalist id="available-models-datalist">` populated at generation time via `_discover_available_models()` (with offline fallback when network access or credentials are unavailable), while accepting any custom future model identifier.
  - Editable numeric inputs for **Judge Thinking Budget** (`#model-thinking-input`) and **Judge Quorum Count** (`#model-judges-input`).
  - Live command preview (`#model-cmd-preview`) with copy buttons for `/pdf-llm-as-judge` agent prompts (`#copy-agent-cmd-btn`) and `python3 scripts/run_pipeline.py` CLI invocations (`#copy-cli-cmd-btn`).

### 5.4 Workspace Tabs
1. **Tab 1 (`1. Document & Rubric`)**: Side-by-side multi-page PDF viewer (serialized `PDF.js` page rendering with `<iframe>` fallback, page stepping, and zoom controls) or image viewer alongside the searchable `Verification Rubric Specification` table.
2. **Tab 2 (`2. Primary Extraction`)**: Stage 1 candidate JSON payload, model/thinking-budget diagnostics, flat `Extracted Field Breakdown` table (`#extracted-fields-table`), and 2D `Extracted Field Completeness & Structure` chart.
3. **Tab 3 (`3. Judge Panel Matrix`)**: `Judge x Field Verification Heatmap`, 2D `Judge Execution Latency & Pass Rate` chart, and 5-column `Parallel LLM Judge Evaluation Matrix` with click-to-inspect reasoning justifications and proposed corrections.
4. **Tab 4 (`4. Consensus & Routing`)**: 2D `Field Agreement Ratios` chart, pan/zoom SVG `Interactive Relationship Node Map`, filterable `Field-Level Consensus & Routing Table`, `Validated Clean Output Payload`, and `Exceptions & Dissent Ledger`.
5. **Tab 5 (`5. Provenance & Audit`)**: `Provenance Audit Trail` table and full structured `PipelineReport` JSON viewer with clipboard copy and JSON download controls.

---

## 6. Verification and Quality Assurance

1. **Automated Test Suite (`tests/`)**:
   - `tests/test_models.py`: Validates rubric deserialization, syntactic rule defaults, and judge report structures.
   - `tests/test_consensus.py`: Verifies vote math across all agreement ratios (`5/5` through `0/5`), degraded quorums, and phantom dissent reconciliation.
   - `tests/test_comb_filter.py`: Validates normalized ROI cropping, vertical tick suppression across monochrome and colored inks, comb auto-detection, and Stage 1/Stage 2 comb crop intercepts.
   - `tests/test_stage1_and_judge_prompts.py`: Verifies Stage 1 template subtraction instructions, Stage 2 Adversarial Forensic Protocol prompts, and failure mode enumerations.
   - `tests/test_synthetic_discrepancies.py`: Injects synthetic SSN format errors, clinic-vs-patient address confusions, and date transpositions to verify rejection and dissent attribution.
   - `tests/test_pipeline_e2e.py`: Executes end-to-end pipeline runs across vector PDFs, scanned/degraded PDFs, and JPEG images, and verifies `DESIGN.md` typography, color tokens, segmented theme controls, tabular extraction views, dynamic model discovery, and offline fallback behavior.

2. **Operational Latency Targets**:
   - Stage 1 primary extraction: single multimodal request with `thinking_budget = 0`.
   - Stage 2 parallel judge panel: concurrent execution bounded by `90.0s` per-judge timeout.
   - Stage 3 consensus aggregation and `dashboard.html` generation: deterministic local execution in `< 100ms`.
