---
name: pdf-extraction-consensus-evaluator
description: >-
  Extracts target data fields from PDF documents into structured JSON and validates each field using a parallel panel of five thinking LLM judges against a verification rubric, producing field-level consensus scores, routing decisions, audit trails, and interactive UI dashboards.
---

# Multi-Agent PDF Extraction and Consensus Evaluator

Use this skill when processing PDF documents (such as intake forms, identity documents, invoices, or medical claims) that require verified extraction with provenance and consensus scoring across an ensemble of independent thinking LLM judges.

---

## Workflow Overview

The skill coordinates a three-stage pipeline:

1. **Stage 1: Primary Multimodal Extraction**
   - High-throughput multimodal model (`gemini-3.8-flash`) parses the PDF document with thinking tokens disabled (`thinking_budget: 0`).
   - Generates an initial candidate JSON object adhering to the target field schema.

2. **Stage 2: Parallel 5x Thinking Judge Panel**
   - Spawns 5 concurrent, isolated instances of `gemini-3.6-flash` with thinking budget enabled (`thinking_budget: 2048+`).
   - Each judge independently cross-references the candidate extraction against both the raw visual PDF and the verification rubric (syntactic rules and semantic grounding criteria).
   - Each judge returns verdicts, specific failure modes, page citations, and proposed corrections.

3. **Stage 3: Consensus Aggregation and Routing**
   - Deterministic post-processing computes field agreement ratios (k / 5 passes).
   - Classifies fields into confidence tiers:
     - 5/5: `UNANIMOUS_PASS` (Auto-Accept)
     - 4/5: `MAJORITY_PASS` (Auto-Accept with Warning)
     - 3/5: `CONTESTED` (Route to Human-in-the-Loop)
     - <= 2/5: `REJECTED` (Flag for Resubmission / Rerun)
   - Emits accepted data, exception reports with dissent justifications, an audit trail, and an interactive UI Dashboard.

---

## UI Dashboard Support

The skill generates a self-contained interactive UI dashboard (`ui/index.html`) upon executing the pipeline for the given input PDF and rubric specification. The dashboard renders inside browser and iframe environments.

Features (Organized across 5 tabs from left to right):
1. **PDF & Rubric Inspector**: Visual multi-page PDF viewer with page navigation and zoom alongside rubric specifications.
2. **Primary Extraction**: Candidate extraction viewer with thinking token diagnostics and model parameters.
3. **Judge Panel Matrix**: 5-column parallel judge grid displaying reasoning tokens, field verdicts, and citations.
4. **Consensus & Routing**: Summary statistics cards, field agreement ratios (k / 5), confidence bands, and validated output payload.
5. **Provenance & Audit Trail**: Complete JSON audit trail viewer with one-click clipboard copy and report download.
- Dark and Light mode toggle button with persistent preferences.

---

## Agent Instructions for Executing This Skill

When a user requests PDF extraction and verification:

### Step 1: Authentication & Environment
The pipeline natively supports both Google Cloud Vertex AI and Google AI Studio:
- **Google Cloud (Default / Recommended)**: If the user is authenticated via Google Cloud (`gcloud auth application-default login` or active `gcloud` session), the pipeline automatically resolves credentials and routes requests through Vertex AI. **Do NOT prompt the user for an API key** if Google Cloud ADC or `gcloud` is available.
- **Google AI Studio (Alternative)**: If `GEMINI_API_KEY` is present in the environment or passed via `--api-key`, the client will use the AI Studio endpoint.

### Step 2: Verify Input Files
Check that the required input files exist:
- Target PDF document (e.g., [samples/healthcare_patient_intake_form.pdf](samples/healthcare_patient_intake_form.pdf) or [samples/cancer_screening_lab_report.pdf](samples/cancer_screening_lab_report.pdf))
- Target Rubric Specification JSON (e.g., [samples/rubric_spec_healthcare_patient_intake_form.json](samples/rubric_spec_healthcare_patient_intake_form.json) or [samples/rubric_spec_cancer_screening_lab_report.json](samples/rubric_spec_cancer_screening_lab_report.json))

If a custom rubric is not supplied, inspect the document and construct a rubric specification adhering to the format defined in the samples.

### Step 3: Execute the Pipeline and Generate Dashboard
Run the pipeline runner script specifying both JSON output and UI dashboard output:

```bash
python3 scripts/run_pipeline.py \
  --pdf <PATH_TO_PDF> \
  --rubric <PATH_TO_RUBRIC_JSON> \
  --output <PATH_TO_OUTPUT_JSON> \
  --dashboard <PATH_TO_DASHBOARD_HTML>
```

Example using the sample files:

```bash
python3 scripts/run_pipeline.py \
  --pdf samples/healthcare_patient_intake_form.pdf \
  --rubric samples/rubric_spec_healthcare_patient_intake_form.json \
  --output output/extraction_report.json \
  --dashboard ui/index.html
```

If evaluating an already extracted candidate JSON, use the candidate evaluation script:

```bash
python3 scripts/evaluate_candidate.py \
  --pdf <PATH_TO_PDF> \
  --rubric <PATH_TO_RUBRIC_JSON> \
  --candidate-json <PATH_TO_CANDIDATE_JSON> \
  --output <PATH_TO_OUTPUT_JSON> \
  --dashboard ui/index.html
```

### Step 4: Interpret Output and Present Dashboard
Read the generated output JSON and present a clear summary:
1. **Accepted Fields**: List all fields that attained unanimous or majority consensus along with their extracted values.
2. **Contested Fields (HITL)**: Highlight fields that received split verdicts (e.g., 3/5 passes). Present the candidate value, the specific dissent reasons from the judges, and any proposed corrections for user confirmation.
3. **Rejected Fields**: Highlight fields that failed verification (<= 2/5 passes), cite the detected failure modes (e.g., `FORMAT_MISMATCH`, `HALLUCINATION`, `FACILITY_ADDRESS_CONFUSED_AS_PATIENT`), and list the judge justifications.
4. **UI Dashboard Link**: Provide a link to the generated [ui/index.html](ui/index.html) dashboard.

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
