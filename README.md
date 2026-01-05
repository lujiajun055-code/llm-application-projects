# LLM Application Projects (Internship)

Production-oriented LLM/VLM application projects focused on **document understanding and OCR automation**:
Image/PDF → structured JSON → normalisation → field-level evaluation.

> Note: This repository is desensitised and does not include confidential customer data.

## What’s Included
- **VLM-based extraction pipeline** (image/PDF ingestion, structured JSON output, robustness against truncated outputs)
- **Schema normalisation** across different JSON formats
- **Field-level accuracy evaluation** with mismatch/missing/extra breakdown
- **Debug utilities** for comparing GT vs prediction consistency
- **Template-based grid support** for stable table layouts

## Project: OCR Production Plan Extraction (Current)
Goal: Extract production plan tables into a consistent `production_plan` JSON for downstream use and evaluation.

Key components:
- `Qwen3.py` — model calling pipeline + dense/sparse fallback strategy
- `convert_pred.py` — convert multiple JSON shapes into a unified `production_plan`
- `ocr_accuracy_calculator.py` — field-level accuracy calculation and detailed error reporting
- `debug_diff.py` — quick structural diff checks between GT and predictions
- `cell_grid_template.py` — template-driven cell bbox generator for stable layouts

## Quick Start
### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt  # if you have one

