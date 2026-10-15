---
title: Music Annotation Project
emoji: 🎼
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🎼 Music Annotation Project

A modular sheet-music annotation platform for:

- Page alignment
- Handwriting detection
- Human review
- Immutable dataset publication
- YOLO segmentation training
- Structural pilot review workflows

## Applications

### Main Application

Entry point:

```text
app.py
```

Functions:

- Align pages
- Generate draft masks
- Build dataset releases
- Validate releases
- Train YOLO models

### Pilot Reviewer

Entry point:

```text
pilot_app.py
```

Functions:

- Load frozen V7.7F policy bundle
- Run mandatory regression gate
- Generate structural correction proposals
- Human-in-the-loop review workflow
- Final approval workflow

⚠️ The Pilot Reviewer is intentionally separated from the main application because it performs expensive validation and policy checks.

## Workflow

### Main Pipeline

```text
Raw Pages
    ↓
Alignment Pipeline
    ↓
Draft Generation
    ↓
Human Review
    ↓
Dataset Builder
    ↓
YOLO Training
```

### Pilot Workflow

```text
Incoming Pages
    ↓
Policy Bundle
    ↓
Regression Gate
    ↓
Structural Analysis
    ↓
Reviewer UI
```

## Project Structure

```text
app.py
pilot_app.py

modules/
│
├── alignment/
│   ├── alignment_utils.py
│   ├── registration.py
│   ├── auto_mask.py
│   ├── alignment_state.py
│   └── alignment_pipeline.py
│
├── hitl_generator.py
├── confidence_maps.py
├── yolo_inference.py
│
├── dataset_builder.py
├── dataset_release.py
├── dataset_stager.py
│
├── trainer.py
├── gst_evaluator.py
├── run_exporter.py
│
└── pilot/
    ├── structural.py
    ├── policy_bundle.py
    ├── regression_gate.py
    ├── hitl_resolution.py
    ├── proposal_builder.py
    ├── decision_manager.py
    ├── reviewer_backend.py
    ├── reviewer_ui.py
    └── pilot_reviewer.py
```

## Deployment

### Option 1 (Recommended)

Two Hugging Face Spaces:

```text
Space 1
└── app.py

Space 2
└── pilot_app.py
```

### Option 2

Run locally:

```bash
python app.py
```

or

```bash
python pilot_app.py
```

## Status

### Alignment

✅ alignment_utils.py

✅ registration.py

✅ auto_mask.py

✅ alignment_state.py

✅ alignment_pipeline.py

### Draft Generation

✅ hitl_generator.py

✅ confidence_maps.py

✅ yolo_inference.py

### Dataset Publishing

✅ dataset_builder.py

✅ dataset_release.py

✅ dataset_stager.py

### Training

✅ trainer.py

✅ gst_evaluator.py

✅ run_exporter.py

### Pilot Reviewer

✅ structural.py

✅ policy_bundle.py

✅ regression_gate.py

✅ hitl_resolution.py

✅ proposal_builder.py

✅ decision_manager.py

✅ reviewer_backend.py

✅ reviewer_ui.py
