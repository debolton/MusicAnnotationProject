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

A Gradio-based workflow for detecting handwritten annotations on sheet music, reviewing corrections, building immutable training datasets, and training improved YOLO segmentation models.

## Features

### Draft Generation

Generate:

- YOLO draft masks
- confidence maps
- draft audit records

for unreviewed pages.

### Review & Correction

Review detections and corrections using:

- human-in-the-loop workflows
- confidence-guided annotation review
- structural risk scoring
- component-level auditing

### Dataset Management

Build immutable dataset releases including:

- page objects
- manifests
- dataset versions
- publication metadata

### Model Training

Train segmentation models from published dataset releases.

Includes:

- release validation
- dataset staging
- YOLO training
- validation metrics
- GST evaluation
- model export

### Structural Pilot Reviewer

Includes:

- frozen V7.7F policy validation
- regression-gated execution
- geometry reconstruction verification
- decision preservation across edits
- supervised review workflows

---

## Application Workflow

```text
Upload Pages
      ↓
Generate Draft Masks
      ↓
Review & Correct
      ↓
Build Dataset Release
      ↓
Train YOLO Model
      ↓
Evaluate
      ↓
Deploy
```

---

## Project Structure

```text
app.py

modules/

├── hitl_generator.py
├── dataset_builder.py
├── dataset_release.py
├── dataset_stager.py
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

---

## Requirements

The application uses:

- Gradio
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLO
- Scikit-learn
- Joblib
- PyTorch

---

## Notes

This repository contains the refactored application architecture originally developed in a collection of Google Colab notebooks.

For Hugging Face deployment:

- avoid Google Drive dependencies
- use relative project paths
- keep large datasets and training artefacts outside the repository
- upload trained models separately when required

---

## Current Status

✅ Draft Generation

✅ Dataset Builder

✅ Dataset Publication

✅ Training Pipeline

✅ GST Evaluation

✅ Run Export

✅ Structural Pilot Reviewer

✅ Decision Preservation

🚧 Deployment Hardening

🚧 Production Model Registry
