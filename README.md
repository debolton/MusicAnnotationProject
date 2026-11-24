---
title: Music Annotation Removal Assistant
emoji: 🎼
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
---

# Music Annotation Removal Assistant

Hugging Face Space packaging for the Music Annotation Project.

## Current deployment boundary

This Space runs the **annotated-only YOLO draft-mask generator** followed by the **canonical V7.8 conservative correction proposal**. It does **not** silently promote the detector mask to an authoritative mask and does not claim automatic production approval.

For the full supervised workflow, the reviewer layer should remain enabled until the annotation-mask authority has separately passed its independent validation gates.

## Before first launch

Copy the production detector into:

`frozen/yolo_detector/best.pt`

See `docs/FROZEN_MODEL_TRANSFER_CHECKLIST.md` for the structural-policy and V8.2 artefacts needed when those layers are enabled.

## Hugging Face

The Space entry point is the root-level `app.py`. `README.md` supplies Space metadata using `app_file: app.py`.
