---
title: Music Annotation Control Centre
emoji: 🎼🎺🎶
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Music Annotation Control Centre

Version 1 is a html launcher for the existing Google Colab workflow. It does not execute training inside GitHub or inspect private Google Drive contents.

## Workflow

1. Prepare Existing Pages.
2. Generate HITL Drafts.
3. Review Draft Masks.
4. Build Training Dataset.
5. Run Dataset Forensics.
6. Train Model.
7. Evaluate Models.

Page normalisation is an optional experiment.

## Existing Drive assumptions

The notebooks use `/content/drive/MyDrive/MusicAnnotationProject` directly or read `project_root` from `/content/drive/MyDrive/MusicAnnotationProject/project_config.json`. Review each configuration cell before running.

## Security and size

The `.gitignore` excludes checkpoints, datasets, source pages, masks, predictions, confidence maps and pipeline state. Keep these in Google Drive or suitable private storage.
