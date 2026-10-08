---
title: Music Annotation Control Centre
emoji: 🎼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
---

# Music Annotation Control Centre

Version 1 is a Gradio launcher for the existing Google Colab workflow. It does not execute training inside the Hugging Face Space or inspect private Google Drive contents.

## Configure once

Edit `config/notebooks.json` and replace the placeholder GitHub and Hugging Face URLs. Do not put tokens, passwords or private Drive links in the file.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.

## Deploy

Create a Gradio Hugging Face Space, upload or synchronise this repository, edit the URLs in `config/notebooks.json`, and restart the Space. The launcher itself does not require a GPU. GPU selection happens in Colab for notebooks marked GPU.

## Workflow

1. Prepare Existing Pages.
2. Generate HITL Drafts.
3. Review Draft Masks.
4. Build Training Dataset.
5. Run Dataset Forensics.
6. Train Model.
7. Evaluate Models.

Page normalisation is an optional experiment.

## Notebook conversion

The eight supplied `.txt` files were valid nbformat 4 notebook JSON exports. Code and markdown cells were preserved as `.ipynb`. Stored outputs and user-specific execution metadata were cleared to avoid committing embedded results and execution identity data.

## Existing Drive assumptions

The notebooks use `/content/drive/MyDrive/MusicAnnotationProject` directly or read `project_root` from `/content/drive/MyDrive/MusicAnnotationProject/project_config.json`. Review each configuration cell before running.

## Security and size

The `.gitignore` excludes checkpoints, datasets, source pages, masks, predictions, confidence maps and pipeline state. Keep these in Google Drive or suitable private storage.

## Troubleshooting

- **Configure GitHub URL first:** replace the placeholder repository URL and restart.
- **Colab 404:** verify repository visibility, branch and notebook path.
- **GPU notebook stops:** select a GPU runtime in Colab, then run again.
- **Reviewer has no pages:** run the HITL draft generator and check matching source, draft and confidence-map files.
- **Space cannot see Drive:** expected in Version 1; Drive is mounted by the Colab notebooks.
