---
title: Music Annotation HF-04.2
emoji: 🎼
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# HF-04.2
Docker/FastAPI and the HF-04.0 mobile mask editor are retained. The backend recreates canonical V7.8 geometry_t80 from the reviewed undilated seed mask, then invokes the frozen Notebook 91 A5 primitives in this exact order: load_bundle, make_padded, score_coordinates, decisions, assemble. Copy the complete unmodified Notebook 91 bundle to frozen/a5_v1/. Generated __pycache__ and .pyc files are excluded from hash verification.

## HF-04.3 additions

HF-04.3 makes the Ardross A5 frozen replay a mandatory startup test and adds a separate diagnostic for comparing HF-generated V7.8 channels with the historical replay arrays. See `docs/HF043.md`.
