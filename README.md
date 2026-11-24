---
title: Music Annotation HF-03.0
emoji: 🎼
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---
# HF-03.0
Docker Space using FastAPI and a purpose-built mobile browser reviewer. Gradio is not used.

The underlying page and mask overlay are ordinary images. A small transparent canvas captures only the current stroke. Each completed stroke is sent immediately to Python, applied to the full-resolution authoritative mask, versioned, and then discarded from browser memory.
