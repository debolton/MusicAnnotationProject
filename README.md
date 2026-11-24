---
title: HF-02.9 Standard Gradio Diagnostic
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
---
# HF-02.9
Minimal iOS diagnostic using only `gr.Image` and `gr.ImageEditor`.

This proves, in order:
1. uploaded image reaches Python;
2. Python-generated NumPy images render through ordinary `gr.Image` outputs;
3. `gr.ImageEditor` renders on iOS;
4. the editor value returns to Python through an ordinary Gradio callback.

No custom HTML, JavaScript, hidden bridge, YOLO, V7.8, V7.7F, or frozen artefacts are required.
