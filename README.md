---
title: Music Annotation Removal Assistant HF-02
emoji: 🎼
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
---
# HF-02 supervised Space
HF-01 is frozen as the first working deployment milestone. HF-02 adds mask review before canonical V7.8, verifies and applies the frozen V7.7F policy, and requires component-level human decisions plus final approval.

## Safety boundary
No automatic acceptance. The reviewed mask is the intervention boundary. Pixels outside the prepared mask remain unchanged.
