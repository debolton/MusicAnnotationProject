# HF-02.2 patch
Overlay this patch on the current HF-02 repository.

Workflow:
1. Upload only.
2. Generate YOLO draft server-side.
3. Automatically navigate to one lightweight canvas.
4. Switch one canvas between Mask, Confidence and Original.
5. Pan / ADD / ERASE / Undo / Redo / Reset.
6. Authorise `(draft OR ADD) AND NOT ERASE`.
7. Automatically navigate to V7.8/V7.7F component review.

The upload tab returns no overlay or heatmap. Full-resolution arrays remain in `runtime_sessions`; the browser receives one preview and compact stroke JSON.
