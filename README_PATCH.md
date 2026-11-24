# HF-02.5 exact Notebook 07 canvas patch

Replace the HF-02.4 root `app.py` and `music_annotation/notebook07_canvas.py`. Keep the rest of HF-02.3/02.4 and the frozen artefacts.

This patch extracts `CANVAS_HTML`, `CANVAS_JS` and `CSS` from the supplied Notebook 07 V12 export. Intentional changes are limited to: 1200px review packaging supplied by the existing backend, Authorise instead of Save+next, and direct invocation of the notebook's exported `window.hitlIngest(raw)` after YOLO generation. The original off-screen action bridge and native value setter are retained for returning ADD/ERASE masks.
