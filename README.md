# HF-02.6 iPad reviewer stabilisation patch
Replace `music_annotation/notebook07_canvas.py` only. Keep HF-02.5 `app.py` and all backend/frozen files.

Changes:
- initial load uses Fit page after two animation frames;
- canvas backing DPR is capped at 1.5;
- viewport height is bounded to 320-720px, eliminating the multi-scroll grey slab;
- ResizeObserver/visualViewport changes no longer create unbounded canvas dimensions;
- undo/redo stores vector brush operations, not two full PNG data URLs per stroke;
- long-press/context-menu/browser touch gestures are suppressed on the canvas;
- PAN, ADD and ERASE are explicit mutually exclusive modes with active-mode highlighting;
- single-finger drag draws only in ADD/ERASE and pans only in PAN.
