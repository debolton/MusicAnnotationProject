# HF-03.0 architecture
- FastAPI owns upload, YOLO, full-resolution mask state, revision history, canonical V7.8, optional frozen V7.7F scoring, decisions and packaging.
- The mobile front end is plain HTML/CSS/JavaScript.
- `<img>` renders the page and mask overlay.
- `<canvas>` captures only the active stroke; it is cleared after every pointer-up.
- The reviewed undilated mask remains the training label and intervention authority.
