# HF-02.1 drop-in patch
Overlay these files onto the working HF-02 repository. Keep the existing frozen artefacts and existing `music_annotation/hitl_generator.py`, `v77_policy.py`, `v78_engine.py`, and `supervised.py`.

Changes:
- full-resolution page/confidence/masks live in server-side `runtime_sessions/` files;
- `gr.State` carries only a short session ID and decision dictionary;
- browser previews have a maximum width of 1200 px;
- ADD and ERASE are separate paint layers;
- authoritative seed is `(draft OR ADD) AND NOT ERASE`;
- edited preview masks are upscaled with nearest-neighbour interpolation;
- V7.8 and V7.7F still run at original resolution;
- only final approval returns one full-resolution image.
