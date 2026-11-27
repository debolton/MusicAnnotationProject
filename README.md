---
title: Music Annotation HF-04.2
emoji: 🎼
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
---
# HF-04.2
Docker/FastAPI and the HF-04.0 mobile mask editor are retained. The backend recreates canonical V7.8 geometry_t80 from the reviewed undilated seed mask, then invokes the frozen Notebook 91 A5 primitives in this exact order: load_bundle, make_padded, score_coordinates, decisions, assemble. Copy the complete unmodified Notebook 91 bundle to frozen/a5_v1/. Generated __pycache__ and .pyc files are excluded from hash verification.

## HF-04.3 additions

HF-04.3 makes the Ardross A5 frozen replay a mandatory startup test and adds a separate diagnostic for comparing HF-generated V7.8 channels with the historical replay arrays. See `docs/HF043.md`.

# HF-04.4 startup-evidence endpoint

Adds `GET /startup-evidence` to the existing Docker/FastAPI application.

After both mandatory startup gates have passed, the endpoint returns `hf_04_4_startup_evidence.zip` containing all files under `startup_evidence/`, including:

- `ardross_v78_equivalence/channel_comparison.csv`
- `ardross_v78_equivalence/summary.json`
- any `*_absdiff.png` files generated if a future channel comparison differs

If the evidence has not been generated, the route fails closed with HTTP 503.

Apply from the repository root:

```bash
python apply_startup_evidence_endpoint.py
```

Then rebuild the Space and visit `/startup-evidence`.

# Software License - MIT

Copyright 2026 David Bolton

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.