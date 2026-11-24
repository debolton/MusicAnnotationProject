# Install HF-03.3 over the working HF-03.2/HF-03.1 repository
1. Copy `app/component_review.py` and the static review files into the existing app.
2. Merge `app/main_hf033_routes.py` into `app/main.py` after the existing models/routes.
3. Add `<link rel="stylesheet" href="/static/review.css">` to the HTML head.
4. Insert `component_review_fragment.html` after the existing Authorise/component area.
5. Add `<script src="/static/review.js"></script>` after the existing app.js script.
6. In the existing successful Authorise handler, call `HF033.openReview(sid)` after the component response has loaded, or add a button: `<button onclick="HF033.openReview(sid)">Review components one at a time</button>`.
7. Do not replace the working Dockerfile, pinned runtime, startup gate, mask editor, V7.8 or V7.7F files.
