# Deploy
1. Create or convert the Hugging Face Space to **Docker** SDK.
2. Copy the verified `best.pt` to `frozen/yolo_detector/best.pt`.
3. Copy the complete V7.7F bundle to `frozen/structural_policy_v7_7f/artefacts/` if component scoring is required.
4. Upload the repository contents to the Space root.
5. Use CPU Basic initially.

V7.7F is optional for the first mask-editor test. If absent, components are labelled `unscored`; canonical V7.8 still runs.
