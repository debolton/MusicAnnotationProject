# Deploy HF-04.0
Copy unchanged frozen artefacts into the placeholders below, then deploy as a Hugging Face Docker Space.

- `frozen/yolo_detector/best.pt`
- complete `frozen/p_print_provider_v8_2f/` bundle
- complete `frozen/notebook52f_remove_bundle/` bundle

Do not reconstruct, retrain, recalibrate, or alter either downstream bundle. The included V8.9 operator validates bundle hashes/schema/policy at load. CPU is accepted by the operator but GPU is preferable for page-scale p_print CNN inference.
