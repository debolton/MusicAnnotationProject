# Frozen model transfer checklist

## Needed for the current Hugging Face `app.py`
- [ ] Copy the promoted YOLO checkpoint from `training_runs/20261006_150912_yolo11n-seg/weights/best.pt` to `frozen/yolo_detector/best.pt`.
- [ ] Do not rename the checkpoint after copying.

The deployed draft generator uses the Notebook 06 contract: confidence 0.20, 640×640 tiles, stride 320, `imgsz=640`, retina masks, nearest-neighbour mask resize, >0.5 instance-mask threshold, maximum confidence per overlapping pixel, and final draft threshold 0.20.

## Required when enabling frozen V7.7F structural scoring / Pilot 09
Copy to `frozen/structural_policy_v7_7f/artefacts/`:
- [ ] `policy.json`
- [ ] `predictor_schema.json`
- [ ] `structural_model.joblib`
- [ ] `isotonic_calibrator.joblib`
- [ ] `replay_test.json`

## Required when enabling/replaying V8.2 observed-print scoring
Root `frozen/observed_print_v8_2/`:
- [ ] `experiment_identity.json`
- [ ] `development_summary.json`
- [ ] `executive_summary.txt`
- [ ] `software_environment.json`

`inputs/`:
- [ ] `development_score_manifest.csv`
- [ ] `fold_map.csv`
- [ ] `channel_schema.json`

`models/`:
- [ ] every `foldN_seed42.pt`

`calibrators/`:
- [ ] every `foldN_seed42_isotonic.joblib`
- [ ] every corresponding `foldN_seed42_isotonic.json`

`audit/`:
- [ ] `model_inventory.csv`
- [ ] `calibrator_inventory.csv`

## Do not copy mutable HITL/runtime data into `frozen/`
Draft masks, confidence maps, corrected masks, review audits and generated pages belong under persistent runtime storage, not frozen model directories.
