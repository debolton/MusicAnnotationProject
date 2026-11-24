# Frozen files to transfer from Google Drive

## Required now: YOLO draft-mask generator
Destination: `frozen/yolo_detector/`
- [ ] `best.pt` copied from `training_runs/20261006_150912_yolo11n-seg/weights/best.pt`
- [ ] `detector_identity.json` recording the original Drive path, MD5, SHA-256, dataset release/training run identity and approval status
- [ ] `inference_config.json` with exactly:
  - `prediction_confidence: 0.20`
  - `tile_size: 640`
  - `tile_stride: 320`
  - `model_image_size: 640`
  - `device: 0`
  - `retina_masks: true`
  - `mask_resize_interpolation: INTER_NEAREST`
  - `instance_mask_threshold: 0.5`
  - `tile_overlap_merge: maximum_instance_confidence_per_pixel`
  - `confidence_storage_dtype: float16`
  - `draft_threshold: 0.20`
- [ ] Any retained validation/GST/hard-negative reports identifying this checkpoint as the incumbent

## Required for Pilot 09 structural scoring
Destination: `frozen/structural_policy_v7_7f/artefacts/`
- [ ] `policy.json`
- [ ] `predictor_schema.json`
- [ ] `structural_model.joblib`
- [ ] `isotonic_calibrator.joblib`
- [ ] `replay_test.json`

## Required only for V8.2 scoring/replay
Destination: `frozen/observed_print_v8_2/`
- [ ] `experiment_identity.json`
- [ ] `development_summary.json`
- [ ] `executive_summary.txt`
- [ ] `software_environment.json`
- [ ] `inputs/development_score_manifest.csv`
- [ ] `inputs/fold_map.csv`
- [ ] `inputs/channel_schema.json`
- [ ] every `models/fold*_seed42.pt`
- [ ] every `calibrators/fold*_seed42_isotonic.joblib`
- [ ] every matching `calibrators/fold*_seed42_isotonic.json`
- [ ] `audit/model_inventory.csv`
- [ ] `audit/calibrator_inventory.csv`

## Runtime data, not frozen models
Do not place these under `frozen/`:
- `source/annotated_only/`
- `human_in_the_middle/draft_masks/`
- `human_in_the_middle/confidence_maps/`
- `human_in_the_middle/draft_audit/`
- `human_in_the_middle/audit/`
- `aligned/`
- `masks/corrected/`
- `human_in_the_middle/previews/`
