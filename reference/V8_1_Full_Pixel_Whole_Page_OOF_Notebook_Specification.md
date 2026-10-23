**MUSIC ANNOTATION PROJECT**

V8.1 Notebook Specification

**Full-Pixel, Whole-Page, Source-Group Out-of-Fold Evaluation**

_Frozen-artifact contract, processing design, outputs and go/no-go gates_

**STATUS AND SCOPE**

V8.1 is a development-only full-pixel evaluation notebook. It applies the already selected 33 × 33 V8 spatial challenger to every eligible held-out pixel, assembles complete corrected pages and evaluates the registered safety and benefit endpoints. V8.1 is not a new architecture, not a centre-region challenger and not an independent promotion test.

| **Field**               | **Specification**                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| **Notebook name**       | 27_V8_1_Full_Pixel_Whole_Page_OOF_Evaluation.ipynb                                          |
| **Runtime**             | GPU preferred for batched CNN inference; CPU permitted only if exact replay is demonstrated |
| **Data role**           | Development source-group OOF only                                                           |
| **Comparator**          | Frozen V7.8 geometry_t80 output                                                             |
| **Mask authority**      | Reviewed authoritative corrected mask; dilation 2 under the frozen contract                 |
| **Primary output**      | Actual whole corrected V8.1 pages plus quantitative OOF evidence                            |
| **Promotion authority** | None. Passing results permit creation of a separate frozen challenger bundle only.          |

# 1. Scientific question and fixed interpretation

Among all eligible observed dark pixels inside each authoritative removal mask, does the frozen V8 spatial model preserve additional genuine printed ink beyond V7.8 while satisfying the already registered full-pixel safety, utility, source-stability and uncertainty gates?

**Output semantics**

The final page may be computed in one pipeline, but its regression identity is fixed: V8.1 must equal V7.8 at every coordinate where V8.1 does not validly emit PRINT. At valid PRINT coordinates only, V8.1 copies the exact observed byte.

## 1.1 V8.1 is not a redesign

- No change to the 33 × 33 channel tensor.
- No new context scale, centre-region target, segmentation loss or architecture search.
- No threshold movement, margin change, label reinterpretation or post-result calibration.
- No use of independent pages or independent outcomes.
- No change to V7.8 mask preparation, reconstruction or routing semantics.

A future centre-pixel-stability or centre-region experiment would be a separately pre-registered V8.2 or later challenger. It must not be smuggled into V8.1.

# 2. Frozen artefact contract

V8.1 must fail before page scoring if any required artefact is absent, ambiguous, mutable, unhashed or inconsistent. The exact durable filenames for every trained V8 fold artefact were not found in the retrieved project references. Therefore the notebook must resolve them through a signed artifact inventory rather than guess paths.

**Mandatory preliminary gate**

Create or locate a V8.0 artifact_inventory.json that names every file below, records SHA-256, byte size, semantic role, fold ID, producing notebook/run and compatibility metadata. If no complete inventory can be constructed from durable outputs, V8.1 must stop. It must not retrain models merely to fill missing artefacts.

| **Artefact family**         | **Expected object**                                                                | **Purpose**                                                           | **Status**                                  |
| --------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------- |
| **Pre-registration**        | Observed-print v1.1 machine-readable constants or a verified transcription         | Target, allowlist, thresholds, margins, gates and metric definitions  | Required                                    |
| **Development roster**      | development_score_manifest.csv                                                     | Page and source-group identity; reviewed-mask and hard-negative flags | Required                                    |
| **Fold map**                | Frozen source-group fold_map.csv                                                   | Exact held-out fold for every page; no group overlap                  | Required                                    |
| **Channel schema**          | Ordered channel_schema.json                                                        | 33 × 33 channel names, order, dtype, normalisation and constants      | Required                                    |
| **Tensor builder identity** | Source hash or packaged implementation hash                                        | Proves full-pixel tensors match V8.0 training tensors                 | Required                                    |
| **Fold checkpoints**        | One compact CNN checkpoint per OOF fold                                            | Scores each page only with the model that excluded its source group   | Required                                    |
| **Fold model manifests**    | Architecture, hyperparameters, seed and training roster hashes                     | Rejects mismatched checkpoints                                        | Required                                    |
| **Calibration objects**     | Frozen fold/cross-fitted calibration artefacts or deterministic calibration tables | Produces calibrated PRINT/ANNOTATION probabilities without refit      | Required                                    |
| **OOD artefacts**           | Frozen training-distribution statistic and 99th-percentile cutoff                  | Implements registered OOD-to-UNCERTAIN rule                           | Required if used in V8.0 decision rule      |
| **Ensemble definition**     | Five-member prediction/decision roster and disagreement rule                       | Implements ≥2 of 5 disagreement trigger                               | Required if V8.0 emitted ensemble decisions |
| **Decision policy**         | policy.json / constants.json with policy ID                                        | 0.995 thresholds, 0.990 margin, connected-run rule and class mapping  | Required                                    |
| **Replay pack**             | replay_test.json plus frozen sample tensors/scores/decisions                       | Exact pre-run replay gate                                             | Required                                    |
| **V7.8 canonical source**   | Frozen implementation hash and equivalence record                                  | Baseline construction and non-PRINT regression                        | Required                                    |
| **V7.5/V7.8 page arrays**   | Observed, mask, Telea, geometry_t80, route/support arrays                          | Canonical page reconstruction inputs                                  | Required                                    |
| **Clean truth method**      | Locked printed-ink implementation/hash                                             | Post-lock labels and scoring only                                     | Required                                    |
| **V8.0d lock identity**     | Known run lock and associated manifest/replay records                              | Pins the previously reproduced full-pixel audit source                | Required when V8.0d is the selected source  |

## 2.1 Artifact inventory minimum schema

| **Field**                           | **Required meaning**                                              |
| ----------------------------------- | ----------------------------------------------------------------- |
| **artifact_id**                     | Stable logical name                                               |
| **relative_path**                   | Path beneath the declared frozen run root                         |
| **sha256**                          | File-content SHA-256                                              |
| **bytes**                           | File size                                                         |
| **role**                            | checkpoint, calibrator, schema, policy, replay, array or manifest |
| **fold_id**                         | Fold identity where applicable                                    |
| **source_group_test_roster_sha256** | Hash of the held-out source groups/pages                          |
| **producer**                        | Notebook/version/run that created the artefact                    |
| **compatibility_id**                | Shared model/channel/policy compatibility identity                |
| **required**                        | Boolean; missing required artefacts stop the notebook             |

# 3. Input cohort and leakage controls

- The page roster must match the frozen development roster exactly unless a pre-outcome amendment explicitly narrows it.
- Every page must have one source_group and one fold assignment.
- No source group may be present in both training and scoring sides of a fold.
- Clean pages and truth-derived maps may be loaded only after probabilities, class decisions and output pages have been written and hashed.
- Raw page coordinates, page identity, title, edition and source-group identity are prohibited inference channels.
- The frozen V7.8 independent collection remains outside model fitting, thresholding and V8.1 development selection.
- Duplicate page/image/tensor identities across folds must be detected by content hashes, not filenames alone.

# 4. Canonical page preparation

1. Load the aligned annotated page and reviewed corrected seed mask by exact filename stem.
2. Validate readability, dimensional identity and input SHA-256 against the development roster.
3. Prepare the authoritative dilation-2 removal mask using the frozen V7.8 implementation.
4. Load or rebuild Telea, background surface, normalised image, geometry_t80 output, preserve/remove/ambiguous routes and horizontal/vertical/compact support maps.
5. Run the V7.8.2-style equivalence gate against durable reference arrays before any CNN scoring.
6. Assert that all V7.8 outputs are byte-for-byte identical outside the removal mask.

**Baseline gate**

A page whose V7.8 geometry, routing or support arrays do not reproduce exactly is excluded as an implementation failure, not scored as model performance. The notebook writes the failure artefacts and stops the run rather than continuing with a partial cohort.

# 5. Eligible full-pixel population

Candidates are the registered population: observed pixels inside the authoritative dilation-2 mask, observed greyscale below 200, and not already preserved by frozen V7.8. The notebook must write the population before model inference.

| **Output**                          | **Required columns / content**                                                                                  |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **eligible_pixel_manifest.parquet** | source_group, page, fold_id, y, x, observed_value, component_id, V7.8 route context, sparse/hard-negative flags |
| **eligibility_summary.csv**         | Eligible pixels by page, source group, fold and route context                                                   |
| **eligibility_hash.json**           | Semantic hash of ordered (page, y, x) rows plus input-array hashes                                              |
| **exclusions.csv**                  | Every excluded page/pixel class and reason, without silently dropping failures                                  |

# 6. Frozen tensor construction and batched OOF inference

| **Channel family**                  | **Contract**                         |
| ----------------------------------- | ------------------------------------ |
| **Observed greyscale**              | 33 × 33 raw observed patch           |
| **Background-normalised greyscale** | Frozen paper-normalised patch        |
| **Authoritative mask**              | Binary mask patch                    |
| **Mask distance**                   | Frozen mask-distance representation  |
| **Horizontal support**              | Frozen V7.8 horizontal support patch |
| **Vertical support**                | Frozen V7.8 vertical support patch   |
| **Compact support**                 | Frozen V7.8 compact support patch    |
| **Preserve route**                  | Frozen V7.8 preserve indicator       |
| **Remove route**                    | Frozen V7.8 remove indicator         |
| **Ambiguous route**                 | Frozen V7.8 ambiguous indicator      |
| **Observed edges**                  | Frozen edge channel                  |

The ordered schema in the frozen channel artefact is authoritative. The list above reflects the controlled V8.0 design and must not override the durable schema if the frozen artefact differs. A mismatch is a stop condition.

- Extract tensors in deterministic page/y/x order.
- Use bounded batches; batch size may change for memory only and must not change outputs.
- Score each page with the checkpoint assigned by its held-out fold.
- Preserve raw logits or raw probabilities before calibration.
- Apply only the frozen calibration objects; do not fit calibration in V8.1.
- Apply frozen OOD and ensemble disagreement rules exactly when those rules were part of V8.0.
- Write all raw and calibrated outputs before clean truth is made available.

# 7. Decision rule and spatial post-processing

| **Condition**                                                                                                                   | **V8.1 decision**                               |
| ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Calibrated PRINT probability ≥ 0.995, PRINT minus ANNOTATION margin ≥ 0.990, and all integrity/OOD/disagreement checks pass** | Provisional PRINT                               |
| **Calibrated ANNOTATION probability ≥ 0.995 and PRINT rule does not pass**                                                      | ANNOTATION                                      |
| **Maximum PRINT probability below 0.995 without high-confidence ANNOTATION**                                                    | UNCERTAIN                                       |
| **Missing/non-finite/schema mismatch/OOD/excluded border/ensemble disagreement**                                                | UNCERTAIN plus fail-closed reason               |
| **Isolated PRINT or connected candidate run shorter than 2 pixels**                                                             | Convert to UNCERTAIN; no singleton preservation |
| **All remaining cases**                                                                                                         | UNCERTAIN                                       |

Spatial connected-run filtering must be applied after all eligible pixels on the page have provisional decisions, not batch-by-batch. Final decisions must therefore be page-global and deterministic.

# 8. Whole-page assembly contract

7. Start from the exact frozen V7.8 geometry_t80 page.
8. At final PRINT coordinates, copy the exact observed greyscale byte.
9. At ANNOTATION and UNCERTAIN coordinates, retain the frozen V7.8 byte.
10. Outside the authoritative removal mask, retain the observed page byte exactly.
11. Assert that V8.1 differs from V7.8 only at final PRINT coordinates.
12. Assert that each changed V8.1 byte equals the observed input byte.
13. Save whole corrected pages before truth scoring.

# 9. Prediction lock before truth scoring

V8.1 is development OOF, but the separation between inference and truth must still be explicit. The notebook writes and hashes all inference artefacts before loading clean pages or constructing truth labels.

| **Locked object**                        | **Minimum content**                                                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **prediction_lock.json**                 | Roster hash, eligibility hash, model/checkpoint hashes, calibrator hashes, schema hash, policy ID, probabilities hash, decisions hash and output-page hashes |
| **all_eligible_oof_predictions.parquet** | One row per eligible pixel with raw/calibrated probabilities, provisional/final class and fail-closed reasons                                                |
| **page_output_hashes.csv**               | Observed, V7.8 and V8.1 semantic hashes for every page                                                                                                       |
| **software_environment.json**            | Python, PyTorch, CUDA/cuDNN, NumPy, OpenCV, pandas and dependency versions                                                                                   |
| **timing_and_resource_usage.csv**        | Per-page candidates, batches, tensor-build time, inference time, peak CPU/GPU memory and output time                                                         |

# 10. Truth scoring and registered metrics

| **Metric**                       | **Definition**                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Extra true print**             | Final V8.1 PRINT pixels not preserved by V7.8 whose frozen truth label is PRINT.                                    |
| **Extra false annotation**       | Final V8.1 PRINT pixels whose frozen truth label is ANNOTATION.                                                     |
| **Truth UNCERTAIN preserved**    | Final PRINT decisions whose truth label is UNCERTAIN; report separately and do not silently count as true or false. |
| **False/true ratio**             | Extra false annotation / max(extra true print, 1).                                                                  |
| **False-preservation rate**      | Extra false annotation / all challenger-added preserved pixels.                                                     |
| **Net safety utility**           | Extra true print − 5 × extra false annotation − 500 × catastrophic components.                                      |
| **Catastrophic component**       | ≥50 extra false pixels, or false/true >1.0 with ≥20 extra false pixels.                                             |
| **Eligible PRINT coverage**      | Final PRINT / all eligible pixels.                                                                                  |
| **UNCERTAIN rate**               | Final UNCERTAIN / all eligible pixels.                                                                              |
| **Page coverage**                | Pages with ≥10 extra true-print pixels and no catastrophic component.                                               |
| **Outside-mask drift**           | Required zero.                                                                                                      |
| **Non-PRINT V7.8 regression**    | Required zero.                                                                                                      |
| **Whole-page perceptual record** | Page-level V7.8 versus V8.1 review; diagnostic, not a replacement for registered gates.                             |

- Use 10,000 page-clustered bootstrap resamples for utility, false/true ratio and false-preservation rate.
- Report Wilson 95% upper bounds for binomial summaries.
- Report pooled, page, source-group, route-context and sparse/hard-negative results.
- Do not treat pixels as statistically independent.

# 11. Development gates D1–D12

| **Gate**                    | **Required V8.1 OOF result**                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **D1 Integrity**            | Zero schema violations, duplicate leakage and non-finite scored inputs; exact replay.                                     |
| **D2 Immutability**         | Zero outside-mask drift and zero non-PRINT differences from V7.8.                                                         |
| **D3 Catastrophic safety**  | Zero catastrophic components pooled and in every source group.                                                            |
| **D4 False preservation**   | False/true ≤0.10 and page-clustered 95% upper bound ≤0.20.                                                                |
| **D5 Pixel safety**         | False-preservation rate ≤0.50% and 95% upper bound ≤1.00%.                                                                |
| **D6 Positive utility**     | Utility >0 and page-clustered 95% lower bound >0.                                                                         |
| **D7 Non-trivial benefit**  | ≥500 extra true-print pixels across ≥10 pages and ≥3 source groups.                                                       |
| **D8 Source stability**     | Each source group with ≥3 eligible pages: false/true ≤0.25 and no catastrophe.                                            |
| **D9 Sparse/hard-negative** | Zero catastrophes; false-preservation rate ≤0.25%.                                                                        |
| **D10 Uncertainty**         | Report UNCERTAIN and PRINT coverage; >95% UNCERTAIN is non-operational evidence.                                          |
| **D11 Baseline dominance**  | Beat the best registered non-learned baseline on utility without worsening D3–D5.                                         |
| **D12 Visual audit**        | Review all catastrophes, 50 most false-heavy, 50 most beneficial and 50 most uncertain; no unmodelled systematic failure. |

**Conjunctive interpretation**

Failure of any development gate prevents creation of the independent-test challenger bundle. If safety passes but D7 fails, retain the result as a low-coverage research baseline without weakening thresholds.

# 12. Required outputs and directory layout

```
observed_print_classifier_v8_1/<run_id>/
  01_input_audit/
  02_v78_equivalence/
  03_eligibility/
  04_oof_predictions/
  05_prediction_lock/
  06_truth_scoring/
  07_whole_pages/
  08_visual_audit/
  09_replay/
  artefact_inventory_resolved.json
  experiment_manifest.json
  v8_1_verdict.json
  executive_summary.txt
```

| **Output**                                    | **Purpose**                                                                     |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| **experiment_manifest.json**                  | All constants, resolved artefacts, hashes, roster, fold and software identities |
| **source_group_split_audit.csv**              | No-overlap proof                                                                |
| **eligible_pixel_manifest.parquet**           | Full registered population                                                      |
| **all_eligible_oof_predictions.parquet**      | Raw/calibrated probabilities and final decisions                                |
| **results_by_page.csv**                       | Page safety, benefit, coverage, uncertainty and timing                          |
| **results_by_source_group.csv**               | Source stability evidence                                                       |
| **catastrophic_components.csv**               | Every catastrophic candidate; expected empty                                    |
| **truth_uncertain_prints.csv**                | Final PRINT assertions with truth UNCERTAIN                                     |
| **whole_pages/&lt;page&gt;/v7_8.png**         | Incumbent rendering                                                             |
| **whole_pages/&lt;page&gt;/v8_1.png**         | V8.1 corrected page                                                             |
| **whole_pages/&lt;page&gt;/decision_map.png** | PRINT/ANNOTATION/UNCERTAIN diagnostic                                           |
| **whole_pages/&lt;page&gt;/difference.png**   | V8.1 minus V7.8 locator                                                         |
| **visual_audit_queue.csv**                    | Registered review cases and status                                              |
| **replay_test.json**                          | Exact rerun results                                                             |
| **v8_1_verdict.json**                         | D1–D12 pass/fail and first failed gate                                          |

# 13. Replay, reproducibility and failure handling

- Require two consecutive unchanged artifact-inventory and input-lock calculations before the definitive run.
- Replay a frozen multi-page sample through tensor construction, model inference, calibration, final decisions and page assembly.
- Require exact decision and byte equality; probability tolerance is permitted only if already specified in the frozen replay contract.
- Write failure tables and manifests before raising RuntimeError.
- Never continue after missing fold artefacts, hash mismatch, incompatible schema, partial roster, non-finite tensor/probability or V7.8 equivalence failure.
- Never overwrite V8.0 artefacts, V7.8 caches or prior V8.1 runs.

# 14. Whole-page review design

The main visual judgement uses actual corrected pages, not the distracting pre-corrected comparison imagery used in the exploratory V8.0f review.

- Show V7.8 and V8.1 at identical whole-page scale and contrast.
- Ask whether V8.1 visibly preserves notation that V7.8 damages.
- Ask whether V8.1 visibly retains annotation that V7.8 removes.
- Allow "no visible difference" as a proper result.
- Keep the annotated input, clean truth, mask and difference overlays as secondary diagnostics, not the initial quality judgement.
- Record Page039-like coherent structural effects at page/component level without retrospectively changing pixel truth.

# 15. Decision logic

| **Outcome**                                                                | **Decision**                                                                                                   |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Any artifact, leakage, replay or V7.8 equivalence failure**              | STOP; invalidate implementation before interpreting metrics.                                                   |
| **Any D1–D6, D8 or D9 safety failure**                                     | Reject V8.1 at the frozen operating point; retain V7.8.                                                        |
| **Safety gates pass but D7 fails**                                         | Retain as low-coverage research baseline; do not commission independent testing.                               |
| **Safety and D7 pass but D12 identifies an unmodelled systematic failure** | Do not freeze; define a new pre-registered challenger rather than patching V8.1.                               |
| **D1–D12 all pass**                                                        | Create a separate immutable V8.1 challenger bundle, replay it, then commission one untouched independent test. |
| **Whole-page improvement is visually negligible despite gate pass**        | Record the quantitative pass and practical limitation in the decision record; no automatic promotion.          |

# 16. Freeze bundle created only after a full pass

V8.1 itself consumes fold artefacts for OOF evaluation. If and only if D1–D12 pass, a separate freeze step may refit the chosen architecture once on all permitted development data and create one immutable independent-test bundle.

| **Frozen bundle object**         | **Required identity**                                                        |
| -------------------------------- | ---------------------------------------------------------------------------- |
| **model checkpoint**             | SHA-256, architecture/configuration and training roster hash                 |
| **calibrator**                   | SHA-256 and calibration fitting record                                       |
| **channel schema**               | Ordered channels, patch size, dtype, normalisation and constants             |
| **policy.json**                  | PRINT/ANNOTATION thresholds, margin, OOD/disagreement/run rule and policy ID |
| **preprocessing implementation** | Source/package hash                                                          |
| **replay pack**                  | Inputs, expected probabilities, decisions and page bytes                     |
| **software environment**         | Pinned package/runtime versions                                              |
| **freeze manifest**              | Hashes every object and records development lock and D1–D12 verdict          |

# 17. Notebook cell map

14. Title, scope and no-promotion warning
15. Configuration with all change controls defaulting to read-only
16. Resolve and validate artifact_inventory.json
17. Load pre-registration constants and assert exact values
18. Validate development roster and source-group fold map
19. Run canonical V7.8 equivalence gate
20. Build and lock the eligible pixel manifest
21. Replay frozen V8 sample
22. Extract 33 × 33 tensors and score held-out folds in batches
23. Apply calibration, OOD, disagreement and provisional class rule
24. Apply page-global connected-run filtering
25. Assemble and lock whole V8.1 pages
26. Unlock clean truth and score registered labels
27. Calculate D1–D12 with clustered uncertainty
28. Generate page/source-group outputs and whole-page review queue
29. Write verdict, manifest, timing/resource report and executive summary

# 18. Known uncertainties and explicit assumptions

| **Item**                                                            | **Specification treatment**                                                                                                                |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Exact durable V8 fold artifact filenames**                        | Not established by the retrieved references. V8.1 requires an explicit inventory and stops if one cannot be verified.                      |
| **Whether V8.0 used a persisted five-member ensemble at inference** | The registered rule requires disagreement handling, but the exact stored ensemble implementation must be verified from durable V8 outputs. |
| **OOD implementation and persisted cutoff**                         | Must be loaded from the frozen artefacts if used. It must not be reconstructed from V8.1 full-pixel outcomes.                              |
| **Calibration persistence format**                                  | May be joblib, JSON lookup or deterministic table, but identity and replay must be exact.                                                  |
| **Runtime and peak memory**                                         | Unknown until measured; report per page and batch. Runtime cost is not a reason to change the scientific population.                       |
| **Truth-UNCERTAIN treatment in headline metrics**                   | Report separately and preserve the registered true/false definitions; do not relabel after outcome inspection.                             |

# 19. Source basis

- **&lt;File&gt;Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx&lt;/File&gt;** — scope, eligible population, three-class rule, uncertainty triggers, metrics, D/I gates, lock and audit requirements.
- **&lt;File&gt;V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx&lt;/File&gt;** — 33 × 33 spatial tensor, compact CNN, common folds/calibration/thresholds and controlled-comparison limits.
- **&lt;File&gt;Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx&lt;/File&gt;** — full-pixel OOF as the required next stage and the subsequent freeze/independent/shadow sequence.
- **&lt;File&gt;V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx&lt;/File&gt;** — authoritative-mask boundary, exact preservation, V7.8 fallback philosophy and separate audit artefacts.
- **&lt;File&gt;17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt&lt;/File&gt;** — canonical page arrays, deterministic semantic locking, sealed truth and exact corrected-mask preparation.
- **&lt;File&gt;18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt&lt;/File&gt;** — stage-by-stage V7.8 equivalence and first-failure semantics.
- **&lt;File&gt;V7_7F_Design_Record_Applying_the_Frozen_Structural_Policy_to_V7_8.docx&lt;/File&gt;** — frozen artifact verification, no refit/recalibration/threshold change and source/page uncertainty reporting.
- **&lt;File&gt;09_Pilot_Proposed_Correction_Routing_Reviewer.txt&lt;/File&gt;** — canonical implementation hash, mandatory regression gate and actual-page review presentation.
- **&lt;File&gt;Production_Retraining_Policy_YOLO_and_Structural_Safety.docx&lt;/File&gt;** — separate versioning, frozen incumbent/challenger and documented non-automatic promotion.

# 20. Final commissioning statement

**V8.1 DECISION**

Build V8.1 as a full-pixel, source-group out-of-fold evaluation of the already selected 33 × 33 spatial challenger. It must resolve and verify frozen fold checkpoints, calibration, schema, policy, replay and V7.8 artefacts before scoring; evaluate every registered eligible pixel; assemble actual whole corrected pages; lock predictions before truth; calculate D1–D12; and refuse to create an independent-test bundle unless every gate passes.

**Confidence: 94%.** The scientific contract, gates and sequence are explicit in the project records. Confidence is below 100% because the retrieved references do not specify the exact durable filenames and formats of all V8 fold checkpoints, calibrators, ensemble and OOD artefacts. The specification therefore makes verified artifact inventory a mandatory first gate rather than inventing those details.

**Next question:** Do the completed V8.0b/V8.0d run directories contain every fold checkpoint, calibration object, ensemble/OOD artefact and replay sample required to execute V8.1 without any refitting?