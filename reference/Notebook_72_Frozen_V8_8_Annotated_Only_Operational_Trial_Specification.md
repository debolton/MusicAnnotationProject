Notebook 72 Specification

Frozen V8.8 Annotated-Only Supervised Operational Trial

**Implementation contract, truth firewall, cohorts, outputs, review protocol and decision gates | 16 November 2026**

COMMISSIONING DECISION  
Notebook 72 must process every eligible reviewed page from source/annotated_only and source/annotated through one identical clean-free runtime path. It must lock all predictions, decisions and corrected pages before resolving source/clean. Clean truth is then used only for post-lock scoring where available; every page receives the same blinded performance-usability review.

# 1. Status and scope

| **Field**           | **Specification**                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Notebook name       | 72_Frozen_V8_8_Annotated_Only_Supervised_Operational_Trial.ipynb                                                            |
| Runtime             | GPU preferred if the frozen V8.8 operator uses CNN inference; CPU permitted only if the operator replay contract permits it |
| Experiment type     | Operational validation / supervised shadow trial; not retraining and not automatic production release                       |
| Runtime populations | Eligible reviewed pages in source/annotated_only and source/annotated                                                       |
| Mask authority      | Reviewer-corrected authoritative mask only                                                                                  |
| Correction engine   | Exact frozen V8.8 downstream operator and policy bundle                                                                     |
| Clean truth         | Prohibited during runtime; post-lock scoring only for paired pages                                                          |
| Human evaluation    | Blinded annotated-original versus corrected-page performance-usability review for the complete cohort                       |
| Primary output      | Complete corrected pages plus decision, probability, integrity, lineage and review artefacts                                |
| Promotion authority | None. The result informs a supervised-pilot, targeted-use or redesign decision.                                             |

PRIMARY QUESTION  
Can the frozen V8.8 downstream system that earned evidence on paired clean/annotated development data produce safe and usable corrected pages when only an annotated page, a reviewer-authoritative mask and runtime-observable evidence are available?

# 2. Non-negotiable operating principles

- Clean truth must not participate in correction. If the corrected page depends on clean pixels or clean-derived fields, Notebook 72 has failed its purpose.
- The reviewer-corrected mask is the sole intervention authority. Pages without one are excluded with an explicit reason; there is no silent fallback to unreviewed YOLO output.
- Pixels outside the prepared authoritative mask remain byte-for-byte equal to the runtime observed page.
- V8.8 is frozen. Notebook 72 must not retrain, recalibrate, move thresholds, change channels, alter the fallback engine or reinterpret the decision policy.
- Where V8.8 does not validly assert PRINT or another registered intervention, the exact frozen fallback/incumbent behaviour is retained.
- Inference and evaluation are separate phases, joined only after prediction/output lock creation.
- Quantitative truth scoring and whole-page usability review are complementary. Neither substitutes for the other.

# 3. Cohort contract

| **Cohort**                                  | **Source**                                                      | **Runtime treatment**                                                                             | **Post-lock evaluation**                                                                                                             |
| ------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| A. Genuine annotated-only                   | source/annotated_only                                           | Run frozen V8.8 from reviewed aligned/observed page and authoritative mask. No clean path exists. | Integrity, replay, decision/coverage diagnostics and blinded performance-usability review.                                           |
| B. Paired pages processed as annotated-only | source/annotated, with clean counterpart sealed in source/clean | Run the identical runtime pipeline. Do not resolve or register clean during inference.            | After lock, reveal clean counterpart and compute quantitative structural/cosmetic metrics, plus the same blinded review as Cohort A. |

The two cohorts are combined for runtime processing and human review. They are separated only for truth-dependent scoring.

# 4. Required inputs and artefact identity

| **Input / artefact**          | **Required contract**                                                                                                  | **Failure behaviour**                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Annotated page roster         | All supported images in the two declared source populations, deduplicated by content identity and filename stem rules. | Write roster failure and stop or exclude the affected page with reason.                 |
| Observed/aligned page         | Resolved through authoritative review lineage; shape-compatible with mask.                                             | Exclude page; do not resize opportunistically.                                          |
| Authoritative seed mask       | Reviewer-corrected, undilated mask with provenance/audit record.                                                       | Exclude page if missing, unreadable, ambiguous or stale.                                |
| Frozen V8.8 operator adapter  | One durable callable implementation, preferably v8_8_operator.py, with exact source hash and declared API.             | Notebook stops before roster processing if unresolved or incompatible.                  |
| Frozen model/policy artefacts | Model/checkpoint, calibrator, schema/channels, thresholds, OOD/disagreement rules and replay pack as applicable.       | Hash or compatibility mismatch stops the run.                                           |
| Frozen fallback engine        | Exact incumbent reconstruction/routing implementation used by V8.8.                                                    | Equivalence failure invalidates run.                                                    |
| Clean pages                   | Not resolved until output lock; matched by explicit post-lock evaluation logic.                                        | Missing clean page changes cohort role only; it does not invalidate runtime processing. |
| Configuration                 | Run constants, source paths, mask preparation settings, software versions and resource controls.                       | Invalid or unrecorded configuration stops the run.                                      |

# 5. Minimum V8.8 operator adapter contract

Notebook 72 must call a separately identifiable frozen operator rather than reconstructing V8.8 from historical notebooks. The adapter may wrap existing code, but its interface and semantic outputs must be explicit.

| **Element**              | **Minimum requirement**                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Callable                 | run_page(observed, authoritative_seed_mask, runtime_context) -> PageResult                                         |
| Observed input           | Single-channel page array, stable dtype and exact shape                                                            |
| Mask input               | Reviewer-authoritative seed mask; adapter applies only the frozen preparation/dilation contract                    |
| Runtime context          | Frozen artefact paths/objects, policy identity, source metadata and deterministic settings; no clean-derived field |
| PageResult.corrected     | Complete corrected page with same shape/dtype as observed                                                          |
| PageResult.baseline      | Exact frozen fallback/incumbent page used where V8.8 does not intervene                                            |
| PageResult.prepared_mask | Exact intervention mask used for assertions                                                                        |
| PageResult.decisions     | Registered class/route map, including PRINT / ANNOTATION / UNCERTAIN or frozen equivalent                          |
| PageResult.probabilities | Raw and calibrated probability arrays for eligible pixels where the operator includes them                         |
| PageResult.changed       | Changed-pixel map and valid intervention map                                                                       |
| PageResult.audit         | Model, calibrator, schema, policy, implementation, replay and configuration identities                             |
| Determinism              | Repeated calls on replay samples produce the contractually required exact decisions and page bytes                 |

# 6. Truth firewall and technical enforcement

FORBIDDEN DURING RUNTIME  
source/clean; clean registration; clean pixels; clean ink/edge maps; truth labels; reconstruction errors against clean; structural or cosmetic outcomes; any field derived after observing clean truth.

- CLEAN_DIR must not be instantiated, globbed, resolved or passed into any runtime function before the lock state is CLOSED.
- Runtime code must execute under a path-access guard or equivalent instrumentation that records every file open beneath the project root and fails on any pre-lock clean access.
- The roster and runtime context must contain no clean path or truth-derived column.
- Prediction lock creation must precede the cell/function that enables post-lock clean resolution.
- The run manifest records clean_reads_before_lock = 0 and includes the access-audit hash.
- If any clean access occurs before lock, invalidate the entire run. Do not continue and merely flag the affected page.

# 7. Notebook execution sequence

**1\. Configuration and immutable run directory.** Create a unique run directory; declare both source populations, reviewed-mask lineage, operator path/artefact override controls and read-only defaults.

**2\. Resolve and verify frozen V8.8.** Hash the adapter, model, calibrator, schema, policy, fallback engine and replay pack. Run replay before processing project pages.

**3\. Build complete eligible roster.** Enumerate both populations, resolve reviewed page/mask lineage, verify shapes/hashes, classify cohort A/B without reading clean, and write explicit exclusions.

**4\. Close truth path.** Activate clean-directory guard and confirm no runtime object contains a clean path or truth field.

**5\. Smoke test.** Run a declared small roster to prove path resolution, operator compatibility, all output writes, immutability assertions and lock construction. Smoke results are not model-performance evidence.

**6\. Full frozen inference.** Process the full eligible roster without changing operator or policy. Save complete page outputs and per-page audit records.

**7\. Integrity and replay gate.** Assert outside-mask immutability, fallback equivalence, finite/shape-compatible arrays and exact replay on the registered sample.

**8\. Prediction/output lock.** Hash inputs, masks, operator identity, probabilities, decisions, corrected pages, access audit, environment and roster. Mark lock CLOSED.

**9\. Post-lock paired evaluation.** Only now resolve source/clean for eligible paired pages, perform any registered evaluation alignment and calculate truth metrics.

**10\. Blinded performance-usability review.** Generate original-versus-corrected A/B packages for every page with clean truth hidden in the initial review.

**11\. Decision package and upload bundle.** Write combined summaries, failures, source-group results, review outcomes, manifests, hashes and standard review ZIP.

# 8. Per-page runtime processing contract

- Resolve the reviewed aligned/observed page and authoritative undilated mask from audit lineage.
- Validate readability, shape, dtype and content hashes. Do not resize or repair mismatches silently.
- Prepare the intervention mask using the exact frozen V8.8 mask contract.
- Invoke the frozen operator using observable runtime evidence only.
- Assemble the final corrected page according to frozen decision/fallback semantics.
- Assert every outside-mask byte equals the observed page.
- Assert every non-intervention byte equals the frozen fallback page where that equivalence applies.
- Assert every changed byte is justified by an authorised decision and appears in the changed-pixel map.
- Save outputs atomically; incomplete page directories are failures, not valid partial results.
- Write page status as PROCESSED, EXCLUDED or FAILED with one primary reason and supporting details.

# 9. Required per-page outputs

| **Artefact**                    | **Required content**                                                              |
| ------------------------------- | --------------------------------------------------------------------------------- |
| observed.png                    | Runtime observed/aligned annotated page                                           |
| authoritative_seed_mask.png     | Reviewer-corrected undilated mask                                                 |
| prepared_authoritative_mask.png | Exact frozen intervention mask used by V8.8                                       |
| fallback_corrected.png          | Exact frozen baseline/incumbent output                                            |
| v8_8_corrected.png              | Complete Notebook 72 corrected page                                               |
| decision_map.png / .npz         | Registered decisions/routes for eligible pixels                                   |
| probabilities.npz               | Raw/calibrated probabilities and uncertainty fields where applicable              |
| changed_pixels.png / .npz       | All output-versus-observed or output-versus-fallback changes, clearly defined     |
| outside_mask_audit.json         | Count and coordinates summary; required drift = 0                                 |
| page_audit.json                 | Lineage, hashes, operator/policy IDs, timings, exclusions/failures and assertions |
| review_A.png / review_B.png     | Blinded original/corrected presentation with randomised side assignment           |
| review_key.json                 | Sealed A/B identity for later decoding                                            |

# 10. Prediction and output lock

| **Locked object**         | **Minimum identity**                                                          |
| ------------------------- | ----------------------------------------------------------------------------- |
| runtime_roster.csv        | Every eligible/excluded page, cohort, source, lineage and status              |
| input_hashes.csv          | Observed, seed-mask and prepared-mask hashes                                  |
| operator_identity.json    | Adapter source hash, model/calibrator/schema/policy/fallback identities       |
| page_output_hashes.csv    | Fallback, corrected, decisions, probabilities and changed-map semantic hashes |
| runtime_access_audit.csv  | All runtime file accesses with explicit zero clean reads before lock          |
| software_environment.json | Python, NumPy, OpenCV, PyTorch/CUDA where relevant and package versions       |
| prediction_lock.json      | Combined semantic payload, lock ID, creation timestamp and state CLOSED       |
| replay_test.json          | Replay roster, expected/current hashes and exact pass/fail                    |

LOCK RULE  
Post-lock evaluation must refuse to run unless the current runtime roster, operator identity and page-output hashes exactly match prediction_lock.json.

# 11. Quantitative evaluation for paired pages

The quantitative branch is available only for Cohort B and begins after the output lock. It must not alter runtime outputs.

| **Metric family**      | **Required measures**                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Integrity              | Clean reads before lock; outside-mask drift; non-intervention regression; replay                                   |
| Printed structure      | Printed-ink recall/F1, tolerant edge preservation, removed-print fraction, catastrophic print-damage concentration |
| Annotation removal     | Residual annotation / false preservation, removal benefit and remaining burden                                     |
| Whole-page fidelity    | Inside-mask and whole-page error measures, clearly separating structural from cosmetic outcomes                    |
| Coverage / uncertainty | Eligible pixels, PRINT/ANNOTATION/UNCERTAIN counts, acted-upon fraction and no-change fraction                     |
| Stability              | Page-level and source-group stratification; do not report pooled pixels alone                                      |
| Uncertainty intervals  | Page-clustered bootstrap and Wilson bounds where the metric definition supports them                               |

Notebook 72 should carry forward the registered V8.8 metric definitions and gates exactly where a durable V8.8 record exists. If a metric or gate cannot be resolved from the frozen bundle, report it as unresolved rather than inventing a replacement.

# 12. Blinded performance-usability review for all pages

PRIMARY REVIEW QUESTION  
Is this corrected page clean and trustworthy enough to perform from?

| **Outcome**                          | **Definition**                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| PERFORMANCE_READY                    | Usable for performance without material concern.                               |
| PERFORMANCE_READY_WITH_MINOR_RESIDUE | Minor annotation remains but does not materially distract or obscure notation. |
| REVIEW_REQUIRED                      | Possible notation damage, residue or ambiguity needs targeted review.          |
| UNUSABLE_ANNOTATION_REMAINS          | Residual annotation makes the page unsuitable.                                 |
| UNUSABLE_PRINT_DAMAGE                | Correction materially removed or damaged printed notation.                     |

- Randomise A/B side assignment per page and conceal which image is corrected.
- Show annotated original and corrected output at identical scale, contrast and crop.
- Do not show clean truth, decision overlays, difference maps or model probabilities in the initial judgement.
- Record annotation removal: complete / partial / none.
- Record notation damage: none / uncertain / visible material damage.
- Record preference: corrected better / same / worse.
- Record performability: yes / yes with minor concern / no.
- Capture rejection/referral region and reason; diagnostics may be opened only after the initial judgement is locked.
- Preserve reviewer identity or reviewer code, timestamp, page order and review revision.

# 13. Required integrity and decision gates

| **Gate**                     | **Pass condition**                                                                                                  | **Failure consequence**                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| G1 Frozen identity           | All required V8.8 artefacts hash and replay correctly.                                                              | Stop before page processing.                     |
| G2 Roster completeness       | Every candidate page is processed or has an explicit exclusion/failure reason.                                      | Run cannot claim full-population evidence.       |
| G3 Truth firewall            | Zero clean/truth-derived reads before lock.                                                                         | Invalidate entire run.                           |
| G4 Outside-mask immutability | Zero changed pixels outside authoritative mask on every page.                                                       | Invalidate run; fix implementation.              |
| G5 Fallback equivalence      | Frozen incumbent behaviour reproduced exactly outside valid V8.8 interventions.                                     | Invalidate implementation.                       |
| G6 Replay                    | Registered replay sample produces exact required decisions/page bytes.                                              | Invalidate run.                                  |
| G7 Structural safety         | No unacceptable catastrophic notation deletion on paired truth; no obvious material print damage in blinded review. | Do not proceed beyond supervised research.       |
| G8 Cosmetic usefulness       | Residual annotation is low enough for a useful share of pages to be performance-ready.                              | Method may be safe but not operationally useful. |
| G9 Source stability          | No material failure concentration hidden by aggregate results.                                                      | Restrict scope or redesign; do not generalise.   |
| G10 Auditability             | Every changed pixel is traceable to mask authority, operator identity and decision evidence.                        | No pilot/release decision supported.             |

# 14. Required run directory structure

notebook72_runs/&lt;run_id&gt;/  
00_configuration/  
01_operator_verification/  
02_roster_and_lineage/  
03_runtime_outputs/&lt;page&gt;/  
04_runtime_integrity/  
05_prediction_lock/  
06_paired_truth_evaluation/  
07_blinded_review/  
08_combined_results/  
09_replay/  
upload/  
README.txt  
upload_file_list.txt  
upload_manifest.csv  
tables/  
review/  
visuals/  
audit/  
notebook72_upload_bundle.zip  
experiment_manifest.json  
notebook72_verdict.json  
executive_summary.txt

# 15. Required tabular and summary outputs

- runtime_roster.csv and exclusions.csv
- operator_artifact_inventory.csv
- page_integrity.csv
- page_output_hashes.csv
- runtime_access_audit.csv
- results_by_page.csv
- results_by_source_group.csv
- paired_quantitative_results.csv
- performance_usability_reviews.csv
- performance_usability_summary.csv
- failure_and_exception_register.csv
- prediction_lock.json
- experiment_manifest.json
- notebook72_verdict.json with each gate and first failed gate
- executive_summary.txt with cohort counts, integrity result, paired evidence, whole-population usability result and decision recommendation

# 16. Explicit non-objectives

- No YOLO retraining, mask-authority automation or acceptance-threshold development.
- No V8.8 retraining, recalibration, threshold sweep or post-result policy tuning.
- No reconstruction of a guessed V8.8 implementation from fragments if the durable operator cannot be identified.
- No requirement to classify entire mask components semantically when the frozen correction contract is pixel/local evidence based.
- No invented quantitative truth for genuine annotated-only pages.
- No clean-assisted registration during runtime for paired pages.
- No automatic production release or retirement of the reviewer based solely on Notebook 72.

# 17. Final decision logic

| **Observed result**                                                                          | **Required decision**                                                                                       |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Cannot run without clean-derived inputs                                                      | Stop. V8.8 has not demonstrated production-like operation.                                                  |
| Truth firewall, immutability, replay or fallback-equivalence failure                         | Invalidate run and correct implementation before considering quality.                                       |
| Paired metrics acceptable; annotated-only usability poor                                     | Benchmark success but operational failure. Investigate domain shift, mask quality and correction behaviour. |
| Annotated-only usability acceptable; paired scoring exposes concentrated structural failures | Remain supervised and isolate failing sources/mechanisms. Do not generalise.                                |
| Safe but cosmetically weak                                                                   | Retain only for targeted or reviewer-assisted correction; do not call it performance-ready.                 |
| Both cohorts show acceptable safety, usability and source stability                          | Proceed to a controlled supervised pilot decision record. Mask automation remains a separate experiment.    |
| Strong result only for defined sources                                                       | Declare a source-scoped operating envelope and gather fresh evidence before expansion.                      |

# 18. Suggested notebook cell map

1. Title, scope, no-clean and no-promotion warnings
2. Configuration and immutable run creation
3. Operator/artefact resolution and hash verification
4. Frozen replay test
5. Roster enumeration across both source populations
6. Review-audit lineage and authoritative mask resolution
7. Truth-path guard activation and access audit
8. Smoke-test controls
9. Full runtime inference loop
10. Per-page integrity assertions and atomic writes
11. Prediction/output lock creation
12. Post-lock clean resolution for paired pages only
13. Quantitative scoring and source-group summaries
14. Blinded A/B package generation and review form/app
15. Review result import and decoding
16. Combined decision gates and verdict
17. Final Upload Package construction and hash verification

# 19. Known uncertainties and fail-closed assumptions

| **Uncertainty**                                                                                  | **Specification treatment**                                                                                                     |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Exact durable V8.8 operator location                                                             | Mandatory operator-resolution gate. Notebook does not guess.                                                                    |
| Exact V8.8 callable signature                                                                    | Adapter contract above must be implemented and replay-tested before the full run.                                               |
| Completeness of reviewer-authoritative masks                                                     | Roster reports eligible/excluded counts; no draft-mask fallback.                                                                |
| Whether paired annotated pages should use raw source/annotated or an existing aligned derivative | Use the authoritative reviewed lineage recorded by the review audit; do not use clean during runtime to create a new alignment. |
| Final quantitative acceptance thresholds                                                         | Carry forward frozen V8.8/project gates where durable. Unresolved gates are reported, not invented.                             |
| Reviewer count and inter-rater process                                                           | Minimum one blinded review per page; if more than one reviewer is used, preserve independent responses before adjudication.     |
| Runtime and memory                                                                               | Measure per page; resource pressure may change batch size only, never scientific decisions.                                     |

# 20. Project source basis

**Music_Annotation_Project_Reset_and_Notebook_72_Plan.docx** — Defines the reset decision, two-cohort operational trial, truth firewall, review outcomes and decision gates.

**Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx** — Defines annotated-page-only runtime and the staged path from development to supervised operation.

**10_Reconstruction_Benchmark.txt** — Establishes clean-for-evaluation-only and outside-mask immutability.

**V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx** — Defines authoritative-mask intervention, separation of annotation confidence and correction safety, and distinct audit artefacts.

**17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt** — Provides deterministic locking, semantic hashes and separation of observable inference evidence from post-hoc outcomes.

**12_V7_7_Asymmetric_Structural_Risk_Calibration.txt** — Separates structural safety and cosmetic quality; excludes truth from predictors.

**09_Pilot_Proposed_Correction_Routing_Reviewer.txt** — Provides review-lineage resolution, frozen processing, immutability checks and supervised decisions.

**V7_9_x_Research_Findings_and_Next_Steps.docx** — Warns against coarse component semantics and retains local observable PRINT protection with V7.8 fallback.

**Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx** — Defines observable-only three-class inference, fail-closed uncertainty, audit and sealed-truth principles.

**V8_Next_Stage_Full_Pixel_Whole_Page_OOF_Reference.docx** — Defines complete corrected page outputs, baseline equivalence, scoring after inference and whole-page review.

**V8_1_Full_Pixel_Whole_Page_OOF_Notebook_Specification.docx** — Defines prediction locks, durable outputs, replay and strict failure handling.

**V8_2_Development_Whole_Page_Experiment_Decision_Record.docx** — Separates development from freeze/independent governance and standardises final upload packaging.

**28_V8_2_Development_Whole_Page_Experiment.ipynb** — Documents why prior development roster construction excluded annotated-only pages and supplies the whole-page output pattern.

FINAL COMMISSIONING STATEMENT  
Build Notebook 72 as a full-population, clean-free runtime trial of the frozen V8.8 downstream operator. Process reviewed pages from both annotated populations through the same path; require zero clean access before lock, zero outside-mask drift and exact replay; use clean truth only after lock for the paired cohort; subject every page to blinded performance-usability review; and issue a combined supervised-pilot / targeted-use / redesign decision without tuning V8.8 from the result.

**Confidence: 94%.** The runtime/evaluation separation, intervention boundary, locking pattern and two-cohort design are well supported. Confidence is below 100% because the exact durable V8.8 operator location, signature and complete artefact bundle have not yet been established.

**Next question:** Where is the exact frozen V8.8 operator bundle, and can it satisfy the minimum adapter and replay contract in Section 5 without reconstructing historical logic?