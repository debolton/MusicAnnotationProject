**MUSIC ANNOTATION PROJECT**

V8.2 Development Whole-Page Experiment

_Decision record: simplifying the next observed-print development stage after the V8.1 artefact-resolution detour_

16 October 2026 | Research development record | Not production authorisation

**DECISION: retire the current V8.1 implementation as a forensic/frozen-artifact prototype and commission V8.2 as a fresh, development-only whole-page experiment. V8.2 will retrain the unchanged 33×33 observed-print model under source-group OOF, fit cross-fitted isotonic calibration within the run, persist every learned artefact, score every eligible held-out pixel, and produce actual V7.8-versus-V8.2 whole-page outputs. Promotion-grade locking is deferred until there is a result worth freezing.**

# 1. Executive summary

The project is not scientifically blocked. The recent difficulty arose because V8.1 combined two different stages: a development whole-page experiment and the later creation/verification of a promotion-grade frozen bundle. This caused the notebook to require historical artefacts that V8.0d had not actually persisted, then to fail on file-level identities after legitimate project housekeeping such as renamed manifest columns.

The corrected approach is to separate the stages again. V8.2 will answer the development question directly: does the 33×33 learned observed-print layer improve whole corrected pages and quantitative safety/benefit outcomes over V7.8 when evaluated out of fold? If V8.2 is promising, a separate freeze step will then package and lock the selected model, calibrator, schema, policy and replay record before any untouched independent promotion test.

**Source basis:** _The project pathway explicitly separates full-pixel development OOF from challenger freezing and then from untouched independent promotion testing. The commissioning specification likewise distinguishes development OOF from the frozen-bundle deliverable._

# 2. Why call the clean-slate experiment V8.2?

Using V8.2 is appropriate as a project-versioning decision, provided the notebook is clearly described as a new development experiment rather than a continuation that claims byte-for-byte identity with the abandoned V8.1 implementation. The existing V8.1 specification itself states that a material change or a separately pre-registered later challenger should receive a new version.

- V8.1 remains useful as an archival record of the stricter frozen-artifact approach and the problems it exposed.
- V8.2 becomes the clean development lineage, with its own input snapshot, fold map, checkpoints, calibrators, channel schema, predictions and whole-page results.
- No V8.2 result is automatically a frozen challenger or production candidate. Freeze and independent testing remain later gates.

# 3. What happened during the V8.1 attempt

| **Issue**                             | **Interpretation**                                                                                                                                       |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manifest hash mismatch                | The project manifest had later been edited (including column-heading changes), so byte-level SHA-256 identity no longer represented the historical file. |
| Missing schema artefact               | The historical V8.0d run recorded the channel definition in code/manifest context rather than a standalone reusable schema bundle.                       |
| Multiple replay/checkpoint candidates | Broad artefact discovery mixed historical runs/replicas and treated distinct experiment records as if they should be byte-identical.                     |
| Missing fold calibrators              | V8.0d fitted isotonic calibration in memory and did not serialise standalone calibrator files.                                                           |
| Calibrator reconstruction mismatch    | Attempting to reconstruct calibration from saved all-eligible outputs did not reproduce the original stable-score calibration path exactly.              |
| Repeated upstream reruns              | Rerunning earlier notebooks while suppressing checks created a moving target for provenance instead of answering the whole-page scientific question.     |

# 4. What V8.2 should keep

**Simplification does not mean removing the controls that make the experiment interpretable. Keep the controls that protect the scientific question; defer the controls whose main purpose is independent-test release governance.**

**Source-group OOF separation:** Every development page is scored only by a model that excluded its source group. This remains the core leakage control.

**Truth firewall:** Clean pages may create labels and post-hoc scores but do not enter CNN inference channels.

**Authoritative intervention mask:** V8.2 acts only inside the reviewed/corrected mask and its frozen dilation rule.

**V7.8 fallback identity:** The V8.2 page equals V7.8 everywhere except valid PRINT coordinates, where the observed byte is copied exactly.

**Outside-mask immutability:** No pixel outside the authorised mask may change.

**Registered 33×33 representation and decision rule:** Keep the selected spatial representation and existing decision thresholds/margin for this development test rather than introducing another architecture search.

**Source basis:** _These controls are explicit in the V8.0 controlled design, V8 full-pixel reference and observed-print commissioning specification._

# 5. What V8.2 should deliberately defer

- Historical artefact_inventory.json reconstruction.
- Byte-level replay against old V8.0b/V8.0c/V8.0d manifest files that have since changed.
- Requiring historical standalone calibrator files that were never serialised.
- Two-consecutive-lock calculations before the development experiment can even run.
- Independent-test release locking, policy IDs and complete promotion bundle packaging.
- Claims that the V8.2 result is deployment-ready or equivalent to an untouched independent validation.

These controls are not discarded. They move to the freeze/independent-test boundary, where they serve their intended purpose: ensuring that a selected challenger cannot change after development outcomes are known.

# 6. Proposed V8.2 experiment contract

| **Stage**                                   | **V8.2 action**                                                                                                                                                   |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1\. Snapshot inputs**                     | Copy the current development manifest into the run directory before modelling. Save the exact page roster and source-group fold assignment used by this run.      |
| **2\. Build unchanged observable channels** | Use the 33×33 spatial channels carried forward from V8.0: observed, normalised, mask, depth, horizontal, vertical, compact, preserve, remove, ambiguous and edge. |
| **3\. Train OOF fold models**               | Train the unchanged compact CNN under deterministic source-group folds, on GPU.                                                                                   |
| **4\. Fit calibration within the run**      | Use the same cross-fitted isotonic approach as the V8 development lineage, but persist each fitted calibrator immediately.                                        |
| **5\. Score every eligible held-out pixel** | Batch full-pixel CNN inference for the page/source-group held out from the corresponding fold model.                                                              |
| **6\. Apply decisions globally per page**   | Apply calibrated PRINT / ANNOTATION / UNCERTAIN rules and the no-singleton/connected-run rule after page-level provisional predictions are assembled.             |
| **7\. Assemble whole corrected pages**      | Start from V7.8; restore the exact observed byte only at valid V8.2 PRINT coordinates.                                                                            |
| **8\. Score and review**                    | After predictions/pages are written, evaluate against clean truth; report pooled, page and source-group results and whole-page V7.8-versus-V8.2 comparisons.      |

# 7. Minimum durable outputs from every V8.2 run

- inputs/development_score_manifest.csv (run snapshot)
- fold_map.csv
- channel_schema.json
- models/fold1.pt through foldN.pt
- calibrators/fold1.joblib through foldN.joblib
- stable_sample_manifest.csv
- eligible_pixel_manifest.parquet
- all_eligible_oof_predictions.parquet
- results_by_page.csv
- results_by_source_group.csv
- selected_components.csv / catastrophic_components.csv
- whole_pages/&lt;page&gt;/observed.png, v7_8.png, v8_2.png, difference.png, decision_map.png
- experiment_identity.json including seed, constants and software versions
- executive_summary.txt

The purpose is not to create a heavyweight lock at every exploratory step. It is to ensure that the experiment is self-contained enough that we never again have to infer which manifest, calibrator or fold artefact a successful run used.

# 8. Hard-negative handling

For V8.2, keep hard-negative/sparse-print challenge material role-separated from the principal modelling cohort unless the development design is explicitly amended before outcomes are inspected. The project record already treats hard-negative behaviour as important validation evidence and warns against weakening safety to increase coverage.

Practical default: run the source-group OOF development experiment on the declared modelling cohort, then report hard-negative behaviour in a separate challenge section. Do not use those challenge outcomes to retune thresholds inside the same run.

**Source basis:** _The V7.9.x findings identify sparse-print/hard-negative behaviour as a major failure mode, while the commissioning specification separately requires sparse/hard-negative reporting and independent role discipline._

# 9. When the heavy locking becomes necessary

The strict freeze discipline remains essential once the project decides, "this is the challenger we want to expose to untouched independent evidence." At that boundary, freeze the selected model, calibrator, ordered schema, thresholds/policy, implementation identity, software environment and replay pack, and do not refit or recalibrate against independent outcomes.

**Development first; freeze second; independent test third. The mistake in V8.1 was to make the development notebook prove the complete freeze bundle before it could answer the development question.**

**Source basis:** _The staged production pathway explicitly orders full-pixel development OOF before challenger freeze and untouched independent promotion testing. The independent-policy design also prohibits refitting/recalibration after independent outcomes are exposed._

# 10. Working practices to avoid another archaeology loop

**Do not overwrite historical manifests:** Create a new manifest version when headings, rows or semantics change. A completed run keeps its own input snapshot.

**Persist every learned object:** Models, isotonic calibrators, scalers/imputers and learned mappings should be saved when created, even for research runs.

**Prefer uploaded/pasted notebook code for review:** The project launcher can point to GitHub, while execution occurs in Colab and data remain in Drive. For design/debugging discussions, sharing the actual notebook content or file gives a stable review target.

**Stop patching after repeated provenance failures:** If a development notebook encounters several incompatible historical-artifact assumptions, create a new versioned development experiment rather than repeatedly suppressing gates.

**Separate scientific gates from release gates:** Leakage, truth firewall and immutability belong in development. Complete frozen-bundle/replay identity belongs at freeze and independent promotion boundaries.

# 11. Standard final upload package requirement for future Notebooks

**All future Music Annotation Project notebooks should conclude with a standard Final Upload Package stage. The purpose is to ensure review artefacts, decision evidence, manifests and reproducibility records are packaged consistently and can be inspected without requiring reviewers to reconstruct notebook outputs manually. This requirement applies equally to development experiments, visual-review tools, counterfactual audits, threshold sweeps and future promotion-candidate notebooks.**

**Purpose:** The project increasingly depends upon durable artefact preservation, reproducible review workflows, manifest-driven provenance and whole-page inspection packages. Future notebooks should therefore produce a standardised review bundle as part of normal execution rather than leaving reviewers to locate outputs across multiple directories. This approach is consistent with the project's existing emphasis on persisted artefacts, manifests, calibration records, page-level outputs and reproducible experiment packaging.

**Standard Directory Layout**

Future notebooks should standardise on a review-oriented structure similar to:

upload/

├── README.txt

├── upload_file_list.txt

├── upload_manifest.csv

├── tables/

├── review/

├── visuals/

├── audit/

└── notebookXX_upload_bundle.zip

Notebook-specific subdirectories may be added where justified, but the package structure should remain recognisable across project stages.

**Review Artefacts versus Working Artefacts**

Not every generated file should automatically be copied into the upload package. Some notebooks may generate:

large intermediate arrays

cached processing artefacts

debug outputs

temporary datasets

millions of rows of diagnostic data

Blindly packaging all generated files would create unnecessarily large review bundles and increase storage, transfer and review burden.

Instead, each notebook should explicitly define a curated upload list, for example:

UPLOAD_ARTEFACTS = \[...\]

The upload package should contain only materials required for review, interpretation, audit, reproduction and decision-making, such as:

executive summaries

decision tables

evaluation results

review images

contact sheets

comparison outputs

audit reports

manifests

provenance records

Heavyweight intermediate artefacts should remain within the run directory and continue to be referenced through manifests and provenance records where necessary.

**Minimum Packaging Controls**

The Final Upload Package stage should fail closed if:

- a required review artefact is missing;
- a listed file cannot be packaged;
- manifest generation fails;
- bundle creation fails; or
- required hash verification cannot be completed.

The resulting package should be sufficient for an independent reviewer to understand the notebook outcome, inspect supporting evidence, and locate the underlying run outputs without reconstructing notebook execution.

# 12. Status of the abandoned V8.1 notebook

Retain the current V8.1 notebook and specification as a forensic design record. It captured useful release-grade requirements and exposed real archival weaknesses, especially the absence of durable calibrator/schema packaging. It should not be treated as the executable route for the next development experiment.

Suggested archival label: 27_V8_1_Full_Pixel_Whole_Page_OOF_Evaluation_FORENSIC.ipynb. The new notebook should be named along the lines of 28_V8_2_Development_Whole_Page_Experiment.ipynb.

# 13. Decision statement

**V8.2 will be a fresh development-only whole-page experiment. It will retrain the unchanged 33×33 observed-print model under source-group OOF, fit and persist cross-fitted isotonic calibrators, score every eligible held-out pixel, preserve V7.8 everywhere except valid PRINT coordinates, produce complete V7.8-versus-V8.2 pages, and evaluate the registered development safety/benefit evidence. Historical V8.0 artefact-lock reconstruction is not a prerequisite. If the V8.2 result is worth progressing, a separate freeze stage will create the promotion-grade immutable bundle before untouched independent testing.**

Confidence: 96%. The stage separation is explicit in the project pathway and commissioning design. Residual uncertainty concerns the exact final V8.2 cohort definition and how the existing hard-negative pages should be role-labelled in the new run; those should be declared before V8.2 outcomes are inspected.

**Next question:** What exact cohort and hard-negative role should be declared in V8.2 before we write the new notebook?

# 14. Project source basis

- V8_Next_Stage_Full_Pixel_Whole_Page_OOF_Reference.docx — page-scale development objective, V7.8 fallback identity, full-pixel outputs and whole-page review.
- Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx — staged order: development OOF → freeze challenger → untouched independent test → supervised/shadow integration.
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx — target, truth firewall, development/independent separation, calibration and commissioning deliverables.
- V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx — 33×33 compact CNN, source-group OOF, matched calibration/thresholds and safe-coverage objective.
- V7_9_x_Research_Findings_and_Next_Steps.docx — rationale for moving from deterministic gates to a learned observed-print classifier and for retaining V7.8 as incumbent.
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx — intervention-mask boundary, exact preservation and outside-mask immutability.
- V8_1_Full_Pixel_Whole_Page_OOF_Notebook_Specification.docx — stricter forensic/frozen-artifact requirements retained as future freeze-stage guidance.