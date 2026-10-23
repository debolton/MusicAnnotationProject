**MUSIC ANNOTATION PROJECT**

V8 Next Stage

**Full-Pixel, Whole-Page Out-of-Fold Evaluation**

_Standalone notebook reference and implementation contract_

**CORE DECISION**

The next stage will apply the frozen V8 observed-print decision layer to every eligible held-out pixel inside each authoritative corrected mask, produce actual whole corrected pages, and evaluate the same run quantitatively against clean truth. It will not be another sampled-centre or crop-only review.

| **Document status** | Research design reference; not production authorisation                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **Date**            | 15 October 2026                                                                                           |
| **Stage**           | Full-pixel V8 development OOF                                                                             |
| **Inputs**          | Annotated/aligned page, authoritative corrected mask, frozen V8 fold models and V7.8 correction artefacts |
| **Outputs**         | Whole corrected pages, decision maps, quantitative OOF results and audit artefacts                        |

# 1. Purpose and experiment boundary

This stage converts V8 from a sampled-centre research comparison into a page-scale test of the actual correction contract. The experiment will score all eligible pixels on held-out pages, assemble complete corrected pages, and preserve the quantitative safety evaluation against frozen V7.8.

**What V8 is**

V8 is an observed-print protection layer inside an already authorised removal mask. It decides whether eligible observed dark material should be copied exactly as PRINT. It is not a page-level annotation detector and it is not a generative reconstruction model.

## 1.1 What "run V8 on the whole page" means

| **ANNOTATED / ALIGNED PAGE + AUTHORITATIVE CORRECTED MASK**                  |
| ---------------------------------------------------------------------------- |
| Prepare the frozen dilation-2 intervention mask and V7.8 observable channels |
| Identify every eligible dark pixel not already preserved by V7.8             |
| Apply the held-out V8 fold model and registered decision rule in batches     |
| PRINT → copy observed pixel exactly                                          |
| ANNOTATION → retain V7.8 removal / reconstruction handling                   |
| UNCERTAIN → retain conservative V7.8 behaviour and log the reason            |
| **ASSEMBLE AND SAVE THE COMPLETE V8 CORRECTED PAGE**                         |

↓

↓

↓

↓

↓

↓

↓

## 1.2 Why the output can be described as "V7.8 plus V8 PRINT"

That phrase describes the regression contract, not the required implementation order. The notebook may compute the incumbent page and V8 decisions in one coherent page pipeline. The final V8 page must equal frozen V7.8 everywhere except at pixels where V8 validly emits PRINT, where the exact observed byte is restored.

**Non-negotiable equivalence rule**

If V8 does not emit PRINT at a coordinate, the V8 page must reproduce the frozen V7.8 output byte-for-byte at that coordinate. Pixels outside the authoritative removal mask must remain identical to the observed input page.

# 2. Decision carried forward from V8.0f

The crop-based perceptual review is closed as an exploratory exercise. The recorded outcome is:

- 1 of 8 reviewed cases had a visible difference: the Page039 16-pixel selection visibly formed or completed the edge of a staff line.
- 7 of 8 reviewed cases had no visible difference under the review conditions.
- The pre-corrected comparison imagery was distracting and made the narrow pixel-level judgement difficult.
- No further enlarged A/B crop review is required before moving to the page-scale experiment.

This observation does not redefine uncertain pixels as PRINT and does not alter the registered classifier, calibration or decision thresholds. It motivates whole-page rendering as a better practical diagnostic alongside the registered full-pixel quantitative test.

# 3. Required inputs and frozen elements

| **Input / element**              | **Required status**                                                       | **Role**                                                           |
| -------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Observed page**                | Aligned annotated page for the held-out development page                  | Runtime image evidence and final outside-mask source               |
| **Authoritative corrected mask** | Reviewed mask with exact page-shape agreement                             | Sole boundary of permissible intervention                          |
| **Source-group fold map**        | Frozen and hashed                                                         | Selects the fold model that did not train on the page/source group |
| **V8 fold model/checkpoint**     | Frozen for the completed development experiment                           | Produces OOF class probabilities for every eligible pixel          |
| **Fold-specific calibration**    | Frozen and cross-fitted                                                   | Converts model scores to calibrated decision probabilities         |
| **Registered decision rule**     | Unchanged PRINT / ANNOTATION / UNCERTAIN thresholds and margin            | Routes each eligible pixel without post-hoc tuning                 |
| **Frozen V7.8 engine**           | Exact canonical mask preparation, Telea, background and routing semantics | Comparator and fallback correction path                            |
| **Clean page**                   | Scoring only; prohibited from inference                                   | Creates truth labels and post-hoc safety/benefit metrics           |

**Freeze rule**

Do not refit the classifier, alter channels, change folds, recalibrate probabilities, move thresholds, reinterpret labels, or add a new context scale after viewing full-page results. Any material change is a new experiment version.

# 4. Full-pixel OOF processing contract

1. Validate page, mask, fold and artefact identities; fail closed on any mismatch.
2. Prepare the authoritative dilation-2 removal mask using the frozen correction contract.
3. Rebuild or load the exact frozen V7.8 correction and observable channel arrays.
4. Define eligible candidates using the registered rule: observed dark pixels inside the authorised mask that V7.8 does not already preserve.
5. Extract the registered 33 × 33 spatial tensor for every eligible coordinate.
6. Score candidates in batches using the model belonging to the page's held-out source-group fold.
7. Apply the frozen calibration and PRINT / ANNOTATION / UNCERTAIN decision rule.
8. Apply the no-singleton and connected-run requirements exactly as registered.
9. Assemble the V8 corrected page: copy exact observed bytes at valid PRINT coordinates; retain V7.8 output everywhere else.
10. Assert zero outside-mask drift and zero non-PRINT regression from V7.8.
11. Only after outputs and decisions are written, use clean truth for scoring.
12. Write page, source-group and pooled summaries plus visual audit artefacts.

## 4.1 Computational implementation

The notebook should not call the CNN once per pixel. Eligible tensors should be extracted and scored in bounded batches. Batch size may be selected to fit the available runtime memory, but batching must not change scores or decisions. A deterministic replay check must establish equality between repeated runs on a frozen sample.

A GPU runtime is the practical default for CNN scoring. The experiment record currently contains no measured full-page runtime benchmark, so processing cost must be reported from the run rather than assumed in advance.

# 5. Required page outputs

| **Artefact**                   | **Required content**                                             | **Purpose**                       |
| ------------------------------ | ---------------------------------------------------------------- | --------------------------------- |
| **observed_page.png**          | Aligned annotated input page                                     | Runtime source and visual context |
| **authoritative_mask.png**     | Undilated corrected mask and prepared dilation-2 mask            | Intervention authority and audit  |
| **v7_8_corrected.png**         | Frozen incumbent output                                          | Comparator and replay reference   |
| **v8_corrected.png**           | Complete V8 page after all eligible decisions                    | Primary practical output          |
| **v8_decision_map.png / .npz** | PRINT, ANNOTATION and UNCERTAIN coordinates                      | Decision audit                    |
| **v8_vs_v7_8_difference.png**  | Only pixels where final pages differ                             | Difference localisation           |
| **probability arrays**         | Calibrated class probabilities at every eligible coordinate      | Threshold and uncertainty audit   |
| **page_metrics.json / .csv**   | Counts, safety, benefit, utility, coverage and integrity results | Page-level evaluation             |
| **run manifest**               | Hashes, model/fold IDs, settings, software and timing            | Reproducibility                   |

# 6. Quantitative evaluation

The whole-page rendering is a diagnostic addition. The formal evaluation remains the registered V8 development OOF evaluation against frozen V7.8.

| **Metric**                  | **Definition in this stage**                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------- |
| **Extra true print**        | V8 PRINT pixels not preserved by V7.8 whose clean-truth label is PRINT.                         |
| **Extra false annotation**  | V8 PRINT pixels not preserved by V7.8 whose clean-truth label is ANNOTATION.                    |
| **False / true ratio**      | Extra false annotation divided by extra true print, with the registered denominator guard.      |
| **False-preservation rate** | Extra false annotation divided by all challenger-added preserved pixels.                        |
| **Net safety utility**      | Extra true print − 5 × extra false annotation − 500 × catastrophic components.                  |
| **Catastrophic component**  | Registered concentration rule applied after full-pixel decisions are assembled spatially.       |
| **Eligible PRINT coverage** | Eligible candidates emitted as PRINT divided by all eligible candidates.                        |
| **UNCERTAIN rate**          | Eligible candidates emitted as UNCERTAIN divided by all eligible candidates.                    |
| **Page coverage**           | Pages meeting the registered true-print benefit and catastrophic-safety definition.             |
| **Integrity**               | Outside-mask drift and any difference from V7.8 where V8 did not emit PRINT; both must be zero. |

## 6.1 Required stratification and uncertainty

- Report pooled, page-level and source-group results.
- Report sparse-print and hard-negative behaviour separately.
- Use the registered page-clustered bootstrap and Wilson bounds rather than treating pixels as independent observations.
- Scan connected spatial concentrations after full-pixel assembly; sampled-centre success cannot substitute for this check.

# 7. Whole-page visual review contract

Visual review should judge the actual generated pages, not oversized diagnostic composites. For each reviewed page, the default view should present the frozen V7.8 corrected page and the V8 corrected page at the same whole-page scale. The observed input, mask, difference map and V8 decision overlay remain optional diagnostics.

**Primary human questions**

1\. Does V8 visibly preserve notation that V7.8 damages?  
2\. Does V8 visibly retain annotation that V7.8 removes?  
3\. Is the page-level difference visually irrelevant?

## 7.1 Review queue

- Every catastrophic candidate, if any.
- The most false-heavy pages/components under clean-truth scoring.
- The most beneficial pages/components.
- The most uncertain and highest-disagreement regions.
- Representative pages with no visible difference, to test whether quantitative gains are practically material.

## 7.2 Presentation rules

- Do not show pre-corrected imagery as the main comparison when it distracts from the correction outcome.
- Keep V7.8 and V8 at identical scale, crop and contrast.
- Use difference maps only to locate changes, not as the primary quality judgement.
- Keep clean truth out of the initial blinded quality judgement; use it afterwards for diagnosis.

# 8. Integrity and fail-closed requirements

- Exact input roster, shape and filename-stem agreement.
- Authoritative corrected mask is the only intervention boundary.
- No clean or truth-derived channel enters inference.
- Correct held-out fold model is used for every page.
- No page or source group is scored by a model trained on that page/source group.
- Frozen channels, calibration, thresholds, margins and connected-run rules are unchanged.
- Zero changed pixels outside the authoritative mask.
- Zero difference from V7.8 where V8 does not validly assert PRINT.
- All probability, decision and output arrays are finite, shape-compatible and content-addressed.
- Repeated replay on a frozen sample is exact.
- Failures are written to audit artefacts before the notebook stops.

# 9. Decision logic for the next stage

| **Outcome**                                                                 | **Decision**                                                                                                                            |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Any integrity or OOF leakage failure**                                    | Invalidate the run. Correct implementation or data handling before interpreting performance.                                            |
| **Full-pixel safety gates fail**                                            | Retain frozen V7.8. Do not freeze or promote V8 at this operating point.                                                                |
| **Safety passes but benefit / coverage is insufficient**                    | Retain V8 as a low-coverage research baseline. Do not weaken safety thresholds.                                                         |
| **Quantitative gains exist but whole-page effect is visually negligible**   | Record a valid but practically limited improvement; decide whether further representation work is justified before independent testing. |
| **V8 visibly preserves notation but also retains objectionable annotation** | Treat as a safety/selectivity failure or unresolved failure mechanism, according to truth scoring and registered gates.                 |
| **All registered development gates pass after full-pixel scoring**          | Freeze one challenger bundle and commission the untouched independent promotion test. This does not authorise production use.           |

# 10. What this stage does not test

- It does not validate raw YOLO or detector-generated masks as authoritative.
- It does not replace the reviewed corrected mask.
- It does not alter or replace the V7.8 reconstruction engine.
- It does not train on the frozen V7.8 independent set.
- It does not use clean truth during page inference.
- It does not authorise automatic acceptance or deployment.
- It does not test a new context scale, centre-region loss, architecture search or post-result threshold change.

# 11. Suggested notebook structure

13. Configuration, paths and read-only controls
14. Load and verify pre-registration, folds, V8 fold bundles and V7.8 canonical implementation
15. Input roster and source-group leakage audit
16. Build / load frozen V7.8 page arrays and authoritative masks
17. Enumerate all eligible pixels
18. Batched fold-specific V8 full-pixel inference
19. Apply calibration, three-class rule and connected-run constraints
20. Assemble V8 whole corrected pages and decision maps
21. Run immutability and V7.8 equivalence assertions
22. Lock predictions and outputs before truth scoring
23. Score clean-truth outcomes and full-pixel registered metrics
24. Generate page/source-group tables and visual review queue
25. Write manifest, verdict and concise executive summary

# 12. Source basis

This reference consolidates the next-stage approach from the following project records:

- **Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx** — three-class target, eligible population, thresholds, metrics, gates, equivalence and audit contract.
- **V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx** — matched scalar-versus-spatial design, 33 × 33 channels, source-group OOF and safe-coverage objective.
- **Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx** — explicit full-pixel V8 OOF stage and complete runtime correction flow.
- **V7_9_x_Research_Findings_and_Next_Steps.docx** — learned observed-print hypothesis and PRINT / ANNOTATION / UNCERTAIN treatment.
- **V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx** — authoritative-mask boundary, exact preservation and outside-mask immutability.
- **17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt** — canonical V7.8 page reconstruction arrays, corrected masks and deterministic identity controls.
- **18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt** — stage-by-stage fail-closed equivalence pattern.
- **09_Pilot_Proposed_Correction_Routing_Reviewer.txt** — actual-page proposal review and separation of page imagery from optional overlays.

# 13. Final decision statement

**DECISION**

Build the next notebook as a full-pixel, source-group out-of-fold V8 evaluation that produces actual whole corrected pages as first-class artefacts. Use the annotated/aligned page and authoritative corrected mask as runtime inputs; apply V8 to every eligible held-out pixel; preserve exact observed bytes for valid PRINT decisions; retain frozen V7.8 handling for ANNOTATION and UNCERTAIN; and evaluate the resulting pages quantitatively and visually without changing the frozen experiment.

**Confidence: 97%.** The page-scale stage, fallback semantics and integrity requirements are explicit in the existing project records. The main implementation uncertainty is measured runtime and memory usage for full-pixel CNN scoring, which must be recorded empirically during the notebook run rather than assumed.

**Next question:** What exact frozen V8 fold artefacts and paths will the full-pixel notebook load, and are all replay/calibration objects already durable and hash-addressable?