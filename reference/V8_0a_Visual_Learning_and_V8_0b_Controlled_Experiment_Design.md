V8.0a Visual Learning and V8.0b Controlled Experiment Design

**Observed-print classifier challenger | 14 October 2026**

**CORE DECISION  
V8.0b will isolate raw spatial topology as the experimental variable. It will rerun the V8.0a scalar logistic baseline and a compact spatial CNN on the same labels, samples, source-group folds, calibration method, thresholds, gates and V7.8 comparator. No independent data or production promotion is involved.**

# 1. What the V8.0a visual audit taught us

- Beneficial selections are not one morphology. The rendered examples include narrow vertical printed strokes crossing staff lines, dense horizontal printed regions intersected by annotation, and small printed fragments inside larger masked areas.
- The two Page036 false PRINT selections occur where a broad looping annotation crosses or closely approaches printed staff and symbol structure. True PRINT, false PRINT and UNCERTAIN pixels are interleaved within a few pixels of one another.
- The central failure is therefore not adequately described as darkness, orientation or continuation alone. The classifier must infer which local structure owns the centre pixel when annotation and print overlap.
- The V8.0a logistic model collapsed spatial evidence into local summary statistics. It could detect some recoverable print, but its extreme PRINT tail included both correct and confidently wrong selections. Tightening the same probability threshold would not separate errors that already received saturated scores.

# 2. Interpretation and remaining uncertainty

**Leading interpretation.** The Page036 failures are genuine visual confusions caused primarily by insufficient spatial/topological representation, rather than obvious clean/observed misregistration. The clean and observed staff geometry appears aligned at the scale relevant to the selected pixels.

**Residual uncertainty.** A tolerance-sensitive label audit remains necessary. V8.0b will report whether a target remains stable under the existing tolerance and adjacent sensitivity checks, but this audit will not be used to opportunistically relabel V8.0a outcomes after model inspection.

# 3. Why V8.0b must be a controlled comparison

A broad redesign would not answer whether raw spatial topology caused any improvement. Adding multi-scale context, centre-region supervision, new loss weights, new labels, new thresholds and new folds simultaneously would make attribution impossible. V8.0b therefore uses a paired two-arm experiment.

| **Element**                   | **Control arm: V8.0a-R**                          | **Treatment arm: V8.0b-T**                                        |
| ----------------------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| Target                        | Same PRINT / ANNOTATION labels                    | Same labels                                                       |
| Eligible candidate population | Same rule                                         | Same rule                                                         |
| Development samples           | Same sampled centre pixels                        | Exactly the same sampled centre pixels                            |
| Grouping and folds            | New authoritative source-group folds              | Exactly the same folds                                            |
| Input representation          | V8.0a scalar observable features                  | Raw 33×33 spatial tensor of the same observable evidence families |
| Estimator                     | Regularised logistic regression                   | Compact CNN                                                       |
| Calibration                   | Cross-fitted isotonic                             | Same method                                                       |
| Decision rule                 | Registered PRINT/ANNOTATION thresholds and margin | Same rule                                                         |
| Safety gates                  | Commissioning v1.1 gates                          | Same gates                                                        |
| Comparator                    | Frozen V7.8                                       | Frozen V7.8                                                       |

# 4. Experimental variable

The only intended causal difference is representation plus the minimum estimator required to consume that representation:

- Control: scalar summaries derived from observed intensity, mask geometry, local dark/edge density, V7.8 support fractions and route fractions, fitted by logistic regression.
- Treatment: spatially aligned 33×33 channels containing observed greyscale, background-normalised greyscale, authorised mask, mask distance, horizontal/vertical/compact support, preserve/remove/ambiguous routes and observed edges, fitted by a compact CNN.
- No multi-scale branch, centre-region output, architecture sweep, probability-threshold movement, hard-negative injection or manual Page036 override is permitted in V8.0b. Those remain possible later ablations only if V8.0b shows that spatial topology adds value.

# 5. Fold and leakage controls

- V8.0b fails closed if development_score_manifest.csv is missing, contains duplicate pages, has missing source_group values, or does not match the frozen V7.5 development roster.
- One deterministic source-group fold map is generated and hashed. Both arms use that exact map.
- No page or source group may occur in both training and test portions of a fold.
- Clean truth is used only to construct development labels and score frozen OOF predictions. It is never an inference channel.
- Raw page coordinates, page identity and source-group identity are excluded from both estimators.
- Frozen V7.8 independent data remains outside training and threshold selection.

# 6. Primary comparison: safe coverage frontier

The primary result is not ordinary classification accuracy. For each arm, the notebook will sweep a predeclared common probability grid over cross-fitted OOF probabilities and calculate:

- additional true-print pixels preserved beyond V7.8
- additional false-annotation pixels preserved beyond V7.8
- false-per-true ratio
- false-preservation rate and Wilson upper bound
- net safety utility
- PRINT coverage and UNCERTAIN rate
- benefiting pages and source groups
- catastrophic spatial concentrations

The treatment demonstrates an improved safe coverage frontier only if, at the same registered safety constraint, the treatment attains higher eligible PRINT coverage than the control without introducing a new catastrophic failure or source-group instability. A higher AUC, lower training loss or larger raw selection count is not sufficient.

# 7. Paired decision logic

- If both arms fail safety, retain V7.8 and record whether topology improved ranking without demonstrating an acceptable operating point.
- If the CNN passes safety but does not exceed control coverage, raw topology has not demonstrated practical frontier expansion.
- If the CNN increases coverage at the same safety bound and required stability gates pass, freeze neither model automatically; first complete the remaining visual, sparse/hard-negative and independent-data prerequisites.
- If results are concentrated in one score/source group, do not generalise the result across the development population.

# 8. Deferred changes

The following are intentionally deferred so V8.0b remains interpretable: multi-scale 17/65/129 branches; centre-region segmentation loss; asymmetric loss-weight searches; architecture searches; new Page036-derived training labels; threshold optimisation after outcome inspection; and inclusion of any independent cohort.

# 9. Required V8.0b outputs

- experiment_manifest.json and fold_map.csv with hashes
- paired_sample_manifest.csv proving identical centres and labels
- source_group_split_audit.csv
- control_oof_predictions.csv and topology_oof_predictions.csv
- paired_frontier_comparison.csv
- registered_operating_point_comparison.json
- results_by_page.csv and results_by_source_group.csv
- catastrophic_components.csv
- visual audit queues for treatment wins, treatment regressions and disagreement cases
- model/checkpoint artefacts only if all freeze prerequisites are separately completed

# 10. Source basis

- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx: controlled-mask boundary, separate safety axis, observable geometry evidence and fixed safety posture.
- V7_9_x_Research_Findings_and_Next_Steps.docx: learned local observed-print classifier, truth-free inputs, grouped OOF evaluation, frozen V7.8 comparator and no further deterministic threshold engineering.
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx: PRINT / ANNOTATION / UNCERTAIN target, safety-first gates and expansion of safe coverage through later challengers.
- 17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt and 18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt: canonical frozen arrays, source-group manifest precedent, semantic identity and fail-closed equivalence.
- 12_V7_7_Asymmetric_Structural_Risk_Calibration.txt and V7_7F design record: grouped OOF modelling, cross-fitted calibration, Wilson bounds and prohibition on post-unblinding tuning.
- Production_Retraining_Policy_YOLO_and_Structural_Safety.docx: frozen incumbent, event-driven challengers, explicit evidence roles and no automatic promotion.

# 11. Decision and confidence

**Decision:** Build V8.0b as a paired scalar-versus-spatial representation experiment. Do not call V8.0b a multi-scale production candidate. Its job is to establish whether raw local topology shifts the safe coverage frontier under otherwise matched conditions.

**Confidence: 93%.** The visual evidence strongly supports a missing spatial-ownership signal, and the controlled design can test that hypothesis. Uncertainty remains over whether 33×33 context is sufficient; larger context is deliberately deferred to preserve causal interpretability.

**Next question:** If 33×33 spatial topology improves ranking but not the registered safe operating point, what single additional context scale should be tested next without changing the loss or decision rule?