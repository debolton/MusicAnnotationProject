**MUSIC ANNOTATION PROJECT**

Observed-Print Classifier

**Pre-Registered Commissioning Specification**

Safety-first research protocol | Version 1.0 | 14 October 2026

**Commissioning decision  
**Commission a learned local classifier only as a challenger to frozen V7.8. The classifier may preserve an observed pixel only when it meets a pre-registered PRINT rule. ANNOTATION and UNCERTAIN remain on the V7.8 removal/ambiguous/review path. No result authorises automatic production promotion.

| **Field**                | **Pre-registered value**                                                            |
| ------------------------ | ----------------------------------------------------------------------------------- |
| Incumbent                | Frozen V7.8 geometry_t80 reconstruction and routing                                 |
| Research unit            | Observed dark pixel with a fixed local patch, inside the authoritative removal mask |
| Outputs                  | PRINT / ANNOTATION / UNCERTAIN                                                      |
| Primary safety endpoint  | False annotation preserved beyond V7.8                                              |
| Primary benefit endpoint | Genuine printed ink preserved beyond V7.8                                           |
| Validation design        | Page- and source-group-separated development; one untouched independent cohort      |
| Promotion posture        | Fail closed; no automatic promotion                                                 |

Status: pre-registration proposal. The numerical gates below are commissioning choices, not previously established project facts.

# Document map

| **Section** | **Purpose**                            |
| ----------- | -------------------------------------- |
| 1           | Commissioning decision and scope       |
| 2           | Scientific question and estimand       |
| 3           | Target and label construction          |
| 4           | Input allowlist and predictor firewall |
| 5           | UNCERTAIN rule                         |
| 6           | Dataset, split and holdout controls    |
| 7           | Model-development protocol             |
| 8           | Metrics and error taxonomy             |
| 9           | Development promotion gates            |
| 10          | Independent promotion gates            |
| 11          | Integrity, reproducibility and audit   |
| 12          | Deliverables and go/no-go decisions    |
| 13          | Deviations and change control          |
| 14          | Source basis and confidence            |

# 1. Commissioning decision and scope

**Scope boundary  
**The experiment tests whether a learned, local, observable-only classifier can identify additional genuine printed ink inside an already authorised removal mask more selectively than deterministic V7.9.x rules. It is not a page-level annotation detector, not generative reconstruction and not a replacement for the authoritative mask.

- Frozen comparator: V7.8 geometry_t80, including its preserve, remove and ambiguous routes.
- Permitted change: add a local PRINT decision layer for observed pixels or small patches inside the authorised mask.
- Prohibited change: alter V7.8 reconstruction, mask dilation, routing semantics, structural policy, YOLO detector, or outside-mask pixels.
- Decision unit: the central observed dark pixel; the model may use a fixed surrounding patch but must emit a decision aligned to the central pixel.
- Operational default: if any prerequisite, input, integrity check, probability rule or coverage rule fails, emit UNCERTAIN and retain V7.8 behaviour.

# 2. Scientific question and estimand

Primary question: among observed dark pixels inside the authoritative removal mask that frozen V7.8 does not preserve, can a frozen learned classifier preserve additional genuine printed ink while keeping additional annotation preservation below the pre-registered safety limits?

| **Item**                 | **Definition**                                                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Population               | Eligible observed dark pixels inside the authorised, dilation-2 removal mask on registered music pages.                                      |
| Intervention             | Preserve the exact observed central pixel when the classifier emits PRINT.                                                                   |
| Comparator               | Frozen V7.8 output for the same pixel.                                                                                                       |
| Primary safety estimand  | Number and rate of extra preserved pixels whose clean-truth label is annotation/non-print.                                                   |
| Primary benefit estimand | Number and rate of extra preserved pixels whose clean-truth label is genuine printed ink.                                                    |
| Unit for uncertainty     | Page-clustered bootstrap and source-group stratification; pixel counts alone are not treated as independent observations.                    |
| Direction of utility     | Benefit only from additional true print preserved; penalties for false annotation and catastrophic concentrations are asymmetric and larger. |

# 3. Target and label construction

## 3.1 Three-class target

| **Class**  | **Truth-side definition**                                                                                                                           | **Inference treatment**                                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| PRINT      | The central observed dark pixel corresponds to genuine printed ink in the registered clean reference, within the registration-tolerance rule below. | Copy the observed pixel byte-for-byte, but only if the model also passes the PRINT decision rule.                             |
| ANNOTATION | The central observed dark pixel does not correspond to genuine printed ink and is attributable to annotation or annotation-related dark material.   | Do not preserve through the challenger; retain V7.8 remove/ambiguous handling.                                                |
| UNCERTAIN  | Truth cannot be assigned reliably because of registration ambiguity, mixed evidence, unstable neighbourhood, or excluded label conditions.          | Exclude from supervised loss unless an uncertainty-aware method explicitly uses it; always route to V7.8/review at inference. |

## 3.2 Label-generation rule

1. Start only from registered development pairs whose registration and mask integrity checks pass.
2. Restrict candidates to pixels inside the authoritative dilation-2 removal mask, with observed greyscale intensity < 200, that frozen V7.8 does not already preserve.
3. Construct a clean-truth printed-ink map using the frozen project method used for structural scoring; the exact implementation and hash must be locked before labels are generated.
4. Assign PRINT only when clean printed ink is present within a 1-pixel tolerance of the central coordinate and local registration confidence passes all label-quality checks.
5. Assign ANNOTATION only when no clean printed ink occurs within that tolerance and the pixel is not in an excluded/ambiguous zone.
6. Assign UNCERTAIN for all other cases, including mixed registration evidence, page-border effects, unreadable pixels, mask/shape mismatch, or disagreement among pre-registered label-quality checks.

**No truth leakage  
**Clean registered pages and all truth-derived maps are used only to construct development labels and score frozen predictions. They are prohibited from model inputs, preprocessing fitted at inference, routing or reconstruction.

# 4. Input allowlist and predictor firewall

The implementation must reject, rather than silently ignore, any predictor not in the allowlist. The ordered schema is frozen before model fitting and hashed in the run manifest.

| **Input family**                                     | **Allowed representation**                                                              | **Constraints**                                                                                                        |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Local observed patch                                 | Raw greyscale patch centred on the target pixel.                                        | Primary patch: 33 x 33 pixels. A pre-registered ablation may test 17 x 17 and 65 x 65; no post-result patch selection. |
| Authorised mask geometry                             | Binary mask patch; signed/unsigned distance to mask boundary; central-pixel mask depth. | Derived only from authoritative mask; no clean truth.                                                                  |
| V7.8 support                                         | Horizontal, vertical and compact support-map patches or central/local fractions.        | Exact frozen V7.8 implementation and hashes.                                                                           |
| Observed edges                                       | Canny or fixed gradient channels from observed image.                                   | Thresholds fixed before fitting; no tuning on independent data.                                                        |
| Background-normalised intensity                      | Observed divided by frozen local paper/background estimate.                             | Background algorithm fixed; no use of clean page.                                                                      |
| Route context                                        | Frozen V7.8 preserve/remove/ambiguous indicators and local fractions.                   | May be used as context, not as truth.                                                                                  |
| Orientation/topology                                 | Fixed line-response, continuity and junction descriptors computed from observed patch.  | Definitions frozen before development selection.                                                                       |
| Annotation confidence, optional secondary experiment | Frozen YOLO confidence map from the detector version named in the manifest.             | Not in the primary model. If tested, it is a separately reported ablation and cannot replace the primary result.       |

## 4.2 Prohibited inputs

- Clean image pixels, clean-derived ink/edge maps, reconstruction error or any post-hoc outcome metric.
- Page name, page number, raw x/y coordinates, score title, edition name or source-group identifier.
- Human final decision on the same candidate, unless that record is used only as a label and is separated by page/source group.
- Independent-cohort outcomes, summary statistics or thresholds derived after unblinding.
- V7.9.x outcome variables, full-development fitted decisions, or any field whose name or lineage contains clean, truth, outcome, failure, precision, recall, MAE or structural-safe.

# 5. UNCERTAIN rule

UNCERTAIN is an operational safety class, not a residual bucket to be minimised. A pixel is preserved only by the PRINT rule. All other cases default to V7.8.

| **Trigger**                                                                                        | **Required output**                                                                                                   |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Maximum calibrated PRINT probability below 0.995                                                   | UNCERTAIN unless ANNOTATION probability is >= 0.995.                                                                  |
| PRINT probability >= 0.995 but margin over ANNOTATION < 0.990                                      | UNCERTAIN.                                                                                                            |
| Predictive disagreement: >= 2 of 5 page-grouped ensemble members disagree on the top class         | UNCERTAIN.                                                                                                            |
| Out-of-distribution score above the pre-registered 99th percentile of development training patches | UNCERTAIN.                                                                                                            |
| Any missing, non-finite or schema-mismatched input                                                 | UNCERTAIN and log a fail-closed event.                                                                                |
| Patch intersects an excluded label zone, unreadable border, or failed registration-quality region  | UNCERTAIN.                                                                                                            |
| Connected candidate run shorter than 2 pixels or isolated single-pixel PRINT                       | UNCERTAIN; no singleton preservation.                                                                                 |
| Otherwise                                                                                          | PRINT only if all PRINT conditions pass; ANNOTATION only if its calibrated probability >= 0.995; otherwise UNCERTAIN. |

**Numerical status  
**The 0.995 thresholds and 0.990 class-margin are safety-first commissioning choices. They are deliberately stringent and must not be relaxed after viewing independent outcomes.

# 6. Dataset, split and holdout controls

| **Cohort**                       | **Permitted use**                                 | **Separation rule**                                                                                        |
| -------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Development training             | Fit model and preprocessing.                      | Grouped by page; no patch from a held-out page may enter training.                                         |
| Development OOF                  | Model selection, calibration and gate assessment. | Outer folds grouped by source group where possible, otherwise by page; inner fitting remains page-grouped. |
| Development audit reserve        | Read-only label and leakage audit.                | At least 10% of eligible development pages, selected before modelling and not used for model choice.       |
| Frozen V7.8 independent set      | Permanent regression reference only.              | Must not be used for fitting, thresholding or model selection.                                             |
| New independent promotion cohort | Single final promotion test.                      | Untouched until model, schema, calibration, UNCERTAIN rule and gates are locked.                           |

- The new independent cohort must contain at least 30 pages, at least 5 source groups and at least 3 pages per source group.
- At least 20% of independent pages must be designated sparse-print/hard-negative pages before unblinding.
- No source group may appear in both development and the new independent cohort.
- Sampling, exclusions and label-quality failures must be reported with exact counts before outcomes are shown.

# 7. Model-development protocol

7. Run a read-only dataset audit: roster agreement, hashes, shapes, class balance, page/source coverage, label uncertainty, duplicates and registration sensitivity.
8. Freeze candidate architectures before outcome comparison: regularised logistic/linear baseline on engineered features; random forest or histogram gradient boosting; compact convolutional patch classifier. No architecture search beyond these families.
9. Use nested page/source-group validation. Hyperparameters are selected inside training folds only.
10. Produce page-grouped out-of-fold class probabilities for every eligible development sample.
11. Calibrate probabilities using only out-of-fold predictions, with calibration itself nested or cross-fitted by page.
12. Select one challenger using the pre-registered safety-first ordering: integrity, catastrophic errors, false-preservation bound, positive utility, stability, then coverage.
13. Refit the chosen architecture once on all permitted development data, freeze artefacts and replay-test it before independent scoring.
14. Run the frozen challenger exactly once on the new independent cohort. No refit, recalibration, threshold change or schema change is permitted.

# 8. Metrics and error taxonomy

| **Metric**              | **Definition**                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------- |
| Extra true print        | Pixels preserved by challenger but not V7.8 that are PRINT under clean truth.                       |
| Extra false annotation  | Pixels preserved by challenger but not V7.8 that are ANNOTATION under clean truth.                  |
| False/true ratio        | Extra false annotation / max(extra true print, 1).                                                  |
| False-preservation rate | Extra false annotation / all challenger-added preserved pixels.                                     |
| Net safety utility      | Extra true print - 5 x extra false annotation - 500 x catastrophic components.                      |
| Catastrophic component  | Any component with >= 50 extra false pixels or false/true ratio > 1.0 and >= 20 extra false pixels. |
| Page coverage           | Pages with at least 10 extra true-print pixels and no catastrophic component.                       |
| UNCERTAIN rate          | Eligible candidates emitted as UNCERTAIN.                                                           |
| Outside-mask drift      | Any changed pixel outside the authoritative removal mask; required value is zero.                   |
| V7.8 regression         | Any pixel difference where challenger did not emit PRINT; required value is zero.                   |

Uncertainty: use a page-clustered bootstrap with 10,000 resamples for utility, false/true ratio and false-preservation rate. Also report Wilson 95% upper bounds for binomial summaries and results by source group, page, route context and sparse-print/hard-negative status.

# 9. Numerical development promotion gates

All gates are conjunctive. Failure of any gate prevents freezing the challenger for independent testing.

| **Gate**                      | **Required development OOF result**                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1 Integrity                  | Zero schema violations; zero duplicate leakage; zero non-finite scored inputs; exact replay on a frozen sample.                                                           |
| D2 Immutability               | Zero outside-mask drift and zero non-PRINT differences from V7.8.                                                                                                         |
| D3 Catastrophic safety        | Zero catastrophic components in pooled OOF and in every source group.                                                                                                     |
| D4 False-preservation         | Pooled false/true ratio <= 0.10 and page-clustered 95% upper bound <= 0.20.                                                                                               |
| D5 Pixel safety               | Pooled false-preservation rate <= 0.50% and 95% upper bound <= 1.00%.                                                                                                     |
| D6 Positive utility           | Net safety utility > 0 and page-clustered 95% lower bound > 0.                                                                                                            |
| D7 Non-trivial benefit        | At least 500 extra true-print pixels across at least 10 pages and at least 3 source groups.                                                                               |
| D8 Source stability           | Every source group with >= 3 eligible pages has false/true ratio <= 0.25 and no catastrophic component.                                                                   |
| D9 Sparse-print/hard-negative | Zero catastrophic components; false-preservation rate <= 0.25%.                                                                                                           |
| D10 Uncertainty behaviour     | UNCERTAIN rate between 20% and 95%; outside this range requires stopping for specification review, not threshold tuning.                                                  |
| D11 Baseline dominance        | Challenger must outperform the best pre-registered non-learned baseline on utility without worsening D3-D5.                                                               |
| D12 Visual audit              | All catastrophic candidates (expected zero), the 50 most false-heavy, 50 most beneficial and 50 most uncertain cases reviewed; no unmodelled systematic failure accepted. |

# 10. Numerical independent promotion gates

Independent promotion means eligibility for a controlled supervised pilot, not automatic production deployment. All gates are frozen before unblinding and are conjunctive.

| **Gate**                      | **Required untouched independent result**                                                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| I1 Lock and replay            | Dataset, model, calibrator, schema, thresholds and software hashes match the registered lock; replay test passes exactly.                          |
| I2 Immutability               | Zero outside-mask drift; zero changes outside challenger PRINT pixels; V7.8 output reproduced exactly elsewhere.                                   |
| I3 Catastrophic safety        | Zero catastrophic components overall and within every source group.                                                                                |
| I4 False-preservation         | Observed false/true ratio <= 0.10 and page-clustered 95% upper bound <= 0.20.                                                                      |
| I5 Pixel safety               | Observed false-preservation rate <= 0.50% and 95% upper bound <= 1.00%.                                                                            |
| I6 Positive utility           | Net safety utility > 0 and page-clustered 95% lower bound > 0.                                                                                     |
| I7 Minimum evidence           | At least 1,000 extra true-print pixels, at least 15 benefiting pages and at least 4 benefiting source groups.                                      |
| I8 Source stability           | No source group has false/true ratio > 0.25; no group has negative utility; no failure concentration hidden by aggregate results.                  |
| I9 Sparse-print/hard-negative | Zero catastrophic components and false-preservation rate <= 0.25%.                                                                                 |
| I10 Operational uncertainty   | UNCERTAIN rate between 20% and 95%, and at least 90% of false annotation candidates are ANNOTATION or UNCERTAIN rather than PRINT.                 |
| I11 Development consistency   | Independent point estimates do not breach any development safety ceiling and no new failure mechanism invalidates the target or input assumptions. |
| I12 Human sign-off            | Independent visual audit completed and a written release decision explicitly limits use to a supervised pilot.                                     |

**Fail-closed interpretation  
**If the selected sample is too small to satisfy I7 or produces a wide upper confidence bound, the result is "insufficient evidence", not a safety pass. If any false-heavy concentration or catastrophic component appears, the challenger fails at the frozen rule.

# 11. Integrity, reproducibility and audit

- Content-address every input file, semantic array, label table, split roster, model artefact, calibrator, predictor schema and threshold record with SHA-256.
- Require two consecutive unchanged lock calculations before creating the development freeze.
- Preserve raw observed pages, authoritative masks, frozen V7.8 routes, classifier probabilities, final three-class decisions and output differences as separate artefacts.
- Record software versions, random seeds, fold assignments, exclusions, calibration method and all pre-registered numerical constants.
- Build an exact equivalence test proving the challenger wrapper reproduces V7.8 byte-for-byte whenever PRINT is not emitted.
- Write all failures before raising an error. Never continue from partial or mismatched inputs.
- Keep independent truth sealed until predictions and decisions are written and hashed.

# 12. Required commissioning deliverables

| **Deliverable**          | **Acceptance content**                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 01_pre_registration.json | Machine-readable target, allowlist, prohibited inputs, splits, models, thresholds, metrics and gates.            |
| 02_dataset_audit         | Roster, label flow, class balance, source coverage, exclusions, registration sensitivity and duplication checks. |
| 03_development_oof       | Sample-level OOF probabilities/decisions, page/source summaries, clustered uncertainty and visual audit sets.    |
| 04_frozen_bundle         | Model, calibrator, ordered schema, constants, policy ID, hashes and replay test.                                 |
| 05_independent_lock      | Untouched cohort roster, input hashes, sealed truth record and deterministic lock ID.                            |
| 06_independent_report    | All I1-I12 outcomes, confidence bounds, source/page stratification, failures and contact sheets.                 |
| 07_decision_record       | Promote to supervised pilot, retain for further research, or reject; no automatic production authorisation.      |
| 08_regression_pack       | Permanent tests for outside-mask immutability, V7.8 equivalence and frozen artefact replay.                      |

# 13. Decision logic

| **Outcome**                                                     | **Decision**                                                                                                 |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Any integrity, leakage or immutability failure                  | STOP. Invalidate the run and correct implementation/data handling before any new lock.                       |
| Development D1-D12 all pass                                     | Freeze one challenger and commission one untouched independent test.                                         |
| Development safety passes but D7 benefit fails                  | Do not commission independent test; classifier is non-operational at the frozen safety posture.              |
| Independent I1-I12 all pass                                     | Eligible for a time-bounded, supervised Pilot 09-style evaluation; still no automatic acceptance.            |
| Independent safety passes but evidence/coverage is insufficient | Retain as promising research; gather a new untouched cohort without moving thresholds.                       |
| Any independent catastrophic or safety gate fails               | Reject the frozen challenger. Do not tune against the independent cohort.                                    |
| Aggregate pass but source-group instability                     | Do not generalise. Either reject or pre-register a future source-scoped experiment using new untouched data. |

# 14. Deviations and change control

- Before development outcomes: an amendment is allowed only with a dated rationale, version increment and preserved prior specification.
- After development outcomes but before independent lock: target, allowlist, thresholds, utility penalties and gates may not change. A material change creates a new experiment version and requires fresh development OOF.
- After independent lock: no changes of any kind to model, calibration, schema, decision thresholds, exclusions or gates. Any correction invalidates the lock and requires a new untouched cohort.
- The independent cohort may never be recycled into threshold selection for this registered experiment.

# 15. Source basis, assumptions and confidence

Project-grounded constraints used in this specification:

| **Project source**                                                     | **Use in this specification**                                                                                                                                                                                        |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V7_9_x_Research_Findings_and_Next_Steps.docx                           | Recommends a local observed-print classifier; PRINT/ANNOTATION/UNCERTAIN outputs; observable truth-free inputs; page/source grouping; frozen V7.8 comparator; new untouched holdout; safety-first predeclared gates. |
| V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx    | Separates annotation confidence from correction safety; confines changes to the authoritative mask; preserves outside-mask pixels; defers automatic acceptance pending independent validation.                       |
| 12_V7_7_Asymmetric_Structural_Risk_Calibration 1.txt                   | Provides page-grouped validation, predictor firewall, out-of-fold calibration, Wilson bounds and asymmetric safety treatment.                                                                                        |
| 17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt       | Provides deterministic locking, sealed outcomes, semantic array hashes, frozen V7.8 implementation and structural label controls.                                                                                    |
| 18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt                        | Provides exact stage-by-stage equivalence and fail-closed regression principles.                                                                                                                                     |
| V7_7F_Design_Record_Applying_the_Frozen_Structural_Policy_to_V7_8.docx | Prohibits refitting, recalibration and threshold movement on independent data; requires uncertainty and source-group reporting.                                                                                      |
| 09_Pilot_Proposed_Correction_Routing_Reviewer.txt                      | Provides the supervised-only operational pattern and exact V7.8 regression gate.                                                                                                                                     |
| Production_Retraining_Policy_YOLO_and_Structural_Safety.docx           | Requires frozen incumbents, challengers, documented promotion and no automatic promotion.                                                                                                                            |

**Confidence  
**Overall confidence: 89%. Confidence is high for the architecture, leakage controls, target separation, fail-closed behaviour and independent-validation design because those are explicit in the project record. Confidence is lower for the numerical D/I thresholds because the source document explicitly says they still need to be frozen; the values here are conservative commissioning decisions intended to make the experiment executable without post-hoc threshold invention.

Next question: Is the project willing to accept a deliberately low-coverage classifier if that is what is required to keep the independent 95% upper bound on false preservation at or below 1.00%?