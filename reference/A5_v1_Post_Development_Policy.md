**MUSIC ANNOTATION PROJECT**

A5-v1 Post-Development Policy

_Freeze and operationalisation decision record | 23 November 2026_

**DECISION: Freeze A5-v1 as the selected post-development observed-print policy at effective calibrated PRINT threshold 0.980, PRINT margin 0.960 and minimum connected PRINT run 2. A5-v1 restores exact observed bytes only at valid PRINT coordinates and otherwise reproduces A1 byte-for-byte. This is a frozen candidate for untouched independent testing, not automatic production authorisation.**

# 1. Status and interpretation

A5-v1 is the selected policy arising from the completed development programme and the corrected Notebook 89R reconstruction audit. The 0.980 operating point was selected retrospectively from development evidence. It is therefore frozen prospectively before any untouched independent test and must not be represented as the original pre-registered 0.995 rule.

- Development policy selected: yes.
- Immutable operational bundle to be created by Notebook 91: yes.
- Untouched independent validation completed: no.
- Automatic production deployment authorised: no.
- Permitted next use after freeze: one untouched independent promotion test, followed only by controlled supervised/shadow use if the independent gates pass.

# 2. Policy identity

| **Field**                                    | **Frozen A5-v1 value**                                              |
| -------------------------------------------- | ------------------------------------------------------------------- |
| Policy ID                                    | A5-v1-print-p0980-margin0960-run2                                   |
| Model family                                 | Compact 33 x 33 spatial observed-print CNN                          |
| Inference target                             | PRINT probability for an eligible observed dark pixel               |
| PRINT threshold                              | 0.980                                                               |
| PRINT margin                                 | 0.960; for the binary model this is equivalent to p(PRINT) >= 0.980 |
| ANNOTATION threshold                         | 0.995                                                               |
| Minimum connected PRINT run                  | 2 pixels, 8-connectivity; singleton PRINT is demoted to UNCERTAIN   |
| Incumbent / fallback                         | A1, derived from frozen V7.8 geometry_t80 behaviour                 |
| PRINT action                                 | Copy the exact observed greyscale byte                              |
| Non-PRINT action                             | Keep the exact A1 byte                                              |
| Intervention boundary                        | Prepared authoritative dilation-2 removal mask only                 |
| Truth at runtime                             | Prohibited                                                          |
| Raw page coordinates and identity at runtime | Prohibited as model inputs                                          |

# 3. Scientific and operational contract

A5-v1 is an observed-print protection layer, not an annotation detector and not a generative reconstruction model. Its only permitted benefit is preservation of visible printed material that A1 would otherwise remove or reconstruct.

| **Decision**               | **Required treatment**                                      |
| -------------------------- | ----------------------------------------------------------- |
| PRINT                      | Restore the observed input byte at the same coordinate.     |
| ANNOTATION                 | Do not restore through A5; retain A1.                       |
| UNCERTAIN                  | Do not restore through A5; retain A1 and record the reason. |
| Missing or invalid input   | Fail closed; emit no A5 change.                             |
| Outside authoritative mask | Immutable observed input; A5 has no authority.              |

# 4. Development evidence supporting selection

The corrected Notebook 89R interpretation treats the saved calibrated score as PRINT-oriented and constructs A5 from A1 by restoring observed bytes. On the two aggregate frontier dimensions, 0.980 dominated 0.990 and 0.995 and provided materially greater coverage with the lowest observed annotation-retained-per-print-restored ratio among the tested thresholds.

| **Effective threshold** | **Genuine PRINT restored** | **Annotation retained** | **Eligible PRINT coverage** | **Annotation / PRINT** |
| ----------------------- | -------------------------- | ----------------------- | --------------------------- | ---------------------- |
| 0.995                   | 5,805                      | 220                     | 1.86%                       | 3.79%                  |
| 0.990                   | 23,656                     | 345                     | 8.13%                       | 1.46%                  |
| 0.980                   | 44,571                     | 553                     | 14.95%                      | 1.24%                  |
| 0.950                   | 62,343                     | 1,159                   | 21.32%                      | 1.86%                  |

The policy decision accepts 0.980 as the post-development operating point. The figures above are development evidence, not independent estimates. Concentrated annotation-retention cases remain material and must be carried into the independent and supervised review stages rather than hidden by pooled ratios.

# 5. Freeze boundary

- Train one final CNN using all permitted development training samples and the frozen architecture/hyperparameters.
- Fit the production isotonic calibrator from development out-of-fold raw predictions and labels, preserving the established calibration lineage.
- Freeze ordered channel schema, preprocessing constants, eligibility definition, thresholds, connected-run logic and class mapping.
- Freeze model, calibrator, inference wrapper, policy JSON, software environment, source identities and replay pack.
- Hash every frozen object with SHA-256 and create a deterministic freeze manifest.
- Require two consecutive unchanged candidate lock calculations before the definitive freeze is created.
- Require exact replay of probabilities, decisions and assembled page bytes on a frozen multi-page sample.

# 6. Required bundle

| **Object**                 | **Purpose**                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| policy.json                | Policy identity, thresholds, class meanings, runtime actions and fail-closed rules. |
| channel_schema.json        | Ordered 11-channel schema, patch size, dtype, scaling and padding.                  |
| model.pt                   | Final all-development CNN checkpoint.                                               |
| model_config.json          | Architecture and training hyperparameters.                                          |
| isotonic_calibrator.joblib | Frozen probability calibration object.                                              |
| inference_runtime.py       | Truth-free runtime wrapper for scoring arrays and assembling A5 output.             |
| training_roster.csv        | Permitted development samples used for the final fit.                               |
| calibration_roster.csv     | OOF rows used to fit the final calibrator.                                          |
| replay/                    | Frozen observable arrays, expected probabilities, final decisions and A5 bytes.     |
| software_environment.json  | Python, PyTorch, CUDA, NumPy, OpenCV, pandas and joblib versions.                   |
| artifact_manifest.csv      | Path, role, size and SHA-256 for every bundle object.                               |
| freeze_manifest.json       | Candidate lock, source identities, policy ID, replay result and freeze status.      |

# 7. Runtime invariants

- A5 output equals A1 at every coordinate not finally classified as PRINT.
- Every changed A5 byte equals the observed input byte exactly.
- No pixel outside the authoritative removal mask changes.
- No clean image, truth label, truth-derived map or post-hoc outcome enters inference.
- Any schema, shape, hash, non-finite, eligibility or model-loading failure stops processing rather than relaxing the rule.
- Connected-run filtering is page-global and is applied after provisional decisions are assembled.
- Batch size may change for memory only; scores and decisions must remain invariant.

# 8. Change control

| **Change**                                         | **Treatment**                                              |
| -------------------------------------------------- | ---------------------------------------------------------- |
| Threshold, margin or run rule change               | New policy version; A5-v1 is not edited.                   |
| Channel order, preprocessing or eligibility change | New policy version and fresh replay/validation.            |
| Architecture or training change                    | New challenger model and new policy ID.                    |
| Correction engine or A1 change                     | New combined compatibility identity and regression pack.   |
| Independent result is disappointing                | Do not tune A5-v1 against that cohort.                     |
| Later authoritative evidence becomes available     | Commission A5-v2 under event-driven challenger governance. |

# 9. Independent-test and deployment boundary

Notebook 91 creates the immutable candidate only. The next notebook must run A5-v1 once on a new untouched cohort whose truth has not influenced model, calibration, policy or exclusions. Independent success makes A5-v1 eligible for controlled supervised/shadow integration. It does not validate detector-generated masks and does not authorise unattended production correction.

# 10. Rationale for stopping development

The remaining errors are concentrated edge cases rather than broad inability to distinguish print from annotation. Continuing to tune the same pages risks optimising against known failures and degrading generalisation. The proper next step is release engineering, replay and untouched validation, while preserving the current development evidence and failure queues.

# 11. Source basis

- Observed-Print Classifier Pre-Registered Commissioning Specification v1.1: three-class target, truth firewall, refit-on-development then freeze, replay and independent-test separation.
- V8.2 Development Whole-Page Experiment Decision Record: development first, freeze second, independent test third.
- V8.1 Full-Pixel Whole-Page OOF Notebook Specification: required bundle objects, inference identity and replay expectations.
- V8.2a Policy Sensitivity notebook: 0.980 / 0.960 / run-2 exploratory policy and binary effective-threshold relationship.
- V8 Next Stage Full-Pixel Whole-Page OOF Reference: A1/V7.8 fallback identity and exact observed-byte restoration at PRINT coordinates.
- Production Retraining Policy: frozen incumbent, separately versioned challengers and no automatic promotion.

# 12. Decision statement

**A5-v1 is frozen as policy A5-v1-print-p0980-margin0960-run2. Notebook 91 shall create one immutable all-development model/calibrator/runtime bundle, prove exact A1 fallback and observed-byte restoration by replay, hash every artefact, and prohibit any subsequent threshold, schema, calibration or model change under the A5-v1 identity. The bundle remains a candidate for untouched independent testing and is not production-authorised.**

Confidence: 92%. The policy semantics, stage separation and bundle requirements are strongly supported by the project records. The principal uncertainty is the calibration-transfer assumption when the final all-development model is paired with a calibrator fitted from source-group OOF raw scores; Notebook 91 records and replay-tests that choice, but untouched validation remains necessary.

Next question: Does the frozen A5-v1 bundle reproduce the selected 0.980 development policy closely enough on its replay pack, and does one untouched independent cohort sustain the safety/coverage balance without concentrated annotation retention?