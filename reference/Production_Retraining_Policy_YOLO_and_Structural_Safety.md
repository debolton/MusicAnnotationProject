**Production Retraining Policy**

_YOLO annotation detector and structural-safety policy_

Music Annotation Project | 12 October 2026

**Core decision: this is not a one-and-done system, but neither model should be retrained simply because time has passed. Collect evidence continuously, retrain when defined triggers occur, and promote only a challenger that passes the existing integrity, regression and safety gates.**

# 1. Two systems, two maintenance policies

The annotation detector and structural-safety policy answer different questions and must remain separately versioned. The YOLO detector estimates where annotation exists. The structural-safety policy estimates whether changing a proposed component risks genuine printed musical information. High confidence on one axis does not establish high confidence on the other.

| **System**               | **Primary question**                                  | **Expected change rate**                                                                            |
| ------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| YOLO annotation detector | Where is annotation likely to exist?                  | Faster-moving. Sensitive to new handwriting, engraving, scanning and acquisition patterns.          |
| Structural-safety policy | Is the proposed correction safe for printed notation? | Slower-moving. Closely tied to reconstruction, routing, predictor definitions and structural gates. |

# 2. YOLO retraining policy

YOLO should be retrained when the reviewed training evidence has materially changed, not automatically on a calendar date. The existing pipeline already creates draft masks and confidence maps, supports human additions and erasures, publishes immutable dataset releases, trains a new best checkpoint, and evaluates one or more checkpoints without modifying them.

## Recommended retraining triggers

- A meaningful batch of newly reviewed corrected masks has accumulated.
- A new source, edition, scanner or handwriting style produces repeated false positives or missed annotation.
- Hard-negative evaluation exposes a recurring new failure pattern.
- The current model creates sustained review burden that corrected labels can address.
- A material data-quality correction changes labels or split assignments.

## Do not retrain because

- a month or quarter has elapsed;
- a small number of pages was added;
- training is technically available;
- a candidate improves one headline metric while regressing hard negatives or important source groups.

## YOLO release process

| **Stage**        | **Control**                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------- |
| Build            | Create a new immutable dataset release. Reuse unchanged page objects and preserve previous releases.            |
| Train challenger | Train a candidate checkpoint against the selected release. Do not overwrite the incumbent.                      |
| Evaluate         | Compare candidate and incumbent on validation, GST, hard negatives, threshold sweeps and visual contact sheets. |
| Promote          | Promote only if the candidate satisfies predeclared gates and does not introduce unacceptable regressions.      |
| Retain           | Keep the incumbent, candidate artefacts, dataset identity, metrics and decision record.                         |

# 3. Structural-safety retraining policy

The structural-safety policy should be retrained less frequently. Its current design freezes the geometry_t80 reconstruction, uses page-grouped cross-validation, excludes truth and raw coordinates from predictors, and treats candidate thresholds as retrospective findings until independent validation is complete.

## Retraining triggers

- The reconstruction algorithm or geometry_t80 implementation changes.
- Preserve, remove or ambiguous routing semantics change.
- The predictor schema or component construction changes.
- Structural outcome gates or the definition of structural_safe change.
- Independent monitoring demonstrates systematic miscalibration or repeated false-safe behaviour.
- A sufficiently substantial new independently labelled structural dataset becomes available and a new policy is intentionally commissioned.

## No-retraining cases

- A new YOLO model is promoted but reconstruction and structural predictors remain unchanged. Better annotation detection should feed more appropriate candidates through fixed safety gates.
- A small number of cosmetic-residual cases appears without structural damage. Cosmetic quality is a separate outcome.
- Coverage appears low. The safety threshold must not be weakened merely to increase automation.

# 4. Periodic monitoring versus periodic retraining

| **Activity**               | **YOLO**                                                                 | **Structural safety**                                                                               |
| -------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Continuous data collection | Yes. Preserve reviewed corrected masks and audit records.                | Yes. Preserve component decisions, review outcomes and source context.                              |
| Routine monitoring         | Yes. Track validation, GST, hard-negative and source-specific behaviour. | Yes. Track selection coverage, false-safe indicators, calibration drift and source-group stability. |
| Calendar retraining        | Not recommended as the primary rule.                                     | Not recommended.                                                                                    |
| Event-driven challenger    | Yes, when new reviewed evidence is material.                             | Yes, but only after structural semantics or evidence materially change.                             |
| Automatic promotion        | No.                                                                      | No.                                                                                                 |

# 5. Data feedback rules

- Unreviewed YOLO predictions are not training truth. Only reviewed or otherwise authoritative masks are eligible for future training.
- Keep raw confidence maps, corrected masks and reconstruction routing maps separate. They answer different audit questions.
- Do not consume the frozen V7.8 independent set into training if it is to remain a permanent regression set.
- Create additional untouched holdouts over time so repeated development does not optimise against one fixed test collection.
- Keep training, validation, hard-negative, regression and independent-validation roles explicit in release manifests.

# 6. Production versioning

A production release should identify the annotation detector and correction-safety policy independently. A detector can be upgraded without silently changing the structural policy, and a safety-policy revision can be audited without obscuring which detector supplied the annotation mask.

| **Release element**         | **Minimum identity**                                                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Annotation detector         | Dataset release ID, training configuration, best.pt hash, validation/GST/hard-negative report and approval decision.                        |
| Structural-safety policy    | Model hash, calibrator hash, ordered predictor schema, threshold, structural-gate definitions, independent-validation report and policy ID. |
| Combined production release | Detector ID, structural-policy ID, reconstruction version, configuration fingerprints and release decision.                                 |

# 7. Recommended operating model

Use continuous evidence collection with event-driven retraining. Keep the current production version frozen while a challenger is built. Evaluate the challenger against the incumbent and all frozen regression gates. Promote only after a documented decision. If a challenger fails, retain the incumbent and preserve the failed experiment for learning.

# 8. Practical default policy

| **Policy**            | **Default**                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| YOLO monitoring       | Continuous through reviewed pages, validation, GST and hard negatives.                               |
| YOLO retraining       | Triggered by material new reviewed evidence or demonstrated drift/failure patterns.                  |
| Structural monitoring | Continuous on selected/reviewed components and source groups.                                        |
| Structural retraining | Triggered by reconstruction/predictor/gate changes or strong independent evidence of policy failure. |
| Promotion             | Never automatic. Challenger must pass integrity, performance, regression and safety gates.           |
| Independent data      | Preserve frozen sets; add new holdouts rather than recycling every test case into training.          |

# 9. Current recommendation

Treat the first production release as a baseline, not a final model. Operate the YOLO detector as the more frequently challenged component and the structural-safety policy as a slower-moving fixed safety layer. The maintenance objective is controlled improvement without allowing detector gains, automation targets or retraining convenience to weaken the correction-safety standard.

# Source basis

- V7.5 design record: two-axis separation, independent validation, fixed safety gates and separate audit artefacts.
- Incremental dataset builder: immutable content-addressed dataset releases and reuse of unchanged page objects.
- GPU training notebook: training from a selected immutable release and evaluation of best.pt.
- Model evaluation suite: read-only checkpoint comparison across GST, hard negatives and confidence thresholds.
- HITL generator and reviewer: confidence maps, draft masks, human additions/erasures, corrected masks and audit records.
- V7.7 calibration: page-grouped validation, predictor firewall, retrospective candidate thresholds and mandatory independent validation.

# Confidence

Confidence: 95%. The separation of the two systems, immutable dataset releases, HITL feedback, page-grouped structural calibration and read-only model comparison are explicit in the project artefacts. The exact numerical trigger for "enough new data" is not currently defined, so this record recommends an event-driven governance policy rather than inventing a page-count or calendar interval.

# Next question

What numerical challenger-promotion gates should be frozen for YOLO and for the structural-safety policy before the first production release?