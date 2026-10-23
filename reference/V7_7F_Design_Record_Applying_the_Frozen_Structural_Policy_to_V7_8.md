**V7.7F Design Record**

_Applying the frozen structural-safety policy to the locked V7.8 independent dataset_

Music Annotation Project | 12 October 2026

**Core decision: apply the policy bundle exactly once to the already locked V7.8 component dataset. Do not refit the forest, refit the imputer, recalibrate probabilities, change the predictor schema, move the threshold, or use the V7.8 outcomes to alter the decision rule.**

# 1. What this step is doing

The structural-policy freeze created a deployable candidate policy bundle from the corrected V7.7 development evidence. The application step now asks a deliberately narrow question: when that unchanged policy is presented with the independent V7.8 predictor rows, which components does it classify as candidate structural-auto-safe, and are any of those selected components structurally unsafe according to the separately released V7.8 truth outcomes?

V7.5 established that annotation confidence and correction safety are different questions. The policy being applied here answers only the correction-safety question. It does not decide whether a mark is annotation and it does not replace the reviewed annotation mask.

# 2. Why refitting is prohibited

The V7.8 dataset is the independent test set. If the model, calibrator, predictor list or threshold is adjusted after seeing V7.8 outcomes, the experiment stops being independent and becomes another development exercise.

| **Permitted**                                           | **Not permitted**                                           |
| ------------------------------------------------------- | ----------------------------------------------------------- |
| Load the saved forest, imputer and isotonic calibrator  | Train or fine-tune any estimator on V7.8                    |
| Use the exact ordered predictor schema                  | Add, remove, rename or reorder predictors opportunistically |
| Apply the threshold stored in policy.json               | Choose a new threshold after inspecting V7.8 outcomes       |
| Compare frozen decisions with released structural truth | Use structural truth as an input to scoring                 |
| Report failures, coverage and uncertainty               | Suppress poor results or weaken the structural gates        |

# 3. Inputs required

| **Input**                     | **Expected location or role**                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Locked V7.8 component dataset | The released component_dataset.csv whose hash is recorded in dataset_release.json                            |
| V7.8 dataset release          | dataset_release.json, containing the V7.8 lock ID and released dataset hash                                  |
| V7.8 lock record              | locks/&lt;lock_id&gt;.json, proving the predictor and input state that was locked before structural outcomes |
| Frozen V7.7F policy           | artefacts/policy.json                                                                                        |
| Frozen predictor schema       | artefacts/predictor_schema.json                                                                              |
| Frozen structural model       | artefacts/structural_model.joblib                                                                            |
| Frozen calibrator             | artefacts/isotonic_calibrator.joblib                                                                         |
| Freeze replay record          | artefacts/replay_test.json                                                                                   |

# 4. Application sequence

1. Verify that component_dataset.csv matches the SHA-256 recorded in dataset_release.json.
2. Verify that the release lock ID exists and that the lock status is the expected pre-outcome lock.
3. Verify the SHA-256 hashes of the model, calibrator and predictor schema against policy.json.
4. Load component_dataset.csv without altering or re-exporting its predictor values.
5. Select the predictor columns in the exact order stored in predictor_schema.json.
6. Convert predictor values to numeric, reject infinite values, and allow only the missing values handled by the frozen median imputer.
7. Call the frozen model predict_proba method to produce raw structural-safe scores.
8. Pass those raw scores through the frozen isotonic calibrator.
9. Apply the unchanged threshold stored in policy.json.
10. Only after decisions have been created, compare them with structural_safe and structural_failure_reasons.
11. Write a scored dataset and an independent-validation summary. Do not overwrite component_dataset.csv.

# 5. Minimal application code

The following form shows the required logic. Paths must be changed to the actual freeze-run and locked V7.8 release directories. The application should be run in a new notebook or script, not by editing V7.7 or V7.8.

```
from pathlib import Path
import json, hashlib, math
import joblib
import numpy as np
import pandas as pd

ROOT = Path('/content/drive/MyDrive/MusicAnnotationProject')
POLICY_DIR = ROOT/'structural_policy_freeze_v7_7'/'<freeze_run>'/'artefacts'
V78_DIR = ROOT/'independent_structural_validation_v7_8_rebuilt_telea_immutable'
OUTPUT_DIR = ROOT/'independent_policy_validation_v7_8'/'<new_run>'
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

policy = json.loads((POLICY_DIR/'policy.json').read_text())
schema = json.loads((POLICY_DIR/'predictor_schema.json').read_text())
release = json.loads((V78_DIR/'dataset_release.json').read_text())

component_path = V78_DIR/'component_dataset.csv'
if release['component_dataset_sha256'] != sha256_file(component_path):
    raise RuntimeError('V7.8 component dataset hash mismatch.')

model_path = POLICY_DIR/'structural_model.joblib'
calibrator_path = POLICY_DIR/'isotonic_calibrator.joblib'
if policy['model_sha256'] != sha256_file(model_path):
    raise RuntimeError('Frozen model hash mismatch.')
if policy['calibrator_sha256'] != sha256_file(calibrator_path):
    raise RuntimeError('Frozen calibrator hash mismatch.')
if policy['schema_sha256'] != sha256_file(POLICY_DIR/'predictor_schema.json'):
    raise RuntimeError('Frozen predictor schema hash mismatch.')

df = pd.read_csv(component_path)
predictors = schema['ordered_predictors']
missing = [c for c in predictors if c not in df.columns]
if missing:
    raise RuntimeError(f'Missing frozen predictors: {missing}')

X = df.loc[:, predictors].copy()
for column in predictors:
    X[column] = pd.to_numeric(X[column], errors='coerce')
if np.isinf(X.to_numpy(dtype=float)).any():
    raise RuntimeError('Infinite predictor value found.')

model = joblib.load(model_path)
calibrator = joblib.load(calibrator_path)
raw_score = model.predict_proba(X)[:, 1]
probability = calibrator.predict(raw_score)
selected = probability >= float(policy['threshold'])

result = df.copy()
result['raw_structural_safe_score'] = raw_score
result['prob_structural_safe'] = probability
result['candidate_structural_auto_safe'] = selected
result['policy_id'] = policy['policy_id']
result.to_csv(OUTPUT_DIR/'v7_8_policy_scored_components.csv', index=False)
```

# 6. Required validation calculations

The primary safety error is a false-safe selection: the frozen policy selects a component for the candidate auto-safe tier, but the independent structural truth is unsafe.

```
truth = result['structural_safe'].astype(bool).to_numpy()
selected = result['candidate_structural_auto_safe'].astype(bool).to_numpy()

selected_count = int(selected.sum())
false_safe_count = int(np.sum(selected & ~truth))
safe_not_selected = int(np.sum(~selected & truth))
structural_safe_total = int(truth.sum())

observed_false_safe_rate = (
    false_safe_count / selected_count if selected_count else None
)
selection_coverage = selected_count / len(result)
structural_safe_recall = (
    np.sum(selected & truth) / structural_safe_total
    if structural_safe_total else None
)
```

Also report results by source_group, page and structural_failure_reasons. A single overall result can hide a policy that works for one edition or scan style but fails for another.

# 7. Uncertainty must be reported

A zero observed failure count does not prove that the true failure rate is zero. The V7.7 calibration work therefore reports a Wilson 95% upper confidence bound for the selected false-safe rate. The same calculation should be applied to the independent selected set.

```
def wilson_upper(errors, n, z=1.96):
    if n == 0:
        return None
    p = errors / n
    denominator = 1 + z*z/n
    centre = p + z*z/(2*n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (centre + radius) / denominator

false_safe_wilson95_upper = wilson_upper(
    false_safe_count,
    selected_count,
)
```

# 8. Output record

| **Output**                         | **Purpose**                                                                                           |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| v7_8_policy_scored_components.csv  | Original V7.8 component rows plus raw score, calibrated probability, selection decision and policy ID |
| independent_policy_summary.json    | Lock ID, policy ID, hashes, counts, rates, Wilson bound and pass/fail decision                        |
| policy_results_by_source_group.csv | Selection coverage and false-safe results for each independent source group                           |
| policy_results_by_page.csv         | Page-level stability and concentrated failure detection                                               |
| false_safe_components.csv          | Every independently unsafe component selected by the frozen policy                                    |
| safe_not_selected_components.csv   | Safe components sent to review, used to quantify conservatism rather than safety failure              |
| application_manifest.json          | Exact input, policy and software fingerprints used for the run                                        |

# 9. Pass/fail semantics

The application notebook should not invent a pass criterion after seeing the result. The threshold is already frozen, but the acceptable independent false-safe bound must also be stated before interpreting the run. If no such acceptance limit was formally frozen, report the observed result and confidence bound as evidence and do not upgrade the policy to production auto-acceptance.

| **Result**                                                                                | **Interpretation**                                                                                                                               |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| No components selected                                                                    | The policy is non-operational on V7.8. Safety cannot be demonstrated through selected cases; coverage is zero.                                   |
| Selected components and one or more false-safe cases                                      | The frozen candidate policy fails the primary independent safety objective at its current threshold.                                             |
| Zero observed false-safe cases but a wide Wilson upper bound                              | Promising but insufficiently precise evidence because the selected sample is too small.                                                          |
| Zero observed false-safe cases with a pre-accepted upper bound and source-group stability | Evidence supports the candidate structural-safety tier, subject to the separately required annotation-confidence policy and production approval. |
| Good aggregate result but failure concentrated in one source group                        | Do not generalise the policy across all sources. Route the affected source group to review or gather further independent evidence.               |

# 10. What this does not authorise

- It does not authorise automatic acceptance based on structural safety alone.
- It does not combine YOLO confidence and structural safety into a single score.
- It does not replace human-reviewed corrected masks.
- It does not permit threshold tuning against V7.8.
- It does not turn V7.8 into training data.
- It does not validate cosmetic quality unless a separately frozen cosmetic policy is applied and tested.

# 11. Production interpretation

If the independent result is acceptable, the structural policy becomes one fixed gate in a two-axis workflow. A component may enter an automatic tier only when the annotation-confidence decision and the independently validated correction-safety decision both satisfy their own frozen requirements. Otherwise the component remains in review.

```
ANNOTATION CONFIDENCE             CORRECTION SAFETY
          |                               |
          v                               v
   frozen detector /                frozen V7.7F
 confidence decision              structural policy
          |                               |
          +---------------+---------------+
                          |
                          v
              BOTH independently acceptable?
                    /               \
                  YES               NO
                   |                 |
        candidate auto-accept      review
```

# 12. Immediate next action

Create a small, read-only V7.8 policy-application notebook that contains the integrity checks and scoring sequence above. Point it at the exact saved V7.7F policy directory and the already released V7.8 component_dataset.csv. Run it once. Do not rerun V7.7F, rebuild V7.8, or touch the threshold before the independent result has been written and preserved.

# Source basis

This record is based on the existing project design and notebook contracts:

- &lt;File&gt;V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx&lt;/File&gt;: two-axis separation, controlled-mask boundary, independent validation requirement and audit trail.
- &lt;File&gt;12_V7_7_Asymmetric_Structural_Risk_Calibration 1.txt&lt;/File&gt;: page-grouped structural-risk calibration, predictor firewall, isotonic calibration, candidate thresholding and independent-validation requirement.
- &lt;File&gt;17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt&lt;/File&gt;: predictor locking, sealed structural outcomes and explicit release of component_dataset.csv.
- &lt;File&gt;18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt&lt;/File&gt;: equivalence gate for geometry, routing, predictors and V7.7 component alignment.

# Confidence

Confidence: 96%. The no-refit, no-recalibration and no-threshold-change procedure follows directly from the independent-validation design. The remaining uncertainty is the final numerical production acceptance limit for the independent false-safe confidence bound; the existing materials require independent validation but do not specify one final universally accepted limit.

# Next question

What independent false-safe upper-bound limit and minimum selected sample size should be frozen before we interpret the V7.8 policy-application result?