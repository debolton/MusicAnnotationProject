**MUSIC ANNOTATION PROJECT**

Recent Experimental Outcomes  
and Objectives for A5

_From V8.9, A1 and A3 comparison to an A1-based selective cleanup challenger_

**Decision in one sentence  
**A1 is the strongest current visual-restoration candidate because it best protects readable printed music, but A5 should start from A1 and selectively remove only those A1-preserved AMBIGUOUS pixels for which there is strong, truth-free evidence of safe annotation cleanup.

| **Current candidate**              | **Status**                                      | **Reason**                                                                                                                                            |
| ---------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| V8.9 incumbent                     | Retain as frozen comparator                     | Operationally reproducible, but unconditional AMBIGUOUS reconstruction is visibly too destructive on difficult HITL pages.                            |
| A1: preserve all AMBIGUOUS         | Leading production candidate for supervised use | Best visual protection of musical print; still removes annotation through the common REMOVE route, but leaves additional annotation within AMBIGUOUS. |
| A3: pixel gate at p_print >= 0.50  | Research comparator, not preferred output       | Improves materially on V8.9 but still returns too many uncertain print pixels to Telea; more annotation is removed than A1, at a visible print cost.  |
| A5: A1 plus selective safe cleanup | Next commissioned challenger                    | Must preserve A1 by default and override it only when strong evidence supports annotation removal without material print damage.                      |

Document status: research decision record and commissioning brief; not production authorisation.

# 1. Purpose and scope

This document consolidates the conclusions from the recent AMBIGUOUS-treatment, mask-profile transport and real-world HITL comparison experiments. It records what has been demonstrated, what remains perceptual or unproven, and the objectives and guardrails for the next challenger, A5.

- The operating objective is a clean, readable part for a future player, not maximum annotation removal at any cost.
- Structural preservation and cosmetic annotation residue remain separate outcomes.
- The reviewed annotation mask remains the sole boundary of permissible intervention; pixels outside authority must remain byte-for-byte unchanged.
- Clean truth may be used only for labels and post-lock scoring on paired pages, never as an inference input.
- The present HITL conclusions are based on actual whole-page visual comparison without clean truth and therefore support candidate ranking, not formal promotion.

**Governing design principle  
**High confidence that a mark is annotation does not imply high confidence that changing it is safe. A5 must preserve that distinction and treat uncertain correction as a reason to retain A1, not a reason to reconstruct.

# 2. Recent experimental sequence

| **Experiment**                      | **Question**                                                                                | **Principal outcome**                                                                                                                                                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Notebook 79 diagnostic              | What happens if AMBIGUOUS reconstruction is disabled?                                       | Restoring observed bytes in AMBIGUOUS recovered substantial visible musical information, establishing A1 as the conservative boundary candidate.                                                                              |
| Notebook 81 treatment benchmark     | How do A0/V8.9-style Telea, A1, blending, A3 and component gating compare on paired truth?  | A3 recovered materially more genuine print than unconditional Telea with very little annotation-cleaning loss on original paired masks. Corrected scoring showed A1 still removed annotation through the common REMOVE route. |
| Notebook 81 page-clustered analysis | Was A3's pooled benefit broad or driven by a few pages?                                     | A3 showed consistent positive print-safety deltas across pages, with no page-level print losses in the reported paired analysis; the annotation cost was small on original paired masks.                                      |
| Notebook 82 mask-profile transport  | Do skinny paired masks represent broad reviewed HITL masks, and does A3 survive broadening? | The mask-profile mismatch was material. Three-pixel dilation was insufficient. A3's print benefit survived and increased with broadening, but its annotation-retention/removal trade-off also became much larger.             |
| Notebook 83 actual HITL comparison  | How do V8.9, A1 and A3 look on reviewed real-world HITL masks?                              | A1 visually preserved the best print. A3 removed more annotation near printed areas but remained too destructive. Both A1 and A3 struggled with annotation away from printed structure.                                       |

# 3. Quantitative findings from paired truth

## 3.1 Correct interpretation of A1

The original "0% annotation removed" result for A1 was an AMBIGUOUS-only metric, not an end-to-end page metric. A1 preserves the observed byte only for AMBIGUOUS. Annotation routed through REMOVE continues to be replaced by the paper estimate. The corrected scoring therefore distinguishes AMBIGUOUS_EXPERIMENT, COMMON_CORRECTION and WHOLE_AUTHORITY.

## 3.2 A3 versus unconditional Telea on original paired masks

| **Whole-authority measure**    | **A0 / Telea** | **A3 pixel gate** | **A3 change**            |
| ------------------------------ | -------------- | ----------------- | ------------------------ |
| Annotation materially improved | 95.47%         | 95.39%            | \-0.08 percentage points |
| Exact print preserved          | 45.24%         | 52.15%            | +6.91 percentage points  |
| Print not worsened             | 71.77%         | 77.07%            | +5.30 percentage points  |
| Print materially worsened      | 21.72%         | 16.47%            | \-5.25 percentage points |

Interpretation: on thin paired masks A3 was close to a dominant improvement over unconditional Telea, but this did not establish superiority over A1.

## 3.3 Page-clustered A3 evidence

| **Metric where positive favours A3**   | **Mean paired-page benefit** | **95% page-bootstrap interval** | **Better / worse / tied pages** |
| -------------------------------------- | ---------------------------- | ------------------------------- | ------------------------------- |
| Exact genuine-print preservation       | +5.58 pp                     | +4.60 to +6.58 pp               | 46 / 0 / 4                      |
| Genuine print not worsened             | +3.90 pp                     | +3.17 to +4.71 pp               | 45 / 0 / 5                      |
| Reduction in materially worsened print | +3.87 pp                     | +3.14 to +4.66 pp               | 45 / 0 / 5                      |
| Annotation materially improved         | \-0.094 pp                   | \-0.162 to -0.042 pp            | 0 / 23 / 27                     |

This established A3 as a real print-safety improvement over unconditional Telea on the original paired-mask distribution. It did not establish that A3 was the best visually useful output, because A1 was not the baseline in this comparison.

# 4. Mask-profile transport findings

| **Morphology measure**  | **Paired median** | **Reviewed HITL median** | **Approximate shift** |
| ----------------------- | ----------------- | ------------------------ | --------------------- |
| Seed fraction of page   | 0.12%             | 0.72%                    | 6.0x                  |
| Area per skeleton pixel | 4.47              | 24.67                    | 5.5x                  |
| Mean mask depth         | 1.61              | 6.97                     | 4.3x                  |
| P90 mask depth          | 2.80              | 13.59                    | 4.9x                  |
| Maximum mask depth      | 5.20              | 26.78                    | 5.2x                  |

- A fixed 3-pixel dilation did not reproduce the reviewed HITL profile.
- M6 was the first fixed radius to place all selected central morphology measures inside the reviewed HITL interquartile range; page-specific target matching generally selected radius 7 or 8.
- Those radii were research transport devices, not a recommendation to dilate production masks.
- Broad masks changed the decision problem by increasing intervention authority, hiding outside-mask structural evidence and producing deeper, more merged components.

## 4.1 A3 under broadening

| **Profile**        | **Exact print preservation benefit** | **Reduction in materially worsened print** | **Annotation-improvement change** |
| ------------------ | ------------------------------------ | ------------------------------------------ | --------------------------------- |
| M0 original paired | +5.58 pp                             | +3.87 pp                                   | \-0.09 pp                         |
| M3                 | +9.63 pp                             | approximately +6.6 pp                      | \-6.56 pp                         |
| M5                 | +10.97 pp                            | approximately +7.8 pp                      | \-8.17 pp                         |
| MT target-matched  | +11.12 pp                            | +8.08 pp                                   | \-8.03 pp                         |

A3 remained robust as a print-protection intervention, but broadening exposed the governing trade-off: A3 protected substantially more print than unconditional Telea while also preserving substantially more annotation. This made the actual HITL whole-page comparison necessary.

# 5. Notebook 83 real-world HITL outcome

Notebook 83 compared observed pages, the frozen V8.9 incumbent, A1 and A3 on the complete reviewed HITL packages. The visual pattern was consistent across difficult score and part pages:

| **Candidate**             | **Observed visual behaviour**                                                                                                                      | **Current conclusion**                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| V8.9                      | Strongest annotation removal, but AMBIGUOUS-to-Telea visibly damages printed notation on difficult pages.                                          | Too destructive for the project's visual-restoration objective. Retain as frozen comparator.              |
| A1                        | Keeps the best musical print and still benefits from annotation removed by the common REMOVE route. Leaves residual annotation within AMBIGUOUS.   | Best current production candidate for supervised use, subject to formal governance and review.            |
| A3                        | Removes more annotation than A1 around printed regions, but returns too much uncertain material to Telea and remains visibly destructive to print. | Better than V8.9, but not better than A1 for the intended reader-facing output.                           |
| A1 and A3 away from print | Both leave or mishandle annotation in paper-dominant areas.                                                                                        | A separate cleanup signal is required; p_print was not commissioned as a blank-paper handwriting remover. |

**Updated candidate ranking  
**1\. A1 - strongest current visual candidate.  
2\. A3 - useful research comparator and source of discrimination evidence.  
3\. V8.9 - frozen incumbent/comparator, but not preferred reader-facing output.

# 6. A5 scientific objective

**A5 hypothesis  
**Start from A1, where every AMBIGUOUS pixel is preserved as observed. Permit cleanup only where a new observable-only decision layer provides strong positive evidence that the A1-preserved material is annotation and that changing it will not materially damage printed music.

A5 reverses the burden of proof used by A3:

| **A3 question**                                             | **A5 question**                                                                                |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Is there enough evidence that this is print to preserve it? | Is there enough evidence that this is safely removable annotation to override A1 preservation? |
| Default for low p_print: V8.9 / Telea                       | Default for uncertainty: A1 / observed                                                         |
| Risk: uncertain print is reconstructed                      | Risk controlled: uncertain material remains visible and reviewable                             |

## 6.1 A5 target population

- Only pixels inside the authoritative prepared mask.
- Only pixels in the frozen AMBIGUOUS route.
- Only pixels that A1 would preserve but the proposed A5 cleanup layer might surrender to cleanup.
- PRESERVE, REMOVE, Ring 2 and outside-authority treatment remain frozen and identical to A1/V8.9 as applicable.

## 6.2 Truth-side classes for paired development

| **Truth class**                           | **A5 treatment target**                                                          |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| Genuine printed music                     | Must remain the exact observed byte. Any cleanup is a structural false positive. |
| Annotation / non-print                    | Eligible for cleanup when the frozen decision rule passes.                       |
| Truth-uncertain or registration-sensitive | Must retain A1 and be reported separately; not silently counted as annotation.   |

## 6.3 Inference treatments

| **A5 decision**                     | **Output treatment**                                                                                                                                                                      |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SAFE_ANNOTATION_CLEANUP             | Apply the pre-registered cleanup treatment for that pixel/region. The treatment itself must be frozen and separately assessed; it should not default automatically to unrestricted Telea. |
| KEEP_A1                             | Copy the observed byte exactly, matching A1.                                                                                                                                              |
| UNCERTAIN / integrity failure / OOD | KEEP_A1 and log the fail-closed reason.                                                                                                                                                   |

# 7. Evidence A5 should use

A5 should use observable, truth-free evidence at inference. The primary architecture should include enough context to distinguish annotation on blank paper from annotation crossing printed structure.

| **Evidence family**         | **Recommended representation**                                                                        | **Reason**                                                                                      |
| --------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Observed image              | 33x33 local patch plus one pre-registered wider context, preferably 97x97                             | The failure is one of local ownership and wider structural context, not intensity alone.        |
| Mask geometry               | Binary mask, distance-to-boundary, central depth, component depth and local thickness                 | Broad HITL masks materially change evidence availability and intervention risk.                 |
| Frozen support maps         | Horizontal, vertical and compact support patches/fractions                                            | Protect staff lines, stems, bar lines, noteheads and compact glyphs.                            |
| Route context               | Frozen PRESERVE / REMOVE / AMBIGUOUS maps and local fractions                                         | Retains exact lineage to the correction engine.                                                 |
| Observed edges and topology | Fixed gradients, line responses, junction/continuity descriptors                                      | Supports ownership of the centre pixel when pencil and print intersect.                         |
| Paper-dominance evidence    | Local paper estimate, surrounding blank fraction, distance to credible print, bilateral paper support | Directly targets annotation away from printed areas, where both A1 and A3 currently struggle.   |
| Existing p_print            | Input feature or comparator, not the A5 decision itself                                               | Carries useful print evidence but has proved insufficiently conservative as a direct 0.50 gate. |
| YOLO confidence             | Optional separately reported ablation only                                                            | Annotation confidence and correction safety must remain separate axes.                          |

## 7.1 Context regimes to report

- NEAR_PRINT: AMBIGUOUS pixels close to credible horizontal, vertical, compact or edge structure. Cleanup requires the strongest evidence and should have the strictest print-damage ceiling.
- PAPER_DOMINANT: AMBIGUOUS pixels with strong blank-paper context and weak credible print structure. This is the main opportunity to remove annotation that A1 and A3 currently leave behind.
- MIXED / OOD: deep masks, page borders, conflicting evidence, registration sensitivity or unusual topology. Default to A1.

These regimes should be reporting and modelling context, not manually tuned source-specific thresholds after outcomes are viewed.

# 8. A5 objectives and predeclared success criteria

| **Objective**                     | **Required direction**                                                    | **Proposed development criterion**                                                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Preserve A1 structural advantage  | A5 must not materially worsen genuine print relative to A1.               | Page-clustered 95% confidence interval for the A5-minus-A1 print-damage change must include zero or favour A5, and its upper harm bound must remain within a small predeclared non-inferiority margin. |
| Remove more annotation than A1    | A5 must create a real cosmetic benefit.                                   | Positive page-clustered lower confidence bound for annotation materially improved beyond A1, with non-trivial gains across pages and source groups.                                                    |
| Avoid concentrated failures       | No page/component may trade a block of notation for cleanup.              | Zero catastrophic print-damage components under a predeclared concentration rule.                                                                                                                      |
| Remain source-stable              | No score, publisher, scan style or mask profile should carry hidden harm. | Report every source group and mask-depth band; no negative-utility group with adequate sample size.                                                                                                    |
| Handle broad masks                | Performance must transport toward reviewed HITL morphology.               | Repeat paired mask-profile arms or stratify by native mask thickness/depth; A5 benefit over A1 must not collapse in broad/deep masks.                                                                  |
| Improve paper-dominant annotation | The new hypothesis must address the observed gap away from print.         | Separate PAPER_DOMINANT endpoint showing additional annotation cleanup without print harm.                                                                                                             |
| Fail closed                       | Any uncertainty retains A1.                                               | Zero changes outside the A5 cleanup decision set and zero outside-authority drift.                                                                                                                     |

## 8.1 Metrics to report separately

- Genuine print: exact observed-byte preservation, not-worsened rate, materially worsened rate, ink/edge continuity and component concentrations.
- Annotation: materially improved toward clean, any improvement, residual annotation and paper-background quality.
- Decision burden: cleanup coverage, KEEP_A1 rate, UNCERTAIN rate and pages/source groups benefiting.
- Mask transport: results by seed thickness, depth, area-per-skeleton-pixel, prepared/seed expansion and component merger status.
- Uncertainty: 10,000 page-cluster bootstrap resamples plus source-group and page-level wins/losses/ties; pixels are not independent observations.

# 9. Recommended A5 experimental design

1. Freeze the A5 pre-registration before model fitting: target, eligible population, observable input schema, cleanup treatment, exclusions, uncertainty rule, models and numerical gates.
2. Use paired development pages for labels and scoring, with source-group-separated outer folds and page-grouped fitting/calibration.
3. Reserve an audit subset before modelling and preserve the frozen V7.8 independent set as a permanent regression reference rather than training material.
4. Compare at least: A1 incumbent, A3 comparator and A5 challenger. V8.9 remains a historical destructive baseline but should not be A5's success comparator.
5. Score A5 first against A1 on the original paired masks, then evaluate mask-profile transport using the Notebook 82 broadening/target-matching framework.
6. Lock all predictions and whole-page outputs before clean truth is loaded for scoring.
7. Produce actual whole-page A1-versus-A5 pages and focused disagreement crops. Difference maps locate changes but do not replace page judgement.
8. If development gates pass, freeze one A5 bundle and test once on a fresh untouched paired cohort before any real-HITL supervised pilot.

## 9.1 Candidate model families

- Interpretable engineered-feature baseline: regularised logistic regression or histogram gradient boosting.
- Primary spatial challenger: compact CNN using local and pre-registered wider context channels.
- Optional two-stage structure: conservative print veto first, then annotation-cleanup classifier only on pixels remaining eligible. This must be pre-registered, not assembled post hoc.
- Do not commission another threshold-only p_print sweep as A5. That would not be a materially new hypothesis.

# 10. Non-negotiable controls

| **Control**              | **A5 requirement**                                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Authority                | No pixel outside the reviewed/authorised prepared mask may change.                                                                                           |
| A1 fallback identity     | A5 must reproduce A1 byte-for-byte everywhere it does not emit a valid SAFE_ANNOTATION_CLEANUP decision.                                                     |
| Truth firewall           | No clean pixel, truth map, outcome metric, page identity, raw coordinate, title or source identity may enter inference.                                      |
| No independent tuning    | No refit, recalibration, threshold movement, exclusion change or schema change after independent lock.                                                       |
| Separate artefacts       | Preserve detector confidence, reviewed seed, prepared authority, frozen routes, A5 probabilities/classes, output differences and human decisions separately. |
| Versioning               | A5 remains a challenger. A1, A3 and V8.9 outputs and identities remain reproducible.                                                                         |
| Incomplete HITL packages | Skip explicitly with recorded reason; do not substitute draft or partial authority for completed reviewed packages.                                          |
| Promotion                | Never automatic. Development success permits freezing; independent success permits only a controlled supervised pilot.                                       |

# 11. Current decisions

| **Decision**                                                           | **Status**                                |
| ---------------------------------------------------------------------- | ----------------------------------------- |
| Use A1 as the design baseline for A5                                   | ADOPT                                     |
| Treat uncertain A5 cases as A1 / observed                              | ADOPT                                     |
| Use a direct p_print >= 0.50 rule as the final policy                  | REJECT                                    |
| Continue using V8.9 as the primary success comparator                  | REJECT - retain only as frozen comparator |
| Treat paper-dominant annotation as a distinct reporting/problem regime | ADOPT                                     |
| Tune thresholds on the Notebook 83 HITL pages                          | REJECT - no clean truth and post-hoc risk |
| Advance A1 directly to unsupervised production                         | NOT AUTHORISED                            |
| Commission A5 as a new, pre-registered challenger                      | RECOMMENDED                               |

**Recommended commissioning statement  
**Commission A5 as an A1-based selective safe-cleanup challenger. A5 should remove additional annotation only when a frozen observable-only rule demonstrates strong evidence of safe cleanup. Every uncertain or invalid case must retain A1 exactly. Success means materially more annotation removed than A1 with no material loss of A1's page-level print safety, including under reviewed-HITL-like broad mask profiles.

# 12. Evidence status and limitations

- Paired-page quantitative results provide clean-truth evidence for A0/A1/A3 treatment behaviour and mask-profile counterfactuals.
- Notebook 83 provides actual reviewed-HITL whole-page perceptual evidence but no clean-truth labels and no promotion authority.
- The numerical results quoted in this record are taken from the recent experiment outputs discussed in the project review. They should be attached to the final A5 pre-registration by exact run ID, file hash and table path before A5 coding begins.
- A1 is presently the leading candidate for supervised visual restoration, not a formally authorised unattended production release.

Confidence: 94%. The direction of the A5 design follows consistently from the paired-truth results, mask-profile transport experiment and actual HITL visual comparison. Confidence is below 100% because A1's superiority on the 151-page HITL cohort is perceptual rather than clean-truth scored, and A5's achievable cleanup frontier has not yet been tested.

# 13. Reference basis

- V7.5 Design Record - Annotation Confidence and Correction Safety
- V7.7 Asymmetric Structural-Risk Calibration
- V7.7F Design Record - Applying the Frozen Structural Policy to V7.8
- V7.8 Rebuilt Independent-Data Builder and V7.8.2 Equivalence Audit
- V7.9.x Research Findings and Recommended Next Steps
- Observed-Print Classifier Pre-Registered Commissioning Specification v1.1
- V8 Full-Pixel Whole-Page OOF Reference and V8.1 Specification
- V8.2 Development Whole-Page Experiment Decision Record
- Production Retraining Policy - YOLO and Structural Safety
- Pilot Notebook 09 supervised correction-routing reviewer
- Recent Notebook 79, 81, 82 and 83 outputs and review images.

**Next question:** What exact non-inferiority margin for print damage and minimum annotation-cleanup gain should be frozen before A5 is implemented?