Pathway from Pilot Notebook 09 to Annotated-Only Production Correction

**A staged route to processing pages that have no associated clean scan**

_Planning record | 14 October 2026_

**Important correction to the premise  
**An acceptable V8.0b development test tomorrow morning would not, by itself, justify replacing Notebook 09. The pre-registration says development gates lead to an untouched independent promotion test, and independent promotion only makes the challenger eligible for a controlled supervised pilot, not automatic production deployment. The route below therefore has explicit validation and supervised shadow/pilot stages before Notebook 09 can be retired.

# 1. Destination

The desired end state is a correction pipeline that accepts an annotated page alone at inference time. A clean page is not required to produce the annotation mask, observed-print decision, V7.8 reconstruction inputs or final corrected image. Clean pages remain development and validation truth where available, not runtime inputs.

**Target runtime contract  
**INPUT: annotated page only.  
RUNTIME EVIDENCE: annotation-detector confidence/mask, observed pixels, mask geometry, observable support/topology, frozen models/policies and reconstruction artefacts.  
OUTPUT: corrected page plus separate audit artefacts.  
NO CLEAN PAGE: registration against clean truth, clean-derived ink maps, truth labels and post-hoc outcome metrics are absent from production inference.

# 2. What already exists

| **Capability**                 | **Existing project evidence**                                                                                                                                                                   | **Role in target system**                                         |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Annotated-only mask generation | Notebook 06 reads images from source/annotated_only and writes draft masks plus confidence maps.                                                                                                | Becomes annotation-detection front end.                           |
| Human corrected-mask path      | Notebook 07 reviews annotated-only drafts and writes corrected masks/aligned pages/audit records.                                                                                               | Remains training-feedback and exception-review tooling.           |
| Production-like proposal path  | Notebook 09 consumes pilot incoming pages, resolves draft or reviewed masks, executes canonical V7.8 reconstruction/routing, applies the frozen structural policy and requires human decisions. | Current supervised incumbent and shadow comparator.               |
| No-clean inference principle   | V7.9.x specifies a learned observed-print classifier using observable, truth-free local evidence; clean registered pages are for development labels/scoring only.                               | V8.x can run after the mask has been created without clean truth. |
| Future exception model         | V7.5 describes progression from full manual review, to exception review, to independently validated automatic high/high cases.                                                                  | Governance shape for reducing Notebook 09 review burden.          |

# 3. Recommended transition architecture

- Annotation detector: produces a confidence map and candidate annotation mask directly from the annotated page.
- Mask authority layer: determines the authoritative intervention mask. During transition this may still be reviewer-corrected; the eventual automatic route needs its own independently validated acceptance policy.
- Observed-print layer: V8.x classifies eligible dark material inside the authorised removal mask as PRINT / ANNOTATION / UNCERTAIN using observable-only inputs.
- Correction layer: PRINT copies the observed pixel exactly; ANNOTATION follows the existing removal/reconstruction process; UNCERTAIN defaults to conservative V7.8/review behaviour.
- Integrity layer: no pixel outside the authorised removal mask may change; canonical reconstruction/routing equivalence remains regression-tested.
- Audit layer: preserve detector confidence, authoritative mask, observed-print probabilities/classes, routing maps, model/policy hashes, corrected output and any human decisions separately.

# 4. Staged pathway

| **Stage**                                      | **Objective**                                                                                                                                 | **Required evidence / gate**                                                                                                                                | **Operational status**                                           |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| A. Complete V8.0b development                  | Determine whether raw spatial topology improves the safe coverage frontier over the matched scalar baseline.                                  | Source-group OOF integrity; paired comparison; registered development safety gates; visual audit.                                                           | Research only. Notebook 09 remains incumbent.                    |
| B. Full-pixel V8.0b OOF                        | Move from sampled-centre screening to all eligible held-out pixels without changing the frozen experiment.                                    | Full-pixel source-group OOF metrics, catastrophic component scan, sparse/hard-negative evidence, stable replay.                                             | Research only.                                                   |
| C. Freeze V8.x challenger                      | Create one immutable observed-print bundle.                                                                                                   | Model/checkpoint, calibrator, channel/schema definition, thresholds, policy ID, software/input hashes and replay test.                                      | Frozen challenger, not deployed.                                 |
| D. Untouched independent promotion test        | Test the frozen V8.x bundle on new source groups whose truth has not influenced development.                                                  | All pre-registered independent gates, including false-preservation bounds, utility, minimum evidence, source stability and sparse/hard-negative behaviour.  | If passed: eligible for controlled supervised pilot only.        |
| E. Integrate V8.x into Notebook 09 shadow mode | Run V8.x beside the incumbent proposal path but do not let it change final output automatically.                                              | Byte-for-byte incumbent replay where V8.x does not assert PRINT; disagreement tables; component/page review; audit identity.                                | Notebook 09 remains decision authority.                          |
| F. Supervised V8.x pilot                       | Allow V8.x proposals to be presented to the existing reviewer and measure reviewer accept/leave/defer/edit outcomes.                          | Predeclared pilot gates and evidence that V8.x reduces review burden without safety regression.                                                             | Human final approval remains mandatory.                          |
| G. Validate annotated-only mask authority      | Establish that the detector-generated mask itself can be trusted for an automatic tier.                                                       | Separately frozen annotation-confidence acceptance policy; hard negatives, source-group stability and independent validation.                               | Critical because V8.x operates only inside the mask it is given. |
| H. Joint automatic-tier validation             | Validate the combined runtime chain: detector/mask authority + V8.x observed-print + correction/routing.                                      | Fresh end-to-end holdout; no clean inputs at inference; truth revealed only after outputs are frozen; outside-mask immutability; combined failure analysis. | Candidate production release.                                    |
| I. Annotated-only canary release               | Process production-like annotated-only pages with automatic output enabled only for the validated tier; retain exception review and rollback. | Release manifest identifies detector, observed-print policy, reconstruction version and configuration fingerprints; monitoring and rollback ready.          | Limited production.                                              |
| J. Retire Notebook 09 as primary UI            | Replace mandatory page/component review with exception-only handling once automatic coverage and monitoring are demonstrably sufficient.      | Documented release decision; stable monitoring; retained regression packs and reviewer fallback.                                                            | Notebook 09 becomes legacy/fallback, not deleted evidence.       |

# 5. The key missing validation is not a clean-page runtime dependency

Clean pages are required to establish truth during development and independent validation, but the target inference architecture does not need them. The harder transition issue is whether the system can trust the annotation mask without a reviewer. The observed-print classifier cannot repair an annotation region that the detector never places inside the authorised mask, and an erroneous broad mask increases the area exposed to correction logic.

- Therefore, V8.x success validates print preservation inside an authorised mask. It does not by itself validate fully automatic annotation-mask creation.
- The annotation detector and correction/print-safety mechanisms should remain separately versioned and separately promoted.
- A combined production release can be assembled only after both axes have their own frozen evidence and the integrated chain passes end-to-end regression and safety tests.

# 6. Proposed runtime flow after Notebook 09

| **Step** | **Annotated-only production action**                                                                              | **Fail-closed behaviour**                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1        | Read annotated page; record file identity and configuration.                                                      | Unreadable/unsupported input: no output.                                   |
| 2        | Run frozen annotation detector and write confidence map plus proposed mask.                                       | Integrity/model mismatch: no correction.                                   |
| 3        | Apply the independently validated mask-authority rule.                                                            | Non-authorised regions go to review or no-change.                          |
| 4        | Dilate/prepare the authorised removal mask using the frozen correction contract.                                  | Schema/settings mismatch: no correction.                                   |
| 5        | Compute observable V8.x channels and classify eligible dark pixels.                                               | Missing/non-finite/OOD/low confidence: UNCERTAIN.                          |
| 6        | Copy PRINT pixels exactly; route ANNOTATION through the existing correction path; treat UNCERTAIN conservatively. | No V8.x PRINT assertion means incumbent behaviour is retained.             |
| 7        | Assert outside-mask immutability and write audit artefacts before publishing output.                              | Any immutability/replay failure invalidates output.                        |
| 8        | Route exceptions to the reviewer; automatic-tier pages proceed only if all runtime gates pass.                    | Exception or policy uncertainty: review/no-change, not relaxed thresholds. |

# 7. What must replace Notebook 09, versus what should survive

| **Notebook 09 function**                                         | **End-state treatment**                                                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Canonical V7.8 structural implementation and immutability checks | Survive as a versioned library/regression contract until a separately validated correction algorithm replaces them. |
| Frozen policy/model integrity checks                             | Survive and expand to include the observed-print bundle and detector identity.                                      |
| Mask resolution from draft/review audit                          | Replace with a production mask-authority service/policy, while retaining reviewed-mask override for exceptions.     |
| Component green/amber/red queue                                  | Replace as the primary workflow with automatic/exception routing; retain equivalent reason codes in audit.          |
| Human accept / leave unchanged / defer                           | Retain for exceptions and evidence collection.                                                                      |
| Mask edit and full rescore                                       | Retain for exceptions and future authoritative training evidence.                                                   |
| Final page approval button                                       | Eventually replaced by release-policy approval for automatic-tier pages; still required for review-tier pages.      |
| Run manifest and decision history                                | Must survive; production audit should be stronger, not weaker.                                                      |

# 8. Release artefact identity

My suggested production release identity, building on the existing separately-versioned detector and safety-policy approach, is:

- annotation_detector_id: dataset release, checkpoint hash, inference configuration and validation report
- mask_authority_policy_id: acceptance/review thresholds and independent-validation report
- observed_print_policy_id: model/checkpoint, calibration, ordered channels/schema, thresholds and independent-validation report
- correction_engine_id: V7.8-compatible reconstruction/routing version and equivalence hash
- combined_release_id: hashes all four identities plus runtime configuration and release decision

# 9. Evidence needed before each reduction in human review

| **Human-review reduction**                          | **Evidence I would require**                                                                                                                    |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Use V8.x recommendations inside Notebook 09         | Independent observed-print promotion gates pass; exact incumbent regression elsewhere; transparent disagreement review.                         |
| Auto-accept some observed-print decisions           | Fresh integrated pilot shows frozen PRINT decisions remain within the registered safety envelope and do not create source-group concentrations. |
| Stop manually correcting every annotation mask      | Independent annotation-mask authority policy passes its own detector/hard-negative/source-group gates.                                          |
| Output corrected annotated-only pages automatically | End-to-end fresh holdout passes with frozen detector, mask policy, V8.x policy and correction engine; no clean page used during inference.      |
| Retire Notebook 09 as primary operator path         | Automatic tier has sufficient independently demonstrated coverage and stability; exception reviewer and rollback process remain available.      |

# 10. Morning V8.0b decision tree

- If V8.0b does not improve the matched safe frontier: do not progress toward Notebook 09 replacement on this hypothesis; retain the evidence and redesign the classifier experiment.
- If V8.0b improves ranking but not a safe operating point: perform only the pre-specified next representation/context ablation. Do not proceed to independent promotion.
- If V8.0b passes development gates after mandatory full-pixel scoring: freeze one challenger and commission the untouched independent test.
- If the independent V8.x test passes: integrate it into Notebook 09 in shadow/supervised fashion before considering any automatic tier.
- Only after the annotation-mask axis is independently acceptable should the integrated annotated-only automatic pathway be tested.

# 11. Risks that would block replacement

- V8.x benefit remains too concentrated in a few source groups or morphologies.
- False PRINT errors survive at high calibrated confidence, especially around annotation/print intersections.
- The annotation detector produces masks whose misses or overreach are not independently controlled.
- Integrated detector + V8.x behaviour differs from the independently validated standalone behaviour because of input distribution or mask changes.
- Exception/UNCERTAIN volume is so high that automatic operation is nominal rather than useful.
- Runtime artefacts cannot reproduce the frozen outputs exactly, or outside-mask immutability is violated.
- Reviewed production evidence is fed back into training without preserving fresh untouched holdouts.

# 12. Governing project evidence

- V7.5 design record: separates annotation confidence from correction safety; describes progression from full manual review to exception-based review and independently validated automatic high/high cases; requires distinct confidence, mask and routing audit artefacts.
- Notebook 06 / Notebook 07: already establish an annotated-only detection and human-mask-review route that does not require a clean counterpart at inference.
- Pilot Notebook 09: current supervised incumbent, with exact V7.8 structural processing, policy integrity checks, regression gate, proposal generation and explicit human accept/unchanged/defer/edit/final approval.
- V7.9.x record: requires future learned observed-print inference to remain observable/truth-free and states that clean truth belongs only in development labels/scoring.
- Observed-print pre-registration v1.1: development gates precede independent testing; independent success permits only a controlled supervised pilot; no automatic production promotion.
- Production retraining policy: keeps detector and correction-safety mechanisms separately versioned, promotion non-automatic, reviewed evidence authoritative for future training, and frozen independent sets preserved.

# 13. Bottom line

**Recommended endpoint  
**Do not plan to replace Notebook 09 with "V8.0b" directly. Plan to replace Notebook 09 with a frozen, independently validated annotated-only runtime stack in which (1) the detector defines a trustworthy intervention mask, (2) the observed-print model protects visible genuine print inside that mask, (3) the correction engine performs only authorised changes, and (4) UNCERTAIN or policy exceptions still have a reviewer path. Notebook 09 should be progressively demoted from mandatory reviewer to shadow comparator to exception reviewer to retained fallback.

**Confidence: 94%.** The no-clean runtime direction is strongly supported by the current annotated-only HITL inputs, Pilot 09 input resolution, V7.9.x truth firewall and V7.5 target architecture. The principal uncertainty is not whether the runtime can technically avoid a clean page; it is whether V8.x and the annotation-mask authority can both earn enough independent evidence to justify reducing human review.

**Next question:** What exact automatic mask-authority rule and independent validation gates must the annotation detector pass before a V8.x-corrected page can bypass human mask review?