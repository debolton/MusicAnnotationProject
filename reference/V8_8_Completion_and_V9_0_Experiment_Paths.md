V8.8 Completion Record

Completion status, operational boundary, YOLO-mask interaction, and proposed V9.0 experiments

Music Annotation Project | 9 November 2026

Decision: document V8.8 as complete as an operational engineering qualification and controlled release-candidate package. Do not interpret completion as proof that raw YOLO masks can replace reviewed masks. That remains a separate mask-authority experiment and should be the first V9.0 track.

| **Project stage**              | V8.8 completion and V9.0 commissioning options                                                |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| **V8.8 status**                | Complete as a release-candidate operationalisation programme                                  |
| **Runtime mask authority**     | Reviewed/corrected mask, or a separately validated mask-authority decision                    |
| **Next operational objective** | Run V8.8 against detector-generated masks in shadow mode before any automatic authority claim |
| **Confidence**                 | 92% overall; see the explicit confidence note in Section 8                                    |

# 1. Completion statement

V8.8 is complete for its defined purpose: turning the frozen V8.7 SAFE_REMOVE policy into a deterministic, auditable correction-engine candidate with replay, operational wrapper, equivalence, shadow, supervised-pilot, release-control and registered-candidate evidence. Completion does not mean that V8.8 has independently validated the annotation detector or authorised raw YOLO masks as the intervention boundary.

The correct completion claim is: "V8.8 is complete as the qualified correction engine behind an authoritative mask." The incorrect claim would be: "V8.8 is complete as a fully automatic annotated-only system using unreviewed YOLO masks."

# 2. V8.8 completion basis

The project sequence defined V8.8 as an engineering qualification programme, not a new mask-learning experiment. The completed sequence covered the following controls:

| **Control**                    | **Completion evidence**                                                                                                                                                                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Freeze and replay**          | The SAFE_REMOVE / M5_REMOVE policy bundle, ordered schema, thresholds and implementation identity were frozen and replay-tested.                                                                                                                           |
| **Operational builder**        | A reusable page pipeline was built from annotated page plus authoritative mask through M3, SAFE_REMOVE, M5_REMOVE, correction output and audit artefacts.                                                                                                  |
| **Equivalence and regression** | Operational outputs were compared with the development golden masters, including exact M5 decisions, outside-authority immutability and repeat-run determinism.                                                                                            |
| **Shadow operation**           | V8.8 was run beside Notebook 09 without allowing V8.8 to determine the approved output automatically.                                                                                                                                                      |
| **Supervised pilot**           | Human accept, unchanged, defer and mask-edit/rescore decisions were retained, with final approval remaining mandatory.                                                                                                                                     |
| **Release control**            | Release identity, monitoring, rollback, launcher integration and the continuing Notebook 09 fallback role were defined.                                                                                                                                    |
| **Registered replay**          | The registered-candidate rerun used the strict registered policy rather than a post-outcome relaxed threshold. The evidence supplied for this record reports 43 pages in total, comprising 36 registered replay pages and 7 Notebook 09 shadow-only pages. |

**Formal status wording.** V8.8 should be marked "complete, release-candidate qualified, controlled-authorisation pending" until the actual AUTHORISE_RELEASE control is deliberately set and the generated release manifest is accepted. The dry-run evidence that intentionally leaves authorisation false is a governance guardrail, not an incomplete experiment.

# 3. What V8.8 does at runtime

- Accepts an annotated page and an authoritative annotation/removal mask.
- Reconstructs the frozen M3 treatment and applies only the registered SAFE_REMOVE subtraction to create M5_REMOVE.
- Never expands the authorised intervention boundary.
- Falls back to exact M3 treatment, or no change where specified, for unsupported, uncertain, invalid, missing or mismatched cases.
- Keeps pixels outside the authoritative mask byte-for-byte identical to the observed page.
- Keeps detector confidence, authoritative mask, M3, SAFE_REMOVE probability/decision, M5_REMOVE, routing, corrected page, difference map, software identity and review outcome as separate audit artefacts.
- Does not reintroduce SAFE_PROTECT, M4 truth diagnostics, threshold relaxation or outcome-informed tuning.

# 4. Interaction with YOLO masks versus handmade masks

V8.8 does not inherently require a handmade mask. It requires an authoritative mask with the same semantic contract as the masks used during development and qualification. A YOLO mask can technically be supplied to the pipeline, but its status changes the safety question.

| **Mask input**                               | **Authority status**            | **Expected V8.8 behaviour**                                                                                        | **Primary risk**                                                                                                        |
| -------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| **Reviewed handmade/corrected mask**         | Authoritative                   | Use as qualified. V8.8 may subtract only registered SAFE_REMOVE regions from M3.                                   | Residual reviewer error, registration mismatch, or out-of-distribution page.                                            |
| **Raw YOLO binary mask**                     | Proposal only                   | Run in shadow mode. Produce output and audit, but do not treat as approved intervention authority.                 | False-positive mask area exposes genuine print to correction; false-negative mask area is never considered for removal. |
| **YOLO mask with confidence map**            | Proposal plus detector evidence | Preserve confidence separately. Use confidence for routing/review, not as a substitute for V8.8 correction safety. | Detector confidence and correction safety answer different questions.                                                   |
| **YOLO mask after automated authority gate** | Conditionally authoritative     | Eligible only after a separately frozen and independently validated mask-authority policy passes.                  | Distribution shift, broad-mask overreach, missed annotation, calibration drift and source-group concentration.          |
| **YOLO mask with human edits**               | Reviewed authoritative mask     | This is the nearest operational bridge from today's system to annotated-only production.                           | Review burden and edit provenance must be measured and retained.                                                        |

## 4.1 The asymmetric effect of detector errors

- YOLO false positive: the mask includes printed notation or blank regions that are not annotation. V8.8 can reduce some overreach through SAFE_REMOVE decisions and exact M3 fallback, but it cannot prove that the proposed intervention boundary itself was justified.
- YOLO false negative: annotation is omitted from the mask. V8.8 will not remove it because the system is prohibited from changing pixels outside the authoritative boundary. This is cosmetically undesirable but structurally safer than boundary expansion.
- Boundary jitter: a YOLO contour is slightly narrower, wider or fragmented relative to the reviewed mask. Dilation, connected-region construction and downstream SAFE_REMOVE features can change materially, so exact equivalence to the reviewed-mask run should not be assumed.
- Confidence-map disagreement: high detector confidence can still coincide with a structurally risky correction. The two axes must remain separate.

# 5. Immediate operationalisation plan with YOLO masks

| **Stage**                           | **Input**                                             | **V8.8 role**                                                 | **Decision authority**                                   | **Exit criterion**                                                                                          |
| ----------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Y0: Paired replay**               | Same pages with reviewed mask and frozen YOLO output  | Run both masks through identical V8.8 code                    | Reviewed-mask output remains comparator                  | Complete paired mask/output inventory with zero implementation drift.                                       |
| **Y1: Shadow mask mode**            | Raw YOLO mask + confidence map                        | Generate M3/M5/corrected page and exceptions                  | Human-reviewed mask/output controls approval             | Quantify mask IoU, overreach, misses, output differences, false structural changes and residual annotation. |
| **Y2: Edit-distance pilot**         | YOLO mask corrected by reviewer                       | Measure additions, erasures, time and downstream reprocessing | Reviewer mask is authoritative                           | Stable low edit burden, no source-group failure concentration, durable audit trail.                         |
| **Y3: Candidate authority gate**    | Frozen detector plus predeclared mask acceptance rule | Accept only a narrow high-confidence mask tier                | Automatic only within validated tier; remainder reviewed | Independent upper bounds, hard-negative pass, source stability and end-to-end immutability.                 |
| **Y4: Integrated supervised pilot** | Detector authority tier + V8.8 correction engine      | Produce final candidates and exception queue                  | Human final approval                                     | Review burden reduced without structural or cosmetic safety regression.                                     |
| **Y5: Controlled automatic tier**   | Fresh annotated-only cohort                           | Automatic output only when all gates pass                     | Release policy                                           | End-to-end holdout passes; monitoring and rollback are live.                                                |

# 6. Proposed V9.0 experiment paths

V9.0 should not be another broad threshold sweep. The strongest progression is to treat V8.8 as the frozen correction engine and test one new causal variable at a time. The recommended programme is below.

## V9.0A - YOLO Mask Authority and Transfer Audit

**Recommended first experiment**

| **Question**         | Test whether V8.8 retains its safety and benefit when the authoritative reviewed mask is replaced by a frozen YOLO proposal, with and without human correction.               |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Design**           | Paired page design: reviewed mask, raw YOLO mask, confidence-gated YOLO mask, and reviewer-corrected YOLO mask. Keep V8.8 frozen.                                             |
| **Primary measures** | Mask overreach, missed annotation, boundary displacement, output XOR, outside-mask drift, structural damage, residual annotation, edit distance/time, source-group stability. |
| **Why it fits**      | This is the missing prerequisite for operating V8.8 on real detector masks.                                                                                                   |

## V9.0B - Selective Residual Annotation Cleanup

**Second priority**

| **Question**         | Address the residual light-grey annotation and dense-region failures that remain after SAFE_REMOVE, without resynthesising broad musical structure.       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Design**           | Learn or calibrate a narrow residual-cleanup policy only inside the already authorised mask and only after V8.8 output. Compare against no extra cleanup. |
| **Primary measures** | False removal of print, cosmetic residual reduction, grey-mark recall, page-clustered uncertainty, catastrophic regions, perceptual review.               |
| **Why it fits**      | Directly targets the residual weakness reported after the registered replay.                                                                              |

## V9.0C - Boundary-Robust M5 Under Detector Jitter

**Use if V9.0A shows boundary sensitivity**

| **Question**         | Make SAFE_REMOVE decisions stable to plausible YOLO contour erosion, dilation, fragmentation and tile seams.                                                               |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Design**           | Pre-register controlled mask perturbations around reviewed masks; run the frozen engine and measure decision stability. Train only if observable robustness signal exists. |
| **Primary measures** | Decision flip rate, M5 XOR, page/output drift, structural harm, confidence localisation, failure by boundary depth.                                                        |
| **Why it fits**      | Separates detector boundary noise from true policy failure.                                                                                                                |

## V9.0D - Joint Two-Axis Exception Router

**Later operational experiment**

| **Question**         | Combine detector confidence and V8.8 correction evidence as two separate gates that route automatic, quick-review and full-review cases. |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Design**           | No single blended score. Pre-register a matrix of annotation-confidence tier × correction-safety tier.                                   |
| **Primary measures** | Automatic coverage, false-authority rate, false-safe correction rate, review burden, calibrated uncertainty, source stability.           |
| **Why it fits**      | Implements the original two-axis architecture without hiding either failure mode.                                                        |

## V9.0E - New Restoration Challenger for Dense Musical Regions

**Research track, not first**

| **Question**         | Test a materially different reconstruction method only for residual regions where V8.8 leaves M3 behaviour and legacy restoration remains inadequate. |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Design**           | Freeze the mask first. Compare restorer arms on identical residual masks. Keep synthesis strictly inside the final residual mask.                     |
| **Primary measures** | Printed topology fidelity, hallucinated notation, residual annotation, edge/staff continuity, uncertainty and whole-page perceptual harm.             |
| **Why it fits**      | Addresses reconstruction limitations without confusing them with mask quality.                                                                        |

## V9.0F - Monitoring and Drift Cohort

**Production research companion**

| **Question**         | Establish whether detector and V8.8 behaviour drifts across scanners, editions, handwriting styles and acquisition sources. |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Design**           | Prospective cohort with source-group labels; no threshold adjustment inside the monitoring run.                             |
| **Primary measures** | Mask edit burden, automatic-tier coverage, exception rate, safety indicators, calibration drift and rollback triggers.      |
| **Why it fits**      | Turns operational evidence into disciplined retraining triggers.                                                            |

# 7. Recommended V9.0 sequence

Commission V9.0A first. Do not start by retraining V8.8 or relaxing SAFE_REMOVE. The unresolved operational variable is mask authority, not the correction engine.

- Freeze the exact YOLO checkpoint, inference configuration, tiling/merging logic, confidence calibration and mask post-processing before comparison.
- Use the same annotated pages and the same frozen V8.8 engine for every mask arm.
- Lock all mask outputs and V8.8 outputs before clean truth or reviewed-mask comparisons are exposed.
- Treat the reviewed mask as the comparator, not automatically as perfect truth. Record reviewer additions, erasures and uncertainty.
- Keep V8.8 completion intact. Any learned change prompted by V9.0A becomes a new challenger version, not an edit to V8.8.
- If V9.0A passes a narrow authority tier, integrate that tier into the V8.8 exception workflow. If it fails, continue with reviewer-corrected YOLO masks and use the failure taxonomy to drive detector retraining.

# 8. Confidence and limitations

**Confidence: 92%.** Confidence is high that V8.8 is complete for its defined engineering qualification purpose and that the next operational question is the mask-authority transfer from reviewed masks to YOLO-generated masks. Confidence is lower on the exact numerical V8.8 release statistics because the most recent search did not expose the complete Notebook 57R1 output package as a directly retrievable file in this turn. The completion record therefore uses the supplied Notebook 57R1 evidence and preserves the distinction between a completed candidate package and deliberate production authorisation.

**Important uncertainty.** V8.8 was qualified behind the controlled-mask contract. Performance under raw YOLO masks cannot be inferred from reviewed-mask performance. It must be measured in a paired, role-separated experiment.

# 9. Source basis

- V8_8_Operationalisation_Workplan.docx
- V8_7_Final_Decision_and_V8_8_Direction.docx
- Notebook 57R1 registered-candidate rerun and release-control outputs supplied in the project workspace
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx
- Production_Retraining_Policy_YOLO_and_Structural_Safety.docx
- Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx
- 09_Pilot_Proposed_Correction_Routing_Reviewer.txt
- V7_9_x_Research_Findings_and_Next_Steps.docx
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx
- V8_7_Residual_Mask_Programme_Change_Record_After_Notebook_49R4.docx

Next question: What frozen YOLO checkpoint, post-processing configuration and paired page cohort should define the V9.0A mask-authority transfer experiment?