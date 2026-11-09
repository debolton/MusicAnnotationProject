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

# 10. Comprehensive V9.0 research portfolio

V9.0 should be treated as a portfolio of separately governed experiments rather than a single notebook or a single attempt to operationalise V8.8. V8.8 remains frozen as the correction-engine baseline. Each V9.0 route changes one principal variable, retains the authoritative-mask boundary, preserves outside-mask immutability, and requires new evidence before it can affect the baseline.

**Portfolio rule: do not alter the V8.8 region policy, YOLO detector, observed-print decision, boundary rule and restoration method in one experiment. Separate the causal questions and preserve the frozen V8.8 comparator.**

| **ID**    | **Research route**                       | **Primary question**                                                                                                             | **Position**                             |
| --------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **V9.0A** | YOLO mask authority and transfer         | Does V8.8 retain its safety and benefit when reviewed masks are replaced by raw, gated or reviewer-corrected YOLO masks?         | Immediate operational prerequisite       |
| **V9.0B** | Selective cosmetic-residue cleanup       | Can faint residual annotation/shadowing be removed after V8.8 without damaging genuine print?                                    | Direct response to the V8.8 pilot        |
| **V9.0C** | Mask-boundary refinement                 | Can boundary-aware or signed-distance adjustment improve removal at edges without changing the frozen region decision?           | Open when failures cluster at boundaries |
| **V9.0D** | Uncertainty, abstention and OOD          | Can unstable pages, sources and neighbourhoods be identified and routed to exact fallback before an unsafe assertion?            | Parallel safety programme                |
| **V9.0E** | Multi-scale ownership context            | Does wider context improve print-versus-annotation ownership at crossings, broad loops and mixed regions?                        | Representation challenger                |
| **V9.0F** | Structure-conditioned restorer routing   | Can observable residual geometry select among frozen restorers after the mask is fixed?                                          | Open only after residual target freeze   |
| **V9.0G** | Learned non-generative restoration       | Can a trained residual restorer beat Telea, biharmonic and the existing learned U-Net on the same frozen residual mask?          | Separate restoration experiment          |
| **V9.0H** | Light/grey print enhancement             | Should genuine observed print preserved as grey be enhanced, and can this be done without inventing notation?                    | Separate from observed-byte preservation |
| **V9.0I** | Generative residual feasibility          | Can a model such as FLUX.1 Fill recover genuinely unsupported gaps without inventing musical content?                            | Late research-only strand                |
| **V9.0J** | Musical-structure and perceptual metrics | Can topology, notation-aware endpoints and blinded page review identify harms hidden by pixel averages?                          | Cross-cutting evaluation programme       |
| **V9.0K** | Two-axis exception routing               | Can annotation confidence and correction safety route automatic, quick-review and full-review cases without collapsing the axes? | Later integration experiment             |
| **V9.0L** | Fresh cohorts, drift and retraining      | How does the chain behave across new editions, scanners, handwriting and acquisition styles, and what triggers a challenger?     | Permanent evidence programme             |
| **V9.0M** | End-to-end annotated-only validation     | Does the complete frozen detector, mask-authority, V8.8 and correction route pass a fresh no-clean inference holdout?            | Final pre-canary integration gate        |

## 10.1 Selective cosmetic-residue cleanup

- Freeze V8.8 PRINT preservation and its M5 decisions. The experiment changes only a narrow cleanup layer applied inside already authorised areas.
- Create labels that distinguish remaining annotation/shadow residue from genuine observed print and paper texture. Include the clean/white regions where Notebook 09 appeared cosmetically cleaner and dense/mucky regions where V8.8 was clearly superior.
- Primary benefit: reduction in faint residual annotation. Primary safety endpoint: genuine printed ink removed by the cleanup layer. Dense-region preservation must be a hard regression gate.
- Compare no extra cleanup, fixed tonal cleanup, boundary-aware cleanup and one learned observable-only challenger. Do not select a policy from the 43 pilot pages alone.

## 10.2 Light or grey preserved print

- The current behaviour is exact observed-byte preservation, not failed black reconstruction. Notebook 09 can leave the same location white because it does not preserve the print at all.
- First determine whether light preserved print is actually a readability problem using blinded whole-page and local musical-legibility review. Do not assume darker is automatically better.
- If commissioned, compare exact-byte control with tightly bounded contrast restoration or local tone normalisation. Require identity of topology and prohibit new connected ink, new noteheads, accidentals, dots, stems, beams, bar lines or slurs.
- Treat enhancement as a separate output layer and version. It must never silently redefine the observed-print classifier or its truth labels.

## 10.3 Boundary refinement

- Freeze the M5 region decision and isolate only the boundary variable.
- Test controlled erosion, dilation, fragmentation and signed-distance perturbations around reviewed masks before fitting any learned boundary model.
- Measure boundary-local print loss, retained residue, M5 XOR, downstream page differences and sensitivity to YOLO contour jitter.
- Do not return to an outcome-selected dilation or collar sweep.

## 10.4 Uncertainty, abstention and OOD

- Retain the fail-closed principle: unstable evidence defaults to exact M3 or V8.8 fallback rather than a lower threshold.
- Assess probability calibration and instability by page, source group, scanner, handwriting style, mask depth, route context and neighbourhood morphology.
- Commission explicit OOD tests on new editions, sparse-print hard negatives, low-contrast scans, unusually broad masks and page-border cases.
- Report useful coverage at the registered safety envelope. Do not improve coverage by weakening the envelope.

## 10.5 Multi-scale ownership and topology

- The earlier V8.0 analysis identified ownership confusion where annotation and printed structure interleave within a few pixels. A 33 x 33 patch may not contain enough context in broad loops and mixed notation regions.
- Run a paired representation ablation with fixed folds, targets, calibration and decision rules. Compare 33 x 33 with one predeclared wider scale, preferably 97 x 97; use 193 x 193 only if already computationally durable.
- Keep page/source identity, raw coordinates, clean pixels and truth-derived maps outside the inference schema.
- Promote only a frontier improvement under the same catastrophic, false-preservation, source-stability and hard-negative constraints.

## 10.6 Restoration after the residual mask is frozen

- First test structure-conditioned routing among frozen Telea, biharmonic and the existing learned U-Net. The question is whether visible residual geometry can choose a better existing restorer.
- If routing is insufficient, commission a learned non-generative residual restorer using the identical frozen residual mask. Do not change mask construction and restoration architecture together.
- Use false inserted ink, missing print, ink/edge F1, topology continuity, outside-mask drift and blinded musical review. Generic SSIM or visual cleanliness cannot be the primary safety endpoint.
- Preserve the prior negative evidence: legacy continuation methods should not be revived through another threshold tweak without a materially different hypothesis.

## 10.7 Generative residual reconstruction

- Limit any generative tool to genuinely unsupported residual gaps after trusted observed print is copied exactly.
- Begin with synthetic occlusions where the exact missing notation is known, then test real-mask counterfactuals, then blinded musical-fidelity review. Human approval is mandatory throughout feasibility.
- Classify invented noteheads, accidentals, articulation dots, stem/beam connections, bar lines, slurs and other musically meaningful additions as failures even when the page looks natural.
- Maintain deterministic fallback and immutable boundaries. Generative work must remain a separate major strand, not a V8.8 patch.

## 10.8 Metrics and review science

- Add topology and notation-aware endpoints alongside MAE, SSIM, ink F1, edge F1, false inserted ink and missing ink.
- Use same-scale blinded whole-page review first, then diagnostic locators and clean truth. Preserve the initial judgement before diagnosis.
- Distinguish trace benefit, local legibility benefit, material notation recovery, critical recovery, harmless residue, structural harm and plausible change in musical meaning.
- Use perceptual evidence to supplement, not replace, pixel-level integrity and safety gates.

## 10.9 Two-axis automation and review routing

- Keep YOLO annotation confidence and correction safety separately versioned. High confidence on one axis does not establish high confidence on the other.
- Pre-register a matrix of annotation-confidence tier by correction-safety tier. Candidate outputs are automatic, quick review, full review, mask edit/rescore or no change.
- Retain raw confidence maps, authoritative masks, V8 decisions, routing maps and human outcomes as separate audit artefacts.
- Only independently validated high/high cases may enter an automatic tier. All other cases retain review or no-change behaviour.

## 10.10 New cohorts, monitoring and retraining

- Acquire new editions, scanners, engraving styles, handwriting styles and sparse-print hard negatives. Lock roles before outcomes are inspected.
- Maintain distinct development, audit reserve, hard-negative, independent promotion and permanent regression cohorts.
- Retrain YOLO when material reviewed evidence or demonstrated drift exists. Retrain the structural/correction policy only when its semantics, predictors, reconstruction or evidence materially change.
- Never use unreviewed predictions as truth and never promote a challenger automatically.

## 10.11 End-to-end annotated-only validation

- Open this only after the detector/mask-authority and correction-safety axes have passed separately.
- Freeze detector checkpoint, inference/tiling settings, mask-authority rule, V8.8 bundle, correction engine, exception rules and all identities.
- Run a fresh holdout with no clean image at inference. Lock masks, decisions and corrected pages before truth is revealed.
- Require zero outside-authority drift, source stability, acceptable detector overreach and misses, acceptable correction safety, measurable review reduction, monitoring and rollback.

# 11. Research sequencing and decision dependencies

| **Phase**                 | **Action**                                                                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Now**                   | Operationalise V8.8 with reviewed masks; run V9.0A YOLO mask-transfer shadow evidence; start monitoring/OOD evidence collection.          |
| **Next**                  | Commission V9.0B cosmetic-residue cleanup using V8.8 as frozen control. Open V9.0C only if boundary analysis shows the primary mechanism. |
| **Representation branch** | Open V9.0E when failures remain ownership/topology confusions rather than detector-boundary errors.                                       |
| **Restoration branch**    | Open V9.0F and then V9.0G only after the final residual mask is frozen. Keep V9.0H grey-print enhancement separate.                       |
| **Late research**         | Open V9.0I generative feasibility only after non-generative residual restoration is benchmarked and safety review is defined.             |
| **Integration**           | Build V9.0K two-axis exception routing, then V9.0M fresh end-to-end annotated-only validation.                                            |
| **Ongoing**               | Maintain V9.0J metrics and V9.0L cohorts, drift, retraining and regression evidence across every strand.                                  |

# 12. Common commissioning rules for every V9.0 strand

- Write a pre-registration or machine-readable experiment contract before outcomes are inspected.
- Name the one principal changed variable and the frozen comparator.
- Use page and source-group separation; keep clean truth out of inference.
- Lock predictions and outputs before truth scoring for independent stages.
- Report catastrophic cases, uncertainty, page/source concentration and hard-negative behaviour, not only pooled averages.
- Retain every incumbent and failed challenger with hashes, manifests, replay evidence and a written decision.
- Create a standard final upload package with executive summary, decision tables, audit, visuals, manifest and a curated ZIP.
- No experiment automatically promotes itself or modifies V8.8.

**Recommended commissioning order: V9.0A mask authority transfer, V9.0B cosmetic-residue cleanup, then open representation or restoration branches according to the observed failure mechanism. Do not treat all items as one combined V9.0 model.**

- V8_7_Final_Decision_and_V8_8_Direction.docx
- V8_7_Notebooks_49_50_51_Residual_Mask_Improvement_Programme 1.docx
- V8_7_Residual_Mask_Programme_Change_Record_After_Notebook_49R4.docx
- Future_Direction_Generative_Residual_Reconstruction_FLUX1_Fill.docx
- V8_3_Residual_Gap_Continuation_Closure_Record.docx
- Notebook38_Envelope_Relaxation_Project_Reference_FULL_DEFINITION_v1_0.docx

**Next question: Should V9.0 be governed as one umbrella programme with separately numbered strands, and which strand receives the first fresh independent cohort?**