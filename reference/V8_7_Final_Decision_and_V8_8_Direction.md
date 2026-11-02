**MUSIC ANNOTATION PROJECT**

V8.7 Final Decision

**SAFE_REMOVE, SAFE_PROTECT and the transition to V8.8**

_Project reference and forward plan | 2 November 2026_

**FINAL DECISION: retain and operationalise M5_REMOVE. SAFE_REMOVE is supported by observable-selection and causal reconstruction evidence. SAFE_PROTECT did not clear its independent gate and is parked. Do not combine it with SAFE_REMOVE or relax it into use.**

Status: V8.7 development decision record. V8.8 is the next operationalisation experiment, not immediate production replacement of Notebook 09.

# 1. Executive decision

V8.7 closes with a REMOVE-only mask challenger. The programme established a clean two-stage evidence chain: Notebook 50 showed that SAFE_REMOVE can be selected from observable evidence under fail-closed controls, and Notebook 51 showed that excluding those selected regions from the reconstruction target causes downstream reconstruction benefit. SAFE_PROTECT did not qualify for crossover and is not part of the retained mask policy.

| **Policy arm** | **Notebook 50**                                    | **Notebook 51**                                      | **Final V8.7 status**                                                    |
| -------------- | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| SAFE_REMOVE    | Passed the required route into crossover           | Tested as M5_REMOVE versus M3; causal gain supported | Retain. Freeze as the V8.7 residual-mask challenger and carry into V8.8. |
| SAFE_PROTECT   | Did not qualify for crossover                      | Not applied as an intervention                       | Park. Do not combine, threshold-relax or operationalise.                 |
| M5 combined    | Permitted only if both arms passed                 | Not tested                                           | Not supported and not the next route.                                    |
| M3 fallback    | Exact fallback for unsupported/uncertain locations | Comparator                                           | Retain as the fail-closed fallback during V8.8.                          |

_Source basis: V8.7 programme records; Notebook 50 verdict logic; Notebook 51 output package and formal verdict._

# 2. What V8.7 established

- The original four-class M3-to-M4 hypothesis was replaced by two asymmetric exclusion decisions inside M3: SAFE_REMOVE as the primary challenger and SAFE_PROTECT as the stricter secondary challenger.
- The decision arms were independent. A combined M5 was allowed only if both arms passed their own integrity, safety, stability and coverage gates.
- Only SAFE_REMOVE earned the route into Notebook 51, producing M5_REMOVE = M3 minus SAFE_REMOVE.
- Notebook 51 supplied causal development evidence because the mask changed while the restorers remained frozen.
- The principal benefit was concentrated in the conventional, mask-sensitive restorers, with coherent improvement in reconstruction quality and reduced false inserted ink; the learned U-Net changed very little.
- Between-restorer dispersion contracted across the tested endpoints, supporting the interpretation that some apparent restorer weakness was actually poor mask conditioning.
- The outcome is development evidence, not production authorisation. M4 remains diagnostic only.

# 3. SAFE_REMOVE decision

SAFE_REMOVE is accepted as the V8.7 residual-mask improvement mechanism. Its role is not to reconstruct pixels differently; it removes selected regions from the reconstruction burden so that the residual mask is semantically cleaner and less likely to provoke reconstruction-induced false ink.

- Retain the trained selection policy, ordered observable schema, thresholds, source-group fold evidence, hashes and replay artefacts.
- Retain exact M3 fallback wherever SAFE_REMOVE is unsupported, uncertain, invalid or unavailable.
- Preserve byte-for-byte immutability outside the authorised intervention mask.
- Keep mask authority, reconstruction routing, model probabilities and corrected output as separate audit artefacts.
- Do not describe SAFE_REMOVE as a production policy until V8.8 completes operational validation and a documented release decision is made.

# 4. SAFE_PROTECT decision

SAFE_PROTECT is closed as an unsuccessful V8.7 intervention arm at its registered specification. It may remain as a research baseline or diagnostic evidence, but it must not be added to M5_REMOVE merely because SAFE_REMOVE subsequently produced positive reconstruction results.

- Do not lower its threshold to force coverage.
- Do not merge failed PROTECT assertions into the successful REMOVE policy.
- Do not run a new combined REMOVE + PROTECT crossover without a materially new, pre-registered PROTECT hypothesis and fresh evidence.
- Do not imply that PROTECT was causally disproved in every possible form. The supported conclusion is narrower: the registered SAFE_PROTECT arm did not clear the route to intervention.

# 5. V8.8 objective: operationalise M5_REMOVE

**V8.8 objective: convert the successful development policy into a frozen, auditable, annotated-only correction pipeline that can progressively replace Notebook 09 after equivalence, independent validation, shadow use and supervised pilot evidence.**

Correction to the immediate replacement aim: V8.8 should aim to replace Notebook 09, but it should not declare Notebook 09 replaced at commissioning. The existing project pathway requires staged evidence before the reviewer can be retired. Notebook 09 should remain the incumbent comparator and fallback until those stages pass.

# 6. Proposed V8.8 work packages

| **Stage** | **Purpose**                            | **Minimum evidence**                                                                                                                                             | **Operating status**              |
| --------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| 8.8A      | Freeze the M5_REMOVE policy bundle     | Model/checkpoint, calibrator if applicable, ordered schema, thresholds, implementation identity, source-run hashes, policy ID and exact replay pack.             | Frozen challenger; no deployment. |
| 8.8B      | Build an operational wrapper           | Annotated page plus authoritative mask in; M5_REMOVE, reconstruction output, decision maps, manifests and failure records out. Exact M3 fallback on any failure. | Development integration.          |
| 8.8C      | Equivalence and regression audit       | Reproduce Notebook 50/51 decisions and M3/M5 arrays; zero outside-mask drift; exact replay; deterministic rerun; canvas/shape handling fixed and tested.         | Technical gate.                   |
| 8.8D      | Untouched or role-separated validation | Use data not used to choose the final policy; report safety, benefit, coverage, source-group stability, uncertainty and visual failures.                         | Promotion evidence only.          |
| 8.8E      | Notebook 09 shadow mode                | Run V8.8 beside Notebook 09. Do not let V8.8 change the approved output automatically. Record agreements, disagreements and reasons.                             | Notebook 09 remains authority.    |
| 8.8F      | Supervised pilot                       | Present M5_REMOVE proposals through an exception reviewer; measure accept, unchanged, defer and mask-edit/rescore outcomes.                                      | Human final approval mandatory.   |
| 8.8G      | Controlled replacement decision        | Replace Notebook 09 as the primary path only if automatic-tier coverage, stability, monitoring, rollback and exception handling are adequate.                    | Notebook 09 retained as fallback. |

# 7. V8.8 runtime and audit contract

- Input authority remains explicit: annotated page, detector/confidence artefacts, and either a reviewed mask or a separately validated mask-authority decision.
- M5_REMOVE may only subtract registered SAFE_REMOVE regions from M3. It must never expand the authorised intervention boundary.
- Any missing artefact, hash mismatch, non-finite value, shape mismatch, canvas mismatch, failed replay or unsupported region falls back to exact M3 or no-change according to the frozen contract.
- The corrected page must be byte-for-byte identical to the incumbent outside the authorised mask and at every location not validly changed by the frozen policy.
- Store separately: detector confidence, authoritative mask, M3, SAFE_REMOVE decision/probability, M5_REMOVE, reconstruction routing, corrected page, difference map, software identity and review outcome.
- Preserve the systematic Notebook 50 canvas-padding issue as an explicit regression test so operational code cannot silently recreate it.

# 8. Candidate V8.8 success gates

Exact numerical promotion limits should be pre-registered before running the definitive V8.8 validation. The following are the required gate families, not invented final thresholds:

- Integrity and replay: all artefacts resolve uniquely and reproduce exact decisions and output bytes.
- Immutability: zero outside-mask changes and zero unapproved deviations from the incumbent path.
- Policy fidelity: operational M5_REMOVE equals the registered Notebook 50 policy on the frozen regression roster.
- Reconstruction evidence: the Notebook 51 benefit direction is reproduced without restorer retuning.
- Source stability: no aggregate pass conceals concentrated failure in a score, scan style or source group.
- Visual safety: false-ink-heavy, missing-ink-heavy, boundary and disagreement cases receive structured review.
- Operational usefulness: review burden and exception volume are measured; low coverage is reported honestly rather than solved by relaxing safety.
- Rollback: the incumbent Notebook 09/M3 route remains available and version-addressable.

# 9. Deferred V9.0 research routes

V9.0 is not commissioned. The following routes should be retained as a structured research backlog and should only be opened after V8.8 has frozen the operational baseline and exposed the remaining failure modes.

| **Research route**                           | **Question and method**                                                                                                                                                                   | **Reason to defer / guardrail**                                                                                                                            |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mask-boundary refinement                     | Learn a boundary-aware correction or signed-distance adjustment around M5_REMOVE, evaluated separately from region classification.                                                        | Do not revert to another hand-tuned dilation/collar sweep. Freeze the region policy and isolate the boundary variable.                                     |
| Uncertainty-aware M5                         | Model abstention and out-of-distribution behaviour explicitly, with M3 fallback for unstable sources or neighbourhoods.                                                                   | Operational errors may arise from confidence failure rather than class separation.                                                                         |
| Multi-scale ownership context                | Test whether larger spatial context improves REMOVE discrimination at crossings, broad loops and mixed notation/annotation regions.                                                       | Use a controlled ablation. Do not change context, target and policy simultaneously.                                                                        |
| Structure-conditioned reconstruction routing | After M5 is frozen, test whether residual geometry can select among frozen conventional or learned restorers without using clean truth at inference.                                      | Notebook 51 showed restorer sensitivity falls with better masks; routing must add value beyond that simpler fix.                                           |
| Learned non-generative restoration           | Train a residual restorer only on the final frozen residual target, comparing it against Telea, biharmonic and the existing learned U-Net.                                                | Requires a new experiment because both mask and restorer would otherwise change together.                                                                  |
| Perceptual / musical-structure metrics       | Add topology and notation-aware endpoints plus blinded page review alongside MAE, SSIM, ink F1, edge F1, false inserted ink and missing ink.                                              | A metric expansion must not replace pixel-level safety or hide losses.                                                                                     |
| Generative residual reconstruction           | Consider a separately governed feasibility study, potentially including FLUX.1-Fill, restricted to genuinely unsupported residual gaps after trusted observed print is preserved exactly. | Synthesis is a different scientific problem. Keep it separate from V8.8 and require deterministic fallback, strict boundaries and blinded fidelity review. |
| New independent cohorts                      | Acquire new editions, scanners, handwriting styles and sparse-print/hard-negative pages with role-locked manifests.                                                                       | Repeated tuning on the same pages will overfit the operational policy and weaken later claims.                                                             |

# 10. Decisions and actions

- Close V8.7 with M5_REMOVE as the retained challenger.
- Close SAFE_PROTECT as a non-qualifying intervention arm at its registered specification.
- Do not commission combined REMOVE + PROTECT testing.
- Commission V8.8 as the operationalisation and staged Notebook 09 replacement programme.
- Keep Notebook 09 as incumbent authority, comparator and fallback until V8.8 gates are completed.
- Defer V9.0. Preserve the research routes above as backlog items, not active work.

# 11. Confidence and open decision

Confidence: 97% in the final V8.7 decision. The policy lineage, asymmetric gate logic, REMOVE-only crossover and causal reconstruction result are mutually consistent. Confidence is lower on the exact V8.8 promotion thresholds because those limits have not yet been frozen in the supplied project evidence; this document therefore specifies gate families and sequencing rather than inventing numerical acceptance criteria.

**Next question:** What exact independent-validation cohort, promotion thresholds and shadow-pilot exit criteria should be pre-registered before the V8.8 operational wrapper is built?

# Source register

- V8_7_Notebooks_49_50_51_Residual_Mask_Improvement_Programme 1.docx
- V8_7_Residual_Mask_Programme_Change_Record_After_Notebook_49R4.docx
- 50_V8_7_Asymmetric_Residual_Mask_M5_Challenger.ipynb
- Notebook 51 output package supplied for review, including the REMOVE-only pre-registration, metrics, restorer-gap analysis, formal verdict, identity and completion records
- 09_Pilot_Proposed_Correction_Routing_Reviewer.txt
- Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx
- Production_Retraining_Policy_YOLO_and_Structural_Safety.docx
- V8_3_Residual_Gap_Continuation_Closure_Record.docx
- V8_6_Residual_Reconstruction_Readiness_Programme.docx
- Future_Direction_Generative_Residual_Reconstruction_FLUX1_Fill.docx