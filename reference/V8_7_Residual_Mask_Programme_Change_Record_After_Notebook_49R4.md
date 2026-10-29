**MUSIC ANNOTATION PROJECT**

V8.7 Residual-Mask Programme  
Change Record

Evidence from Notebooks 49 / 49R4 and revised direction for Notebooks 50 / 51

_Project reference | 29 October 2026 | Development-only research record_

**Change decision** The original V8.7 four-class M3-to-M4 formulation is superseded as the working hypothesis. Notebook 49 found no RECONSTRUCT population inside M3, while 49R4 demonstrated coherent and observably distinguishable REMOVE and PROTECT subregions and confirmed that M4 is set-disjoint from M3 on the completed roster. Notebook 50 should therefore be pre-registered as two asymmetric exclusion challengers inside M3, with REMOVE primary, PROTECT secondary, and exact M3 fallback for every unsupported or uncertain location.

**Status: superseding research-direction record  
Not production authorisation**

# 1. Executive decision

The V8.7 programme remains active, but its causal model must change. The original specification assumed that M3 could be decomposed into PROTECT, REMOVE, RECONSTRUCT and UNCERTAIN and that M4 could act as an oracle target for improving M3. The completed diagnostic work does not support that formulation. The correct next experiment is narrower: identify coherent subregions already inside M3 that can be safely excluded from reconstruction because they are either removable material or trustworthy observed print. Everything else remains M3.

| **Question**                  | **Previous direction**                                                                      | **Revised direction**                                                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| What is M5 learning?          | Four-way routing including an explicit RECONSTRUCT truth class inside M3.                   | Two asymmetric exclusion decisions: SAFE_REMOVE and SAFE_PROTECT. Neither decision passes => exact M3 fallback.                                     |
| What role does M4 play?       | Oracle residual against which M3/M5 distance and reconstruction recovery could be measured. | Diagnostic reconstruction ceiling / separate clean-print oracle set. Do not use M4 as the class label or presumed subset of M3.                     |
| What is the decision unit?    | Pixel and/or residual region, unresolved.                                                   | Local spatial evidence followed by coherent connected-subregion decisions. Whole M3 components are too coarse.                                      |
| Which challenger is primary?  | Single general M5 policy.                                                                   | REMOVE primary; PROTECT secondary and materially stricter.                                                                                          |
| What happens to uncertainty?  | UNCERTAIN remains M3.                                                                       | Unchanged: fail closed to exact M3.                                                                                                                 |
| What should Notebook 51 test? | M3 vs M5 vs M4 crossed with frozen restorers.                                               | M3 vs revised M5 across frozen restorers; retain M4 only as a diagnostic ceiling/reference, not a deployable mask arm assumed commensurate with M3. |

# 2. What changed since the original V8.7 specification

## 2.1 Original programme hypothesis

The original programme was a sensible pre-49 design: Notebook 49 would diagnose the M3-to-M4 oracle gap; Notebook 50 would build one observable-only M5 challenger; Notebook 51 would then test whether mask conditioning improved frozen reconstruction methods. It interpreted M3 as containing PROTECT, REMOVE, RECONSTRUCT and UNCERTAIN material and made M3 the fail-closed fallback.

## 2.2 Notebook 49 changed the evidential picture

The completed Notebook 49 decomposition did not produce the intended RECONSTRUCT reference population within M3. That made the planned PROTECT/REMOVE-versus-RECONSTRUCT separability test non-identifiable rather than demonstrating that observable mask learning was impossible. The run nevertheless yielded a locked M3 population and useful truth-side PROTECT, REMOVE and UNCERTAIN diagnostics.

## 2.3 Notebook 49R4 resolved the key ambiguity

Notebook 49R4 was deliberately read-only. It reused the completed Notebook 49 artefacts, built page-global 8-connected truth-side subregions, measured coherence and descriptive observable separability, and explicitly diagnosed the set relationship between M3 and M4. Its interpretation boundary prohibited model fitting or M5 selection.

**Result interpretation** 49R4 supports proceeding to a new pre-registration, not directly to a trained M5. The evidence is strong enough to change the target and decision unit, but it remains development evidence and does not validate an operational policy.

# 3. Evidence that now governs the programme

## 3.1 Spatial coherence supports connected-subregion decisions

| **Truth-side class** | **Pixels** | **Connected subregions** | **Pixels in runs ≥2** | **Source groups** | **Subregions touching another class** |
| -------------------- | ---------- | ------------------------ | --------------------- | ----------------- | ------------------------------------- |
| REMOVE               | 25,469     | 2,855                    | 99.1%                 | 17                | 3.8%                                  |
| PROTECT              | 890        | 194                      | 91.6%                 | 13                | 83.5%                                 |
| UNCERTAIN            | 8,085      | 1,028                    | 97.2%                 | 16                | 28.6%                                 |

Interpretation: REMOVE is a broad, highly coherent target whose connected subregions seldom touch another truth class. PROTECT is much smaller and usually lies against another class, making an entire-component or permissive preserve decision unsafe. UNCERTAIN remains substantial and coherent enough to justify keeping it as a genuine fail-closed state rather than forcing a decision.

## 3.2 Observable signal is strong enough to justify modelling

49R4 found high descriptive one-feature separation for both targets. REMOVE showed especially strong signal in p_print summaries and distance from protected print, with broad source-group eligibility. PROTECT also showed strong local ambiguity, adjacency and p_print contrasts, but with fewer source groups meeting the per-group sample rule. These are descriptive diagnostics, not held-out multivariate performance, so they justify pre-registration rather than threshold deployment.

| **Target** | **Illustrative 49R4 evidence**                                                                                                                                | **Programme consequence**                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| REMOVE     | p_print mean direction-free AUC about 0.951; distance-to-protected features about 0.87–0.89; leading comparisons available across 11 eligible source groups.  | Primary Notebook 50 challenger. Region-level coherence/safety gate; optimise safe burden removal.                                                      |
| PROTECT    | local ambiguity and adjacent-UNCERTAIN features roughly 0.92–0.93; p_print summaries roughly 0.90; fewer eligible source groups at the per-group sample rule. | Secondary, stricter challenger. Require local spatial evidence and high-confidence connected runs; prioritise false-preservation safety over coverage. |

## 3.3 M4 semantics invalidate the old oracle-subset assumption

49R4 reconstructed M4 from Notebook 45 durable mask artefacts and found M3 and M4 set-disjoint on the completed roster: M3 contained 34,444 pixels, M4 contained 2,605, and their intersection was zero. All 2,605 M4 pixels aligned to clean printed ink, while only 489 were dark in the observed image. The operational conclusion is not that M4 is wrong; it is that M4 is not the purified subset of M3 that the original Notebook 50 formulation implicitly required.

**Revised M4 role** Retain M4 as post-lock diagnostic evidence about unsupported clean print and as a reconstruction ceiling/reference. Do not train an M3 classifier to reproduce M4 membership and do not call M3-to-M4 overlap an ordinary mask-refinement target.

# 4. Revised conceptual mask model

The next development target is an exclusion problem inside M3, not a four-way oracle imitation problem.

| **M3 subregion decision** | **Evidence meaning**                                                                                                                       | **M5 action**                                                                            | **Failure posture**                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| SAFE_REMOVE               | Observable evidence supports that the subregion does not require reconstruction and should follow the frozen removal/background treatment. | Exclude from residual reconstruction target; apply the pre-registered removal treatment. | If confidence/coherence/safety fails, do nothing: retain M3.   |
| SAFE_PROTECT              | Observable evidence supports that trustworthy genuine observed print is being surrendered to reconstruction.                               | Exclude from residual reconstruction target and preserve exact observed bytes.           | Higher safety bar; any ambiguity or adjacency risk retains M3. |
| NO ASSERTION / UNCERTAIN  | Neither exclusion decision is sufficiently supported.                                                                                      | Exact M3 treatment.                                                                      | This is the intended fail-closed default, not a defect.        |
| M4 diagnostic             | Post-lock clean-derived unsupported-print evidence outside the deployable inference contract.                                              | Scoring / ceiling only.                                                                  | Never an inference input or direct M5 target.                  |

# 5. Revised Notebook 50 pre-registration direction

## 5.1 New scientific question

**Notebook 50 question** Can observable-only local spatial evidence, evaluated out of fold by source group, safely identify coherent REMOVE and PROTECT subregions within frozen M3, so that the residual reconstruction burden is reduced without structural harm, while every unsupported location retains exact M3 treatment?

## 5.2 Primary and secondary hypotheses

- Primary REMOVE hypothesis: observable-only evidence can identify coherent M3 subregions that do not require reconstruction and route them through a frozen removal/background treatment, reducing reconstruction burden without damaging genuine print.
- Secondary PROTECT hypothesis: a stricter local-spatial policy can identify a smaller coherent population of trustworthy observed print within M3 and preserve the exact observed bytes safely.
- Null/fallback hypothesis: if either target fails grouped safety, stability or usable-coverage gates, retain M3 for that target. The other target may still proceed if separately supported.

## 5.3 Decision and evaluation levels

| **Level**                     | **Role**                                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| Pixel / local patch           | Learn local ownership/topology. Candidate model sees observable spatial evidence, not truth. |
| Connected candidate subregion | Operational unit at which SAFE_REMOVE or SAFE_PROTECT is asserted.                           |
| Parent M3 component           | Concentration / catastrophe accounting; never assumed to have one truth class.               |
| Page                          | Clustered uncertainty and failure concentration.                                             |
| Source group                  | Primary generalisation and OOF separation layer.                                             |

## 5.4 Asymmetric model policy

REMOVE and PROTECT should not be collapsed into one symmetric classifier merely for convenience. Their errors have different structural meanings and 49R4 shows materially different spatial behaviour.

| **Policy** | **Recommended posture**                                                                                                               | **Why**                                                                                             |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| REMOVE     | Primary arm. Test scalar baseline plus local-spatial challenger; group compatible high-confidence decisions into coherent subregions. | Large population, 99.1% run coherence, all 17 source groups, comparatively low cross-class contact. |
| PROTECT    | Secondary arm. Local-spatial model with a stricter probability/margin/coherence rule and no singleton or weak-boundary preservation.  | Smaller population and 83.5% of subregions touch another class; false PROTECT retains annotation.   |
| UNCERTAIN  | No learned action.                                                                                                                    | Protects the incumbent when evidence is weak, invalid, OOD or conflicting.                          |

## 5.5 Inputs and firewall retained

- Observed greyscale and background-normalised local context.
- Authoritative mask geometry, depth and distance features.
- Frozen V7.8 horizontal, vertical and compact support and route context.
- Saved V8 evidence such as p_print where lineage is OOF and observable-only.
- Local edges/topology and connected-run geometry defined before outcome evaluation.
- No clean pixels, clean-derived maps, truth labels, reconstruction outcomes, page/source identifiers or raw coordinates as predictors.

## 5.6 Training and validation

- Source-group OOF is the default outer separation; page grouping remains mandatory where group structure requires it.
- All preprocessing, model fitting, calibration and threshold selection occur only inside training/OOF development evidence.
- A whole-page provisional prediction map is assembled before connected-subregion filtering; grouping must not depend on truth at inference.
- REMOVE and PROTECT receive separately calibrated decisions and separately pre-registered safety gates.
- Any missing, non-finite, schema-mismatched, OOD or unsupported case produces no challenger assertion and therefore remains M3.

## 5.7 Success measures to pre-register

| **Family**  | **REMOVE primary**                                                                                                           | **PROTECT secondary**                                                                                     |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Safety      | Genuine print wrongly routed out of reconstruction/removal damage; catastrophic structural concentrations; source stability. | Annotation falsely preserved; false-heavy concentrations; source stability and overlap-boundary failures. |
| Benefit     | M3 pixels/area safely removed from reconstruction; cleaner boundary conditions; reduced residual complexity.                 | Genuine observed print safely excluded from reconstruction and copied exactly.                            |
| Coverage    | Connected subregions and pixels safely asserted, pages/groups benefiting.                                                    | Connected high-confidence runs and recovered print, with coverage explicitly secondary to safety.         |
| Uncertainty | Page-clustered intervals plus source-group reporting.                                                                        | Same, with particular attention to limited positive support by source group.                              |
| Integrity   | Zero intervention outside authority and exact M3 fallback elsewhere.                                                         | Zero intervention outside authority and exact observed byte at protected coordinates.                     |

Exact numerical Notebook 50 thresholds should be frozen in the Notebook 50 pre-registration. 49R4 does not establish operational probability cut-offs, and its high univariate AUCs must not be converted into handcrafted thresholds after the fact.

# 6. Revised M5 construction

The deployable development challenger should be easy to state and audit:

**M5 construction** M5 starts as exact M3. Remove from the reconstruction target only those connected subregions that pass the frozen SAFE_REMOVE rule or the stricter SAFE_PROTECT rule. SAFE_REMOVE receives the frozen removal/background treatment. SAFE_PROTECT receives exact observed bytes. Every other coordinate remains exactly M3.

- No whole-component label is permitted merely because a parent M3 component contains a selected subregion.
- No selected REMOVE/PROTECT region may be grown using clean truth or outcome-adaptive morphology.
- Connected grouping constants, minimum run size and any region-coherence vetoes must be frozen before definitive OOF results.
- The M5 reason map should retain separate SAFE_REMOVE, SAFE_PROTECT, FALLBACK_M3 and invalid/OOD reason codes.

# 7. Revised Notebook 51 crossover

Notebook 51 remains the causal reconstruction test, but its interpretation must change. The principal comparison is frozen M3 versus frozen M5 under unchanged restorers. M4 remains a diagnostic reference and may be rendered/scored as a ceiling, but it should not be described as the deployable oracle mask that M5 is expected to reproduce.

| **Question**                      | **Revised Notebook 51 estimand**                                                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Does mask refinement help?        | For each frozen restorer, paired M5-minus-M3 change in false inserted ink, missing genuine ink, ink/edge F1 and secondary MAE/SSIM.                  |
| Is REMOVE doing useful work?      | Compare residual target area/complexity and reconstruction errors attributable to regions safely removed before reconstruction.                      |
| Is PROTECT doing useful work?     | Measure exact observed-print recovery and whether false annotation preservation remains within the frozen safety rule.                               |
| Does restorer choice matter less? | Compare between-restorer dispersion under M3 and M5; contraction supports the mask-conditioning hypothesis.                                          |
| What does M4 contribute?          | Diagnostic ceiling/reference for unsupported clean print and reconstruction opportunity; report separately rather than assuming mask-set continuity. |

## 7.1 Frozen restorer set

Use the smallest frozen set that spans the already observed restoration regimes, preferably biharmonic, Telea and the frozen learned U-Net. Do not add new blends, binary ensembles or restorer retuning inside Notebook 51. Those would confound the causal mask comparison.

## 7.2 Decision outcomes

| **Observed result**                                               | **Programme interpretation**                                                                                              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| M5 improves multiple frozen restorers and reduces restorer spread | Mask conditioning is the principal active lever; prioritise mask refinement and operational simplicity.                   |
| Only REMOVE helps                                                 | Commission/retain a REMOVE-only M5 and keep PROTECT on M3 until new evidence supports it.                                 |
| Only PROTECT helps                                                | Retain REMOVE as research only; commission a narrow exact-preservation challenger if safety remains acceptable.           |
| Mask-only metrics improve but reconstruction does not             | The selected mask endpoints were not causally useful; revisit the pre-registration rather than tuning restorers post hoc. |
| Benefit is source-concentrated or catastrophic                    | Reject general M5; report scoped evidence rather than a pooled success.                                                   |
| M5 fails integrity/immutability                                   | Invalidate the run before performance interpretation.                                                                     |

# 8. What remains unchanged

- V7.8 remains the frozen incumbent/fallback. V8.7 remains development-only.
- The reviewed/corrected authoritative mask remains the intervention boundary.
- Clean truth is post-lock development/scoring evidence only and never an inference channel.
- Pixels outside the authorised mask remain byte-for-byte immutable.
- Source-group OOF and page/source stratification remain required.
- Uncertain or invalid challenger cases fail closed to the incumbent treatment.
- No deterministic threshold grid should replace the learned/local hypothesis; earlier V7.9.x work already closed further handcrafted positive-selector development.
- Development evidence does not become production evidence; any later promotion still requires a frozen bundle and fresh untouched validation.
- Final upload packages remain curated, hashed review records rather than copies of every working cache.

# 9. Superseded statements and replacement wording

| **Superseded / qualified statement**                                                 | **Replacement**                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Notebook 49 must establish whether the M3-to-M4 oracle gap contains usable signal." | Notebook 49/49R4 establish whether M3 contains coherent exclusion targets identifiable from observables and separately diagnose M4 semantics.                                                        |
| "M3 is a mixture of PROTECT, REMOVE, RECONSTRUCT and UNCERTAIN."                     | The completed M3 decomposition supports PROTECT, REMOVE and UNCERTAIN; RECONSTRUCT was absent under the implemented truth-side definition and must not be required as a Notebook 50 reference class. |
| "M5 predicts PROTECT / REMOVE / RECONSTRUCT / UNCERTAIN."                            | M5 is constructed from two pre-registered assertions, SAFE_REMOVE and SAFE_PROTECT; no assertion means exact M3.                                                                                     |
| "M5 closes the M3-to-M4 mask gap."                                                   | M5 reduces demonstrably avoidable reconstruction burden and structural exposure inside M3. M4 is reported separately as diagnostic clean-print opportunity.                                          |
| "Notebook 51 compares M3/M5/M4 as commensurate mask arms."                           | Notebook 51 causally compares M3 and M5 under frozen restorers; M4 may remain a diagnostic ceiling/reference but is not presumed set-compatible.                                                     |
| "One M5 model/policy."                                                               | Two asymmetric target policies may be fitted/calibrated separately and combined only through a frozen, fail-closed construction rule.                                                                |

# 10. Notebook sequence from this point

| **Stage**                           | **Purpose**                                                                                                                       | **Go / no-go**                                                                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 49 / 49R4 - complete                | Lock M3 evidence; diagnose truth-side composition, connected coherence, observable signal and M4 semantics.                       | Completed evidence supports a revised Notebook 50 pre-registration; does not itself train M5.                             |
| 50 pre-registration - next          | Freeze REMOVE/PROTECT targets, candidate construction, grouping, inputs, folds, models, calibration, gates and exact M3 fallback. | No executable fitting until the amended pre-registration is written.                                                      |
| 50 executable                       | Generate source-group OOF REMOVE and PROTECT decisions; build frozen-candidate M5 only if gates pass.                             | May return REMOVE-only, PROTECT-only, combined M5, scoped research, or rejection.                                         |
| 51 crossover                        | Cross M3 versus frozen M5 with unchanged frozen restorers; retain M4 as diagnostic reference.                                     | Determines whether revised mask conditioning causally improves reconstruction and whether restorer sensitivity contracts. |
| Later freeze / independent evidence | Only after a successful development crossover, create immutable bundle and fresh promotion evidence.                              | No automatic production authority.                                                                                        |

# 11. Implementation and audit requirements for the amended programme

- Resolve exact completed runs by durable identity; never silently select a semantically incompatible "latest" run.
- Checkpoint expensive per-page work and validate resumable state with content identities, so audit/inventory construction is not repeated unnecessarily.
- Use explicit schemas for empty result tables so a legitimate empty class or contrast becomes a diagnostic result rather than a late KeyError.
- Keep observable route fractions and truth-side class fractions under distinct names; do not reuse "remove_fraction" for two different semantics.
- Write source inventories/state sidecars in a safely adoptable format so prior completed inventories can be validated and reused without full rehashing.
- Write all integrity failures before raising an error and preserve failed challengers as project evidence.

# 12. Relationship to established project evidence

| **Reference**                       | **Carried-forward principle**                                                                                                                                                     |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V7.5 design record                  | Preserve observed print exactly, remove confident non-structural material, reconstruct ambiguity only; keep intervention inside the controlled mask.                              |
| V7.9.x closure                      | Whole components are too coarse; deterministic positive selector tuning failed grouped safety/generalisation; use learned local evidence rather than more handcrafted thresholds. |
| Observed-print commissioning / V8.0 | Use observable-only evidence, source/page grouping, fail-closed UNCERTAIN and local spatial topology; coverage never substitutes for safety.                                      |
| V8.2 development decision           | Separate development evidence from promotion-grade freezing and independent testing; persist learned artefacts and whole-page evidence.                                           |
| V8.4 specification                  | Connected candidate groups and source-group OOF provide precedent for coherent region-level decisions built from local predictions.                                               |
| V8.6 programme                      | Mask readiness includes genuine print exposure, annotation handling, boundary conditioning and oracle opportunity; restorer crossover should hold restoration methods fixed.      |
| Notebook 49R3 / 49R4                | Actual durable mask lineage, post-lock truth diagnostics, connected subregion analysis and explicit M4 semantic audit provide the immediate evidence for this change record.      |

# 13. Decision record

**Formal V8.7 amendment** Close the original four-class M3→M4 interpretation as an unsupported working hypothesis. Retain the completed Notebook 49/49R4 artefacts as the evidence that caused the change. Commission a revised Notebook 50 pre-registration with REMOVE as the primary connected-subregion exclusion challenger, PROTECT as a stricter secondary exact-preservation challenger, and M3 as the exact fallback. Notebook 51 should causally test M3 versus the resulting frozen M5 across unchanged restorers, with M4 retained only as a separately interpreted diagnostic reference.

## 13.1 What is not authorised

- No hand-set REMOVE rule based directly on the attractive 49R4 p_print or distance AUC results.
- No global threshold relaxation to increase coverage.
- No whole-M3-component routing merely because a component is majority REMOVE or PROTECT.
- No clean-truth-guided subregion growing at inference.
- No restorer retuning in Notebook 51.
- No claim that M4 is the ground-truth deployable residual mask.
- No production or automatic-review reduction from this development programme.

# 14. Confidence and unresolved choices

**Confidence: 97%** The direction change is strongly supported by the completed Notebook 49/49R4 evidence and is consistent with the project's earlier local-topology, connected-group, fail-closed and source-group-validation architecture. Confidence is below 100% because 49R4 reports descriptive separability rather than source-group-held-out trained-model performance; Notebook 50 must establish that generalisation before M5 can be frozen.

## Choices to freeze in the Notebook 50 pre-registration

- Exact REMOVE and PROTECT truth targets and excluded/uncertain label zones.
- Candidate generation and connected-subregion grouping rule at inference.
- Primary local context representation and bounded model family.
- Calibration method and distinct REMOVE / PROTECT operating thresholds.
- Catastrophic false-REMOVE and false-PROTECT definitions.
- Minimum evidence/coverage and source-stability gates.
- How the frozen removal/background treatment is applied at SAFE_REMOVE subregions.
- Whether REMOVE and PROTECT policies may both fire near the same parent M3 component and the deterministic precedence rule if they do.

**Next question** What exact Notebook 50 pre-registration should now be frozen, including asymmetric REMOVE/PROTECT targets, source-group OOF folds, connected-subregion construction, calibration and pass/fail gates?

# 15. Source basis

This change record should be read alongside, rather than as a deletion of, the earlier project records. It supersedes the research direction in the original V8.7 49/50/51 specification where the new evidence conflicts with that pre-experiment hypothesis.

| **Project reference**                                                         | **Role**                                                                                                     |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| V8_7_Notebooks_49_50_51_Residual_Mask_Improvement_Programme 1.docx            | Original V8.7 hypothesis and 49/50/51 structure.                                                             |
| 49R3_V8_7_Residual_Mask_Oracle_Gap_Decomposition_SAFE_RECOVERY.ipynb          | Completed/rebuilt M3 truth-side decomposition and evidence lock.                                             |
| 49R4_V8_7_Connected_Subregion_Coherence_Separability_and_M4_Semantics.ipynb   | Connected-subregion coherence/separability and M4 semantics that drive this amendment.                       |
| V8_6_Residual_Reconstruction_Readiness_Programme.docx                         | M0-M4 roles, mask-readiness endpoints and frozen-restorer crossover principle.                               |
| V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx           | Preserve/remove/ambiguous architecture and mask/immutability guardrails.                                     |
| V7_9_x_Research_Findings_and_Next_Steps.docx                                  | Evidence against whole-component / deterministic positive selectors and support for learned local decisions. |
| Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx | Fail-closed learned challenger, grouped evaluation and safety-first governance.                              |
| V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx           | Local spatial topology and controlled-comparison principle.                                                  |
| V8_4_Structurally_Valuable_PRINT_Selection_Specification.docx                 | Connected candidate-group decision precedent and source-group OOF design.                                    |
| V8_2_Development_Whole_Page_Experiment_Decision_Record.docx                   | Development first, freeze second, independent test third; durable self-contained outputs.                    |
| Production_Retraining_Policy_YOLO_and_Structural_Safety.docx                  | Frozen incumbents/challengers, separate evidence roles and non-automatic promotion.                          |