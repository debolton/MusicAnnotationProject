**MUSIC ANNOTATION PROJECT**

Experiment V8.8 Workplan

**Operationalising M5_REMOVE and progressively replacing Notebook 09**

_Execution and promotion plan | 2 November 2026_

**V8.8 will operationalise the frozen SAFE_REMOVE policy as M5_REMOVE. It may replace Notebook 09 as the correction engine after exact replay and operational regression pass. Notebook 09 cannot be retired as the mask-authority and exception-review path until separate validation supports that reduction in human review.**

Status: commissioned implementation and validation workplan. No threshold movement, retraining or SAFE_PROTECT re-entry is authorised.

# 1. Commissioning decision

V8.8 is an engineering qualification programme, not a new mask-learning experiment. V8.7 has already selected SAFE_REMOVE and demonstrated downstream reconstruction benefit for M5_REMOVE. The V8.8 task is to freeze that policy, reproduce it exactly in a reusable page pipeline, run it beside the incumbent workflow, and make a documented replacement decision.

- SAFE_REMOVE is the only V8.8 intervention arm.
- SAFE_PROTECT remains parked and must not be combined, threshold-relaxed or reintroduced.
- Every unsupported, invalid, uncertain or failed case retains exact M3 treatment.
- M4 remains diagnostic only and is not a runtime input or deployable mask arm.
- No development result alone authorises automatic annotated-only output without a separately validated mask-authority path.

_Evidence basis: Final V8.7 decision; the V8.7 asymmetric mask programme; the staged path from Notebook 09 to annotated-only operation._

# 2. V8.8 objective and end state

| **Objective**                 | **Definition of done**                                                                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Technical operationalisation  | A versioned implementation accepts an observed page plus authoritative mask, builds M3, applies the frozen SAFE_REMOVE policy, constructs M5_REMOVE, performs frozen reconstruction and writes complete audit artefacts. |
| Exact regression              | On the frozen V8.7 roster, the operational implementation reproduces the registered SAFE_REMOVE decisions and M5_REMOVE masks exactly, with zero unauthorised page changes.                                              |
| Correction-engine replacement | V8.8 becomes the normal correction engine where an authoritative mask is supplied; Notebook 09 remains comparator and rollback until a documented release decision.                                                      |
| Review-path reduction         | Human review becomes exception-focused only after shadow and supervised-pilot evidence demonstrates acceptable coverage, stability and review burden.                                                                    |
| Full Notebook 09 retirement   | Deferred until annotation-mask authority and the complete annotated-only chain have separate independent evidence and an operational rollback path.                                                                      |

# 3. Scope boundaries

| **In scope**                                                                                           | **Out of scope**                                                               |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Freeze and replay the fitted REMOVE model, calibration, ordered schema, constants and decision policy. | Retraining or recalibrating SAFE_REMOVE.                                       |
| Build an operational M5_REMOVE page wrapper with deterministic artefacts and fail-closed fallback.     | Testing SAFE_PROTECT or a combined REMOVE + PROTECT M5.                        |
| Exact V8.7 equivalence, immutability, shape/canvas and determinism testing.                            | Changing M3 semantics, thresholds, folds, labels or reconstruction algorithms. |
| Shadow comparison against Notebook 09 and supervised exception review.                                 | Claiming that V8.8 validates raw detector masks automatically.                 |
| A documented go/amber/no-go decision on replacing Notebook 09 as correction engine.                    | Automatic production promotion from a single development/shadow result.        |

# 4. Notebook and workstream plan

| **Stage** | **Notebook**                   | **Purpose**                                                                                                              | **Exit gate**                                                                                                          |
| --------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| A         | 52 - V8.8A Freeze & Replay     | Create the immutable M5_REMOVE policy bundle and exact replay pack from the successful Notebook 50 run.                  | All required artefacts hash-verified; exact replay succeeds; no missing model/calibrator/schema object.                |
| B         | 53 - V8.8B Operational Builder | Create the reusable page pipeline from authoritative mask through M3, SAFE_REMOVE, M5_REMOVE and corrected page.         | Fail-closed page tests pass; outputs are deterministic; no unauthorised pixel changes.                                 |
| C         | 54 - V8.8C Equivalence Audit   | Compare operational SAFE_REMOVE/M5_REMOVE/page outputs with the V8.7 golden masters.                                     | Zero SAFE_REMOVE mismatch, zero M5_REMOVE XOR, zero outside-authority drift, exact repeat run.                         |
| D         | 55 - V8.8D Shadow Comparison   | Run V8.8 and Notebook 09 on the same operational pages without allowing V8.8 to determine approved output automatically. | Disagreement inventory complete; source/page concentrations and exception burden understood.                           |
| E         | 56 - V8.8E Supervised Pilot    | Present V8.8 output and exceptions for human accept, unchanged, defer and mask-edit/rescore decisions.                   | Pilot gates pass and correction-engine release decision is signed.                                                     |
| F         | 57 - V8.8F Release & Control   | Create release identity, monitoring, rollback, launcher update and decision record.                                      | V8.8 is primary correction engine or retained as shadow/exception challenger; Notebook 09 role is explicitly recorded. |

# 5. Workstream A: freeze the SAFE_REMOVE policy

Notebook 52 must load the completed Notebook 50 durable run. It must not infer artefact identities from filenames alone or reconstruct missing learned objects from saved probabilities.

- Freeze the fitted REMOVE model/checkpoint, fitted calibrator, ordered predictor/channel schema, threshold, decision-margin and connected-subregion/coherence rules.
- Record source run, training roster/fold identities, software environment, implementation hashes and the Notebook 50 REMOVE-only verdict.
- Create a replay population covering multiple pages/source groups and expected raw scores, calibrated probabilities, SAFE_REMOVE decisions and M5_REMOVE arrays.
- Hash every bundle object with SHA-256 and assign one policy ID and compatibility ID.
- Fail if SAFE_PROTECT objects are accidentally included in the runtime policy.

# 6. Workstream B: operational page builder

Notebook 53 should accept an observed page and an authoritative mask. It should produce M5_REMOVE and a corrected page without clean truth at inference. Clean pages remain evaluation evidence only.

**Observed page + authoritative mask -> frozen M3 preparation -> frozen SAFE_REMOVE inference -> M5_REMOVE = M3 minus SAFE_REMOVE -> frozen reconstruction -> corrected page + audit pack**

- Runtime truth firewall: no clean image, truth label, post-hoc score or oracle mask may enter inference.
- SAFE_REMOVE may subtract only from M3. It must never expand the intervention boundary.
- Any hash, schema, shape, finite-value, compatibility or replay failure falls back to exact M3 or blocks output under the frozen contract.
- Corrected output must retain source/incumbent bytes outside the authorised mask and wherever the policy has not validly changed treatment.
- Write observed image, authoritative mask, M3, probability/decision artefacts, SAFE_REMOVE, M5_REMOVE, corrected page, difference map, reason map and page manifest separately.

# 7. Workstream C: exact equivalence and regression

| **Test family**       | **Required result**                                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Artefact identity     | All policy objects resolve uniquely and match registered hashes.                                                                       |
| SAFE_REMOVE decisions | Exact equality with the V8.7 golden-master decisions on the complete regression roster.                                                |
| M5_REMOVE mask        | Zero XOR pixels against the V8.7 golden-master M5_REMOVE arrays.                                                                       |
| Page shape/canvas     | Authoritative full-page shape retained; Notebook 50 truncated-canvas/padding defect is a permanent regression test.                    |
| Immutability          | Zero changed pixels outside the authoritative intervention boundary.                                                                   |
| Fallback identity     | Zero deviation from the incumbent at every location without a valid SAFE_REMOVE assertion.                                             |
| Determinism           | A repeated unchanged run yields identical decisions, masks, output bytes and semantic hashes.                                          |
| Failure behaviour     | Corrupt, missing, duplicate, incompatible or non-finite inputs write audit evidence and stop/fall back rather than continue partially. |

_Evidence basis: Exact stage-by-stage equivalence and fail-closed testing are established project patterns in the V7.8.2 audit and Notebook 09 mandatory regression gate._

# 8. Workstream D: Notebook 09 shadow comparison

Notebook 55 runs both paths on the same pages. Notebook 09 remains decision authority during shadow mode. V8.8 outputs are recorded but cannot silently replace approved output.

- Record page/component/region agreements and disagreements between Notebook 09 and V8.8.
- Stratify disagreements by source group, page, mask status, SAFE_REMOVE coverage, reconstruction method and reason code.
- Create review queues for largest output differences, false-ink-sensitive regions, missing-ink-sensitive regions, boundary cases and any integrity fallback.
- Measure how many pages can proceed without component-by-component review and how many need leave-unchanged, defer or mask-edit/rescore handling.
- Do not retune the SAFE_REMOVE policy from shadow outcomes. A material policy change creates a new experiment version.

# 9. Workstream E: supervised pilot and release decision

| **Pilot route**         | **Decision treatment**                                                                                                                    |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Automatic candidate     | Only pages/regions satisfying every frozen runtime gate are eligible to be presented as V8.8 automatic candidates.                        |
| Exception review        | Uncertain, failed, unsupported or mask-authority exceptions remain review/no-change cases.                                                |
| Human actions           | Retain accept, leave unchanged, defer and mask-edit/rescore decisions with audit history.                                                 |
| Promotion               | Promotion is never automatic. A written release decision must state scope, evidence, exceptions, monitoring and rollback.                 |
| Notebook 09 disposition | Primary correction engine may move to V8.8 while Notebook 09 remains a versioned shadow comparator, exception reviewer and rollback path. |
| Full retirement         | Only after mask-authority and end-to-end annotated-only validation support it; Notebook 09 remains retained evidence rather than deleted. |

# 10. Required inputs before execution

| **Input**                         | **Minimum requirement**                                                                                                                                          | **If unavailable**                   |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Completed Notebook 50 durable run | Actual fitted REMOVE model/checkpoint, calibrator, schema, decision constants, model/calibrator inventories, OOF prediction/decision table and M5 map inventory. | Stop NB52.                           |
| Completed Notebook 51 durable run | Golden-master M3/M5_REMOVE arrays, reconstruction outputs, verdict, metrics and completion/integrity records.                                                    | Equivalence limited; do not promote. |
| V7.8/M3 canonical implementation  | Versioned page/mask preparation, reconstruction implementation, semantic hashes and regression arrays used by Notebook 09.                                       | Stop NB53.                           |
| Notebook 09 incumbent             | Executable notebook/source, policy bundle, regression references and representative operational inputs.                                                          | No shadow comparison.                |
| Authoritative-mask pages          | Representative reviewed-mask pages with source-group metadata; separate any genuinely untouched validation pages.                                                | Pilot evidence insufficient.         |
| Release controls                  | Named output roots, immutable run directories, rollback location and launcher/config update path.                                                                | No release switch.                   |

# 11. Required output structure

Each V8.8 notebook must finish with a curated upload package while retaining heavyweight arrays and caches in the durable run directory.

```
v8_8/<run_id>/
  policy_bundle/
  inputs/
  tables/
  masks/
  corrected_pages/
  difference_maps/
  reason_maps/
  regression/
  shadow/
  review/
  audit/
  upload/
  experiment_identity.json
  software_environment.json
  verdict.json
  executive_summary.txt
```

# 12. Decision gates

| **Decision** | **Criteria**                                                                                                                                                                                                       | **Action**                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| GREEN        | Freeze/replay, operational builder and exact regression all pass; shadow/pilot reveal no unmodelled systematic failure; exception volume is operationally acceptable; written release and rollback controls exist. | V8.8 primary correction engine.     |
| AMBER        | Engineering controls pass, but shadow/pilot coverage, stability or exception burden is insufficient for primary use.                                                                                               | Keep V8.8 in shadow/exception mode. |
| RED          | Any artefact, replay, equivalence, immutability, deterministic-output or fail-closed requirement fails.                                                                                                            | Retain Notebook 09 unchanged.       |

No gate may be converted from RED or AMBER to GREEN by lowering SAFE_REMOVE thresholds, removing source groups, suppressing failures or changing reconstruction semantics after results are known.

# 13. Delivery ownership and execution order

| **Order** | **Deliverable**                                                            | **Review decision**                                                             |
| --------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1         | Notebook 52 plus frozen bundle inventory and replay verdict.               | Can unseen-page inference be executed from complete durable objects?            |
| 2         | Notebook 53 operational builder plus sample outputs and fail-closed tests. | Does the wrapper implement the intended contract?                               |
| 3         | Notebook 54 exact equivalence report.                                      | Is V8.8 computationally identical to the successful V8.7 policy where required? |
| 4         | Notebook 55 shadow report and disagreement review pack.                    | Is operational behaviour stable and understandable?                             |
| 5         | Notebook 56 pilot outcomes and burden/safety record.                       | Can V8.8 become the primary correction engine?                                  |
| 6         | Notebook 57 release manifest, launcher/config change and rollback record.  | What is Notebook 09 now: primary, fallback or exception-only?                   |

# 14. Risks and mitigations

| **Risk**                                      | **Mitigation**                                                                                                                       |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Missing fitted model/calibrator               | Stop and recover the durable Notebook 50 run. Do not rebuild from saved OOF probabilities and call it the same policy.               |
| Model runs only on historical M3 features     | Package the observable feature builder and verify unseen-page feature equivalence before shadow use.                                 |
| Canvas/shape defects recur                    | Full-page authoritative shape is mandatory; maintain an exact regression case for every padded Notebook 50 map.                      |
| Mask authority mistaken for mask conditioning | Keep reviewed mask status explicit. V8.8 validates M5_REMOVE inside the mask supplied; it does not independently validate that mask. |
| Good aggregate, concentrated failure          | Report page/source-group disagreement and exception rates; do not generalise across unstable sources.                                |
| Pressure to replace on schedule               | Use GREEN/AMBER/RED gates. Replacement is an evidence decision, not a calendar milestone.                                            |
| Audit artefacts collapse meanings             | Persist detector/confidence, authoritative mask, M3, SAFE_REMOVE, M5_REMOVE, routing/reconstruction and final page separately.       |

# 15. Immediate starting action

The first executable deliverable is Notebook 52, not another model experiment. Execution begins by resolving the completed Notebook 50 durable run and proving that the actual fitted REMOVE model, calibrator, schema, constants and reusable inference implementation exist and are mutually compatible.

- If the durable learned objects are complete, build the bundle and replay gate.
- If any learned object is missing, record the gap and recover it from the successful run. Do not retrain under the V8.8 label.
- Once Notebook 52 passes, proceed directly to the operational builder and exact regression audit.

# 16. Confidence and next decision

Confidence: 96%. The programme sequence, fail-closed boundaries, exact equivalence requirement and staged Notebook 09 demotion are strongly supported by the project record. The remaining uncertainty is whether the successful Notebook 50 durable run contains every fitted object and preprocessing implementation needed for unseen-page inference. That uncertainty is deliberately made the first Notebook 52 gate.

**Next question:** Does the completed Notebook 50 durable run contain a complete deployable REMOVE bundle, including the fitted model, calibrator, ordered schema and reusable observable-feature builder, or only development OOF outputs?

# Source register

- V8_7_Final_Decision_and_V8_8_Direction.docx
- V8_7_Residual_Mask_Programme_Change_Record_After_Notebook_49R4.docx
- 50_V8_7_Asymmetric_Residual_Mask_M5_Challenger.ipynb
- Notebook 51 completed output package and formal verdict
- Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx
- 09_Pilot_Proposed_Correction_Routing_Reviewer.txt
- 18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt
- V7_7F_Design_Record_Applying_the_Frozen_Structural_Policy_to_V7_8.docx
- Production_Retraining_Policy_YOLO_and_Structural_Safety.docx
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx
- V8_2_Development_Whole_Page_Experiment_Decision_Record.docx