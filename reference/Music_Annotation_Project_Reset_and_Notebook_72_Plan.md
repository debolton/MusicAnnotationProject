Project Reset: From Paired Development to Annotated-Only Operation

Key objectives, actions and implementation steps for Notebook 72 and the next phase of the Music Annotation Project

**Decision record and working plan | 16 November 2026**

RESET DECISION  
The immediate objective is not another clean-assisted benchmark. It is to prove that the frozen downstream correction stack can process annotated-only pages using only runtime-available evidence and a reviewer-authoritative mask. Clean pages, where they happen to exist, are revealed only after outputs are written and locked, and are used solely for quantitative validation.

# 1. Why the project needs a reset

The project drifted from its production question into repeated forensic comparisons between historical implementations and clean-truth development artefacts. Those investigations produced useful engineering evidence, but they do not answer whether the system can clean a page when no clean page is available at runtime.

- If producing the corrected page requires the clean page, the result is a reconstruction benchmark rather than a practical handwriting-removal tool.
- V8.x was the development laboratory: paired clean and annotated pages supplied truth for labels, scoring and diagnosis.
- Experiment 9 should be the runtime transition: remove clean truth from the processing path and test the frozen downstream method on annotated-only inputs.
- Notebook 71 and 71R1 should be retained as forensic evidence, particularly for mask lineage and reproducibility, but they should not define the next experiment. Do not build 71R2.

# 2. The governing distinction

| **Stage**                     | **What the system may use**                                                                                                                      | **Purpose**                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| V8.x development / validation | Annotated or aligned page; reviewer-authoritative mask; observable image evidence; frozen models and policies. Clean truth only after inference. | Learn, compare and quantify structural safety and cosmetic benefit.                                      |
| V9 / Notebook 72 runtime test | Annotated page; reviewer-authoritative mask; observable image evidence; frozen V8.8 operator and policy. No clean page or clean-derived field.   | Demonstrate that the developed correction method can operate under production-like evidence constraints. |
| Post-lock evaluation          | Clean page only where a counterpart exists; blinded human review for every page.                                                                 | Quantitative validation for the paired cohort and performance-usability assessment for the full cohort.  |

## Truth firewall

- Clean image pixels, clean ink maps, clean edge maps, truth classes, reconstruction errors and post-hoc outcome fields are forbidden during inference.
- The clean directory must not be read or resolved until predictions, decisions and corrected outputs have been written and hash-locked.
- The authoritative reviewed mask is the sole boundary of permissible intervention.
- Pixels outside the authorised mask must remain byte-for-byte unchanged.
- Where V8.8 does not make a valid PRINT preservation decision, the frozen incumbent behaviour must be retained exactly.

# 3. Restated Experiment 9 objective

Primary question: Can the frozen downstream stack that earned its evidence on paired clean/annotated development data now produce usable corrected pages from annotated-only runtime evidence and a reviewer-corrected mask?

This is the crucial transition the project must demonstrate:

| **Paired development**                  | **Freeze**                   | **Annotated-only inference**        | **Output lock**                           | **Evaluation branches**                                                  |
| --------------------------------------- | ---------------------------- | ----------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| Clean truth supports labels and scoring | Freeze V8.8 downstream stack | Annotated page + reviewed mask only | Hash corrected pages, decisions and audit | Clean exists: quantitative scoring<br>No clean: blinded usability review |

# 4. Notebook 72: Frozen V8.8 Annotated-Only Supervised Operational Trial

Notebook 72 should be a single end-to-end operational trial across both available page populations. It should process every eligible reviewed page in source/annotated_only and source/annotated, while treating clean counterparts as sealed evaluation material rather than runtime inputs.

| **Contract area**        | **Notebook 72 requirement**                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inputs                   | All eligible pages from source/annotated_only and source/annotated; reviewer-corrected authoritative masks; frozen V8.8 implementation and artefacts. |
| Eligibility              | Exclude pages without a reviewer-corrected authoritative mask. Never fall back silently to unreviewed YOLO predictions.                               |
| Forbidden runtime inputs | source/clean, clean registration, clean-derived maps, truth labels, post-hoc reconstruction metrics and any outcome field.                            |
| Runtime outputs          | Corrected page, decision map, probabilities and uncertainty, authoritative mask, changed-pixel map, outside-mask audit, model/policy/replay identity. |
| Lock                     | Write semantic hashes for inputs, decisions, probabilities, corrected pages and configuration before clean truth is made available.                   |
| Evaluation               | Quantitative truth scoring on paired pages after lock; blinded whole-page performance-usability review on the entire population.                      |
| Packaging                | Create a standard upload/review bundle with summary, tables, review images, audit records, manifests and hashes.                                      |

# 5. Cohort design

| **Cohort**                                  | **Runtime processing**                                                         | **Objective scoring**                                                               | **Human review**                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| A. Genuine annotated-only                   | Run frozen V8.8 using the annotated page and reviewer-authoritative mask.      | Not available because no clean counterpart exists.                                  | Required. Blind original-versus-corrected performance-usability review. |
| B. Paired pages processed as annotated-only | Run the exact same runtime path, with the clean page hidden until output lock. | After lock, reveal the clean counterpart and calculate structural/cosmetic metrics. | Required using the same blinded whole-page review as Cohort A.          |

The cohorts are combined for processing and review, but not for truth-dependent scoring. This prevents the earlier mistake of excluding annotated-only pages merely because they cannot contribute to clean-grounded metrics.

# 6. Required execution sequence

**1\. Freeze and identify the operator.** Resolve the exact durable V8.8 operator adapter and its callable interface. The notebook must not reconstruct or guess V8.8 from historical notebooks. Record code, model, calibrator, policy and settings hashes.

**2\. Build the full runtime roster.** Enumerate reviewed pages from both source populations. Resolve each authoritative mask and observed/aligned page through auditable lineage. Write exclusions and reasons.

**3\. Close the truth path.** Technically block access to source/clean and all truth-derived artefacts. Add assertions proving no clean reads occurred before the lock.

**4\. Run frozen inference.** Generate baseline/fallback arrays, probabilities, PRINT/ANNOTATION/UNCERTAIN decisions, corrected pages and component/page audit records.

**5\. Assert integrity.** Require zero outside-mask drift, exact replay, finite arrays, shape agreement and exact frozen behaviour outside valid V8.8 interventions.

**6\. Create the prediction/output lock.** Hash the roster, inputs, authoritative masks, operator identity, probabilities, decisions, corrected pages and software environment.

**7\. Reveal evaluation truth for paired pages.** Only after the lock, resolve clean counterparts and compute registered structural and cosmetic outcomes.

**8\. Run blinded whole-page review for every page.** Present annotated original versus corrected output at identical scale without clean truth in the initial judgement.

**9\. Produce a combined decision package.** Summarise quantitative outcomes, human usability outcomes, failures, exceptions, source-group stability and next-step decision.

# 7. Evaluation framework

## 7.1 Integrity and replay: all pages

- Clean truth used during runtime: exactly zero.
- Outside authoritative mask drift: exactly zero pixels.
- Model, calibrator, policy, schema and operator hashes match the frozen record.
- Repeated inference on the replay sample produces identical decisions and corrected-page bytes.
- Every excluded page has an explicit reason; no silent cohort loss.

## 7.2 Quantitative evaluation: paired pages only

- Printed-ink preservation and structural-edge preservation.
- Annotation removal and residual-annotation burden.
- Catastrophic print-damage cases and their spatial concentration.
- Page and source-group stability, not only pooled pixel totals.
- Separate structural safety from cosmetic quality. A structurally safe page may still contain distracting residue.

## 7.3 Performance-usability review: all pages

The review question is not "does this pixel agree with clean truth?" It is "is this corrected page clean and trustworthy enough to perform from?"

| **Review outcome**                   | **Meaning**                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| PERFORMANCE_READY                    | Page is usable for performance without material concern.                       |
| PERFORMANCE_READY_WITH_MINOR_RESIDUE | Minor annotation remains but does not materially distract or obscure notation. |
| REVIEW_REQUIRED                      | Potential notation damage, residue or ambiguity requires targeted review.      |
| UNUSABLE_ANNOTATION_REMAINS          | Residual annotation makes the page unsuitable.                                 |
| UNUSABLE_PRINT_DAMAGE                | Correction has removed or damaged printed notation materially.                 |

- Was visible annotation removed completely, partly or not at all?
- Was any printed notation visibly damaged or made uncertain?
- Is the page better, the same or worse than the current incumbent output?
- Would the reviewer perform from this page?
- What component or region caused any rejection or referral?

# 8. What success should mean

| **Dimension**        | **Minimum success condition**                                                                                     | **Failure interpretation**                                              |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Runtime validity     | Every processed page is produced without reading clean truth.                                                     | The project has not demonstrated annotated-only operation.              |
| Integrity            | Zero outside-mask changes and exact replay.                                                                       | Invalidate the run and fix implementation before interpreting quality.  |
| Structural safety    | No catastrophic notation deletion on truth-scored pages; no obvious print damage in blinded review.               | Frozen stack is unsafe at the tested operating point.                   |
| Cosmetic usefulness  | Residual annotation is low enough not to distract from performance on a useful share of pages.                    | Method may be safe but not practically useful.                          |
| Whole-page usability | A meaningful proportion of the full population is PERFORMANCE_READY or READY_WITH_MINOR_RESIDUE.                  | The system remains a research benchmark rather than an operational aid. |
| Auditability         | Every decision and changed pixel can be traced to mask authority, frozen operator identity and recorded evidence. | Result cannot support a controlled pilot or future release decision.    |

Do not define success as perfect agreement with the clean page. The clean page is validation truth, not the product. The product is a corrected annotated page that preserves musical information and is usable in practice.

# 9. Immediate actions and ownership sequence

- Close Notebook 71 and 71R1 as forensic detours. Preserve their evidence; stop further 71R development.
- Confirm the exact V8.8 downstream implementation and create or locate v8_8_operator.py. Record its required inputs, outputs and durable hashes.
- Inventory authoritative reviewed masks across source/annotated_only and source/annotated. Produce explicit eligible and excluded rosters.
- Build Notebook 72 around the truth firewall and two-cohort evaluation design in this document.
- Run a small smoke test only to prove path resolution, operator compatibility, lock creation and review packaging. Do not use smoke-test outcomes for model decisions.
- Run the full reviewed population, expected to be roughly 200 pages, without changing the frozen operator.
- Complete the blinded review before opening clean-truth diagnostic views to reviewers.
- Write a decision record: proceed to supervised pilot, retain for targeted correction, or stop and redesign the downstream method.

# 10. Explicit non-objectives

- Do not automate mask approval in Notebook 72. Human-reviewed masks remain authoritative for this experiment.
- Do not retrain, refit, recalibrate or adjust V8.8 from Notebook 72 outcomes.
- Do not solve whole-component semantic classification unless it is proven necessary for the correction contract.
- Do not use region-semantic coverage as the principal blocker when the downstream task is a local observable PRINT-protection decision inside an authorised mask.
- Do not weaken structural-safety thresholds merely to increase the number of pages changed.
- Do not recycle genuine annotated-only pages into invented quantitative truth. Their evidence is integrity plus blinded human usability.

# 11. Decision gates after Notebook 72

| **Observed result**                                                                         | **Decision**                                                                                                                   |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Runtime cannot execute without clean-derived artefacts                                      | Stop. The frozen downstream stack is not production-like and must be redesigned.                                               |
| Integrity failure or outside-mask drift                                                     | Invalidate the run. Fix implementation before quality interpretation.                                                          |
| Paired metrics acceptable, annotated-only usability poor                                    | The benchmark works but the operational product does not. Investigate domain shift, mask quality and reconstruction behaviour. |
| Annotated-only usability acceptable, paired metrics expose concentrated structural failures | Do not generalise. Retain supervised review and isolate the failure mechanism/source group.                                    |
| Both cohorts show acceptable safety and usability                                           | Proceed to a controlled supervised pilot and separately evaluate whether mask review can later be reduced.                     |
| Strong results but only on narrow sources                                                   | Scope the next trial to those sources or gather fresh representative evidence; do not claim general production readiness.      |

# 12. Project records underpinning this reset

**Pathway_From_Pilot_Notebook_09_to_Annotated_Only_Production_Correction.docx** — Defines annotated-page-only inference, staged transition, separate mask authority and no-clean runtime contract.

**10_Reconstruction_Benchmark.txt** — Establishes that clean truth is used only after reconstruction and that outside-mask pixels are immutable.

**V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx** — Separates annotation confidence from correction safety and treats clean-truth diagnostics as post-hoc validation.

**17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt** — Defines leakage controls, predictor locking and separation of observable inference evidence from clean-truth outcomes.

**12_V7_7_Asymmetric_Structural_Risk_Calibration.txt** — Separates structural safety from cosmetic quality and excludes truth variables from predictors.

**09_Pilot_Proposed_Correction_Routing_Reviewer.txt** — Provides the supervised operational pattern, authoritative-mask resolution, frozen processing and reviewer decisions.

**V7_9_x_Research_Findings_and_Next_Steps.docx** — Warns that component-level semantic decisions are often too coarse and retains V7.8 as the incumbent.

**Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx** — Defines observable-only local inference, PRINT/ANNOTATION/UNCERTAIN treatment and strict truth firewall.

**V8_Next_Stage_Full_Pixel_Whole_Page_OOF_Reference.docx** — Defines whole-page outputs, clean truth as scoring-only and whole-page visual review questions.

**V8_1_Full_Pixel_Whole_Page_OOF_Notebook_Specification.docx** — Defines prediction locking before truth scoring and durable output/replay requirements.

**V8_2_Development_Whole_Page_Experiment_Decision_Record.docx** — Separates development from freeze and independent-test governance, and requires durable experiment packaging.

**28_V8_2_Development_Whole_Page_Experiment.ipynb** — Shows why the V8.2 roster was restricted to pages with clean, aligned and cached development truth, explaining the missing annotated-only outputs.

BOTTOM LINE  
Process both populations through the same clean-free runtime path. Lock every output. Use clean truth afterward where it exists, and use blinded performance-usability review everywhere. The next deliverable is Notebook 72, not another historical equivalence detour.

Confidence: 96%. The architecture and truth firewall are explicit across the project records. The remaining material uncertainty is the exact durable location and callable interface of the frozen V8.8 operator, and the completeness of reviewer-authoritative masks across the two source populations.

**Next question:** Where is the exact frozen V8.8 operator and what minimum adapter contract must Notebook 72 enforce before the full-population run can start?