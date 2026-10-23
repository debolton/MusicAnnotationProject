**MUSIC ANNOTATION PROJECT**

V8.3 Residual-Gap Continuation Feasibility

_Final findings, closure decision and retained evidence_

**DECISION STATUS: Close V8.3 as a completed, informative failed challenger. Retain frozen V7.8 reconstruction and do not continue legacy deterministic continuation tuning on the current development population.**

Project reference document | 20 October 2026

# 1. Purpose and scope

This record closes the V8.3 research thread opened by Notebook 40. V8.3 tested whether additional observed-print preservation at the 0.50 threshold, relative to 0.75, reduced residual gaps enough for historical reconstruction and continuation methods to perform better. The experiment was development-only and had no authority to change the registered threshold or production policy.

**Core conclusion  
**The rehabilitation hypothesis was not supported. Every tested legacy reconstruction method produced a negative mean local MAE benefit for 0.50 over 0.75. The historical staff-continuation implementation directly covered only 10.87% of reviewed selected pixels. The thread should close without further threshold or rule engineering.

# 2. Experimental question and comparator

Primary question: after preserving the additional 0.50-but-not-0.75 observed-print pixels, does the smaller residual gap improve clean-truth reconstruction under the historical methods?

- Population: 184 reviewed primary regions containing 1,591 reviewed selected pixels across the adjacent 0.75-to-0.50 bands.
- Comparator: reconstruction with the 0.75 residual mask.
- Treatment: reconstruction with the 0.50 residual mask.
- Methods: Telea, Navier-Stokes, biharmonic, hybrid_conservative and hybrid_routed.
- Primary direction: positive local MAE benefit means 0.50 improved reconstruction over 0.75; negative means it worsened reconstruction.

# 3. Continuation applicability

| **Applicability class**               | **Regions** | **Selected pixels** | **Share** |
| ------------------------------------- | ----------- | ------------------- | --------- |
| Implemented old staff continuation    | 32          | 173                 | 10.87%    |
| Plausible, but not old implementation | 52          | 465                 | 29.23%    |
| Not a continuation target             | 90          | 856                 | 53.80%    |
| Non-print, mixed or indeterminate     | 10          | 97                  | 6.10%     |

The broad semantically plausible share, combining implemented staff continuation with plausible unimplemented stem/barline, beam, slur/tie and hairpin cases, was 40.10%. Its page-clustered 95% bootstrap interval was 21.15% to 57.75%. This is an opportunity estimate, not evidence that any completion algorithm works on those structures.

# 4. Method results

| **Method**          | **Mean MAE benefit** | **Median** | **Regions with positive benefit** |
| ------------------- | -------------------- | ---------- | --------------------------------- |
| hybrid_conservative | \-1.091              | \-0.398    | 41.3%                             |
| biharmonic          | \-1.288              | \-0.480    | 40.2%                             |
| hybrid_routed       | \-1.408              | \-1.010    | 31.0%                             |
| Telea               | \-1.410              | \-1.004    | 25.0%                             |
| Navier-Stokes       | \-1.576              | \-1.427    | 23.4%                             |

All five aggregate means are negative. The result is stronger than a failure of one algorithm: the ranking spans PDE inpainting, biharmonic completion and two hybrid methods. Even the best aggregate method, hybrid_conservative, remains on the wrong side of zero.

# 5. Subgroup interpretation

- Implemented old staff continuation was not rescued. Mean MAE benefit was negative for every method, including hybrid_conservative (-2.747) and hybrid_routed (-2.527).
- Plausible but unimplemented structures were less negative under biharmonic (-0.620) and hybrid_conservative (-0.533), but still did not provide a positive aggregate result.
- A minority of regions improved under individual methods, but region-level wins did not produce a positive page- or population-level case for continuing the branch.
- The failure does not negate the value of exact observed-pixel preservation. It rejects the narrower claim that legacy reconstruction becomes better merely because preservation shrinks the remaining gap.

# 6. Relationship to Notebook 39 and earlier V7.9.x research

Notebook 39 showed that exact preservation of genuine observed structure can improve outcomes and that the benefit varies by musical structure. V8.3 asked a different question: whether old reconstruction algorithms could exploit the smaller residual gaps. The negative V8.3 result is compatible with Notebook 39: preservation may be valuable even when legacy completion remains inadequate.

The outcome also reinforces the V7.9.x decision to stop hand-crafted deterministic continuation as a positive selector. Deterministic geometry remains useful for audit, context and conservative vetoes, but further tuning on the same development evidence would repeat a hypothesis already rejected under grouped evaluation.

# 7. Closure decision

**Formal decision  
**Close V8.3. Retain Notebook 40 and its upload package as a completed failed-challenger record. Do not create a V8.3.1 continuation-threshold sweep, safe_staff_continuation_v2, or another legacy inpainting comparison on the same cohort.

# 8. Consequences and controls

- Frozen V7.8 geometry_t80 remains the correction fallback and comparator.
- V8 observed-print research continues as an exact-preservation problem, not a deterministic completion problem.
- No V8.3 method or applicability class is authorised for production, threshold movement or automatic reconstruction.
- The 29.23% plausible-but-unimplemented population is archived as future hypothesis material only. It does not justify immediate model development.
- Any future reconstruction challenger must be materially different, pre-registered and compared with V7.8 under the same truth firewall, page/source grouping, immutability and visual-review controls.

# 9. Archival package and decision record

| **Artefact**                                     | **Role**                                                 |
| ------------------------------------------------ | -------------------------------------------------------- |
| executive_summary.txt                            | Headline population, applicability and interpretation.   |
| continuation_applicability_summary.csv           | Reviewed-region and selected-pixel applicability totals. |
| plausibly_continuable_pixel_share.json           | Point estimate and page-clustered interval.              |
| paired_050_vs_075_method_deltas.csv              | Region-level paired method deltas.                       |
| v8_3_method_summary_by_structure.csv             | Method results by semantic structure.                    |
| residual_gap_span_comparison.csv                 | Residual-span comparison across thresholds.              |
| method_mae_benefit_050_vs_075.png                | Headline method comparison.                              |
| continuation_applicability_share.png             | Headline applicability distribution.                     |
| experiment_manifest.json and upload_manifest.csv | Run identity, provenance and upload integrity.           |

# 10. Final statement

V8.3 is closed because its principal mechanism failed: reducing residual gaps from the 0.75 to the 0.50 preservation boundary did not improve mean local reconstruction error for any tested historical method. The project retains the negative evidence, the exact-preservation results from Notebook 39 and V7.8 as incumbent fallback. A future reconstruction research strand, if commissioned, must start from a new model class and a new pre-registration rather than continue deterministic continuation tuning.

Confidence: 98%. The closure decision is supported by complete paired results for 184 reviewed regions, uniformly negative method means, limited direct applicability of the historical implementation and the earlier V7.9.x generalisation failures.

# Source basis

- Notebook 40 upload: executive_summary.txt; continuation_applicability_summary.csv; plausibly_continuable_pixel_share.json; paired_050_vs_075_method_deltas.csv; v8_3_method_summary_by_structure.csv; residual_gap_span_comparison.csv; experiment_manifest.json.
- V7_9_x_Research_Findings_and_Next_Steps.docx.
- 10_Reconstruction_Benchmark.txt.
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx.
- Production_Retraining_Policy_YOLO_and_Structural_Safety.docx.