**MUSIC ANNOTATION PROJECT**

V8.6 Residual Reconstruction Readiness Programme

_Notebook 45: Mask Readiness and Oracle Audit  
Notebook 46: Frozen Restorer Mask Crossover_

Project reference and implementation specification | 21 October 2026

**Decision: use notebook numbers 45 and 46. Call the research strand V8.6, provided V8.6 is explicitly described as a new development-only residual-mask and reconstruction-readiness programme, not as promotion of V8.4 and not as a change to the V7.8 incumbent.**

# 1. Executive decision

Notebooks 43 and 44 already exist, so the proposed work should be implemented as Notebooks 45 and 46. V8.6 is an appropriate project version because the causal question changes materially: V8.4 selects safe and valuable observed PRINT, whereas V8.6 tests whether those preservation decisions can condition a residual mask sufficiently well to make reconstruction measurably easier.

- V8.4 remains the source selector and exact-preservation layer.
- V8.6 does not lower the global PRINT threshold or alter the V7.8 production fallback.
- Notebook 45 isolates mask construction and measures reconstruction readiness before any learned restorer is judged.
- Notebook 46 holds the mask arms fixed and crosses them with frozen restoration methods.
- Clean truth remains prohibited from mask inference and reconstruction; it is used after locks for oracle construction and scoring only.

# 2. Why V8.6 is a separate version

| **Lineage** | **Scientific variable**                                                                                                              | **Status**                     |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| V8.3        | Whether smaller residual gaps rescue historical deterministic reconstruction                                                         | Closed failed challenger       |
| V8.4        | Whether lower-probability connected groups are safe and structurally valuable to preserve                                            | Development-only selector      |
| V8.6        | Whether ownership-aware residual carving creates a better-conditioned reconstruction problem, and whether a frozen restorer benefits | New development-only programme |

The version name should not imply that V8.5 was skipped accidentally. The run manifests should include a short lineage note stating that existing Notebooks 43 and 44 occupy the immediately preceding notebook numbers and that V8.6 is the next material experiment version.

# 3. Governing principles

- Authoritative-mask boundary: intervention remains confined to the reviewed dilation-2 mask.
- Exact preservation: selected observed PRINT bytes are immutable inputs, never reconstructed.
- Fallback identity: where V8.6 does not authorise a change, retain frozen V7.8 behaviour.
- Truth firewall: masks M0-M3 are locked before clean truth is used to create M4 or score outcomes.
- Paired source-group evidence: every eligible region receives all applicable mask and restorer arms.
- Asymmetric risk: invented notation and damaged genuine print are materially worse than unresolved residue.
- No post-outcome tuning: collar constants, mask definitions, restorer settings and gates are recorded before scoring.

# 4. Programme architecture

| **Stage** | **Notebook** | **Purpose**                                                                                    | **Primary output**                                                   |
| --------- | ------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| A         | 45           | Build and lock M0-M3, then create post-lock oracle M4 and mask-only readiness evidence         | Locked mask manifest, readiness metrics, oracle gap and review queue |
| B         | 46           | Apply identical frozen restorers to M0-M4 and estimate the causal benefit of mask conditioning | Paired crossover metrics, oracle-normalised recovery and verdict     |

# 5. Notebook 45 specification

## 5.1 Scientific question

Does V8.4-derived exact preservation, with a conservative uncertainty collar, produce a residual mask that contains less genuine PRINT, exposes more trustworthy boundary context and approaches an oracle residual-mask construction?

## 5.2 Required inputs

- One complete V8.4 Notebook 41 run, including experiment manifest, verdict, candidate lock, OOF group predictions and candidate-group pixels.
- The exact V8.2/V7.5 source paths named by the V8.4 manifest.
- Observed, V7.8 baseline, authoritative mask and frozen route/support arrays.
- Registered clean pages, loaded only after the M0-M3 lock is written.
- Development score manifest or source-group identity from the source predictions.

## 5.3 Mask arms

| **Arm** | **Definition**                                                                                                                                                                                             | **Truth role**                               |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| M0      | Frozen V7.8 ambiguous route.                                                                                                                                                                               | No clean truth.                              |
| M1      | M0 minus strict valid V8.2 PRINT coordinates.                                                                                                                                                              | No clean truth.                              |
| M2      | M1 minus selected safe-and-positive-value V8.4 candidate groups.                                                                                                                                           | No clean truth.                              |
| M3      | M2 plus a withheld uncertainty collar around protected PRINT and around calibrated ownership-uncertain candidates. The collar is excluded from reconstruction, not filled or whitened.                     | No clean truth.                              |
| M4      | Post-lock oracle residual: preserve genuine observed PRINT; designate annotation-only pixels as removable; retain only unsupported genuine-print loss or truth-ambiguous pixels as reconstruction targets. | Clean truth permitted after M0-M3 lock only. |

## 5.4 Frozen uncertainty-collar rule

The first V8.6 implementation uses a conservative observable-only collar. A pixel is withheld from reconstruction when it lies inside the authoritative mask and is within the configured dilation radius of either protected PRINT or a lower-probability candidate group that fails either the V8.4 safety or value decision. The notebook exports sensitivity tables for radii 1, 2 and 3, but the primary M3 arm is fixed at radius 2 before outcomes are scored.

## 5.5 Mask-only endpoints

- Residual pixels, components, area, depth and maximum unsupported span.
- Genuine PRINT incorrectly left inside the reconstruction target.
- Annotation incorrectly protected from removal.
- Protected genuine PRINT and protected false annotation.
- Boundary contact with trusted PRINT and paper/removal evidence.
- Two-sided horizontal and vertical support, endpoint compatibility and local edge support.
- Topological ambiguity proxies: number of structures entering the boundary, branching and component intersection.
- Oracle gap: normalised distance from M0-M3 to M4 for each mask-quality measure.

## 5.6 Notebook 45 verdict

| **Verdict**              | **Meaning**                                                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| READY_FOR_CROSSOVER      | M3 improves safety and boundary conditioning over M0/M2, remains immutable, and captures measurable oracle opportunity across more than one source group. |
| MASK_RESEARCH_REQUIRED   | M3 improves some metrics but remains materially distant from M4 or is source-concentrated.                                                                |
| REJECT_COLLAR_HYPOTHESIS | M3 does not improve M2 or worsens genuine-print exposure/protection.                                                                                      |
| INVALID_RUN              | Any truth leakage, identity mismatch, outside-mask error or lock failure.                                                                                 |

# 6. Notebook 46 specification

## 6.1 Scientific question

Holding restoration methods and settings fixed, do the M2/M3 masks produce materially better reconstruction than M0/M1, and how much of the M4 oracle opportunity do they recover?

## 6.2 Restorer arms

| **Arm** | **Method**                                                                                                     | **Role**                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| R0      | Leave unresolved / paper-only baseline                                                                         | Safety floor and no-hallucination comparator.                                            |
| R1      | Biharmonic inpainting                                                                                          | Simple deterministic benchmark.                                                          |
| R2      | Telea inpainting                                                                                               | Canonical historical local benchmark.                                                    |
| R3      | Frozen small partial/gated-convolution U-Net trained only on synthetic masks drawn from training source groups | Tests whether learning, without diffusion generation, exploits the conditioned residual. |

R3 is trained separately inside each source-group outer fold. Test pages and test-source clean pixels are excluded from training. Synthetic occlusions use the shape library exported by Notebook 45, including real residual component geometry but applied to clean training pages at unrelated locations. The model, optimiser, epochs and seed are fixed before evaluation.

## 6.3 Primary estimands

- Within-restorer mask effect: M3 minus M0 and M3 minus M2.
- Within-mask learning effect: R3 minus R1/R2.
- Interaction: whether R3 benefits more from M3 than deterministic restorers do.
- Oracle-normalised recovery: (M3 minus M0) divided by (M4 minus M0), with denominator guards and separate reporting where the oracle offers no improvement.
- Source-group and semantic-stratum stability.

## 6.4 Scoring hierarchy

- Printed-ink precision, recall and F1.
- Tolerant edge F1.
- Skeleton connectivity, connected components, endpoints and junctions.
- False inserted ink and missing genuine ink.
- MAE and SSIM as secondary tonal measures only.
- Outside-mask drift and protected-PRINT modification, both required to be zero.
- Blinded visual audit of the largest benefits, largest regressions and oracle-gap cases.

# 7. Statistical and locking design

- Region is the paired local unit; page is the bootstrap cluster; source group is the generalisation stratum.
- Notebook 45 writes a semantic lock over M0-M3 before loading clean truth. M4 and truth-derived metrics are stored under a separate post-lock section.
- Notebook 46 verifies Notebook 45 hashes and writes all restoration outputs before scoring against clean truth.
- Ten thousand page-clustered bootstrap resamples are used by default.
- No restorer-specific mask tuning and no best-of-seed result selection are permitted.
- Any rerun with changed collar rules, mask definitions, training corruption distribution or network architecture is a new V8.6 experiment run.

# 8. File and directory contracts

Notebook 45 output root: observed_print_classifier_v8_6_notebook45/&lt;run_id&gt;/

Notebook 46 output root: observed_print_classifier_v8_6_notebook46/&lt;run_id&gt;/

| **Notebook 45 required artefact**           | **Purpose**                                                  |
| ------------------------------------------- | ------------------------------------------------------------ |
| 01_pre_registration.json                    | Frozen arm definitions, constants and permitted truth roles. |
| source_record.json                          | Exact V8.4, V8.2 and V7.5 identities and hashes.             |
| tables/region_manifest.parquet              | Paired eligible residual regions.                            |
| masks/&lt;page&gt;/&lt;region&gt;/M0-M4.png | Exact mask-arm arrays.                                       |
| mask_lock.json                              | Semantic hashes for M0-M3 before truth.                      |
| tables/mask_readiness_metrics.csv           | Mask-only and post-lock truth metrics.                       |
| tables/oracle_gap_summary.csv               | Distance to M4 by arm/page/source group.                     |
| visual_audit/                               | Blinded and diagnostic review panels.                        |
| upload/notebook45_upload_bundle.zip         | Curated final package.                                       |

| **Notebook 46 required artefact**                                     | **Purpose**                                           |
| --------------------------------------------------------------------- | ----------------------------------------------------- |
| source_record.json                                                    | Verified Notebook 45 mask lock and source identities. |
| restorer_pre_registration.json                                        | R0-R3 settings and training controls.                 |
| outputs/&lt;page&gt;/&lt;region&gt;/&lt;mask&gt;/&lt;restorer&gt;.png | All paired reconstruction outputs.                    |
| tables/paired_region_metrics.csv                                      | Complete paired results.                              |
| tables/oracle_normalised_recovery.csv                                 | M3 recovery of M4 opportunity.                        |
| tables/results_by_page.csv                                            | Page-cluster evidence.                                |
| tables/results_by_source_group.csv                                    | Generalisation evidence.                              |
| v8_6_verdict.json                                                     | Fail-closed decision.                                 |
| upload/notebook46_upload_bundle.zip                                   | Curated final package.                                |

# 9. Decision rules

| **Observed result**                                                             | **Decision**                                                                                                                       |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| M3 does not improve M2 mask readiness                                           | Reject the collar hypothesis; do not commission a more complex restorer on this mask design.                                       |
| M3 improves readiness but remains far below M4                                  | Continue mask/ownership research; reconstruction is premature.                                                                     |
| M3 approaches M4 but all restorers remain poor                                  | Treat reconstruction as information-limited or representation-limited; do not blame mask construction.                             |
| R3 improves M0 and M3 equally                                                   | Learning helps, but V8.4 carving is not the cause.                                                                                 |
| R3 improves M3 materially more than M0/M2, with topology and source stability   | V8.6 demonstrates that V8.4-conditioned carving makes reconstruction easier; commission a separate learned-restoration challenger. |
| Any invented-notation concentration, protected-byte drift or outside-mask drift | Fail closed regardless of aggregate gains.                                                                                         |

# 10. Implementation sequence

1. Run Notebook 45 with all configuration values frozen and automatic source resolution enabled.
2. Review the input/source hash tables and ensure M0-M3 lock creation succeeds.
3. Inspect readiness and oracle-gap tables; do not alter the primary collar after outcomes appear.
4. Proceed to Notebook 46 only if Notebook 45 is valid; Notebook 46 may still run for diagnosis when the readiness verdict is negative, but the result cannot be called a positive V8.6 progression.
5. Run Notebook 46 on GPU for R3; CPU is sufficient for R0-R2 smoke testing.
6. Complete blinded visual audit fields where provided, rerun only the summary/finalisation cells, and preserve the original reconstructions.
7. Save the Word reference, both notebooks and their upload bundles in the project reference archive.

# 11. Source basis

- V8_4_Structurally_Valuable_PRINT_Selection_Specification.docx
- 41_V8_4_Structurally_Valuable_PRINT_Selection.ipynb.txt
- 42_V8_4_Failure_Mode_Semantic_Breadth_and_Visual_Audit.ipynb.txt
- V8_3_Residual_Gap_Continuation_Closure_Record.docx
- 10_Reconstruction_Benchmark.txt
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx
- V8_2_Development_Whole_Page_Experiment_Decision_Record.docx
- 28_V8_2_Development_Whole_Page_Experiment.txt
- Future_Direction_Generative_Residual_Reconstruction_FLUX1_Fill.docx

# 12. Confidence and next question

Confidence: 92%. The notebook numbering correction is definite from the project sequence supplied. Confidence is high that V8.6 is an appropriate development-version label because the experiment changes the causal variable from PRINT selection to residual-mask conditioning and reconstruction readiness. The remaining uncertainty is whether the existing Notebook 41 outputs contain every field needed to rebuild candidate groups and selected groups without minor schema adaptation; both notebooks therefore include explicit column checks and fail-closed source resolution.

**Next question: what result threshold should convert a positive V8.6 development finding into a separately frozen learned-restoration challenger rather than another exploratory notebook?**