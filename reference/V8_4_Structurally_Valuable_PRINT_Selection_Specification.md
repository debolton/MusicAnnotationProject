**MUSIC ANNOTATION PROJECT**

V8.4 Structurally Valuable PRINT Selection

_Pre-registered development specification for lower-probability observed-print preservation_

**STATUS: Development-only challenger specification. V8.4 tests a learned value-and-safety selector over frozen V8.2 probabilities. It does not authorise a lower global threshold, alter V7.8 reconstruction, or reintroduce deterministic continuation rules.**

Project reference document | 20 October 2026

# 1. Correction to the research premise

V8.4 should not attempt to identify a musical symbol first and then preserve pixels because a hand-written rule says that beams, slurs or noteheads are valuable. That would recreate the deterministic-rule failure in a semantic form. V8.4 should instead learn whether a lower-probability candidate is both genuine PRINT and materially valuable to preserve, using observable spatial evidence and labels derived only after outputs are locked.

**Scientific question  
**Among connected candidate groups below the current strict PRINT boundary, can a learned, observable-only selector identify additional observed pixels whose exact preservation yields material structural benefit, while retaining the frozen V8.2/V7.8 safety, immutability and source-stability requirements?

# 2. Evidence carried forward

- Notebook 38: lower thresholds recover real benefit, but no tested global threshold jointly satisfied the compact D3-D8 envelope. The first truth-defined catastrophes appeared in the 0.925-to-0.900 transition, so global threshold relaxation is not the answer.
- Notebook 39: preservation benefit is structure-dependent. Beams, slurs/ties and noteheads/compact glyphs were prominent, and seeded reconstruction gains included spillover beyond directly copied pixels.
- Notebook 40/V8.3: legacy continuation and inpainting did not exploit smaller residual gaps successfully. V8.4 must therefore select what to preserve, not attempt deterministic completion.
- V7.9.x: hand-crafted geometry and continuity may remain inputs or audit descriptors, but not positive decision rules.

# 3. Experiment boundary

| **Element**          | **Frozen / permitted treatment**                                                                                                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Incumbent page       | Frozen V7.8 geometry_t80.                                                                                                                                                                                |
| Base probabilities   | Saved source-group OOF V8.2 p_print; no retraining or recalibration in the primary V8.4 experiment.                                                                                                      |
| Candidate population | Pixels or connected candidate groups newly available below the strict boundary, with primary analysis predeclared for 0.50 <= p_print < 0.75 and sensitivity strata aligned to existing threshold bands. |
| Permitted change     | A learned selector may elevate an eligible candidate group to PRINT. PRINT copies exact observed bytes.                                                                                                  |
| Fallback             | Candidates not selected retain the existing V8.2/V7.8 decision and page bytes.                                                                                                                           |
| Prohibited           | Global threshold adoption, clean-truth inference channels, raw page identity/coordinates, hand-coded per-symbol acceptance, deterministic line continuation and reconstruction changes.                  |

# 4. Unit of analysis

Use a connected candidate group, not a whole mask component and not an isolated pixel. Build groups from page-global connectivity among pixels in the predeclared probability stratum, with exact grouping constants frozen before outcome comparison. Retain pixel-level probabilities and outputs, but make the promotion decision at group level to capture coherent musical structure and concentration risk.

- Primary unit: connected candidate group inside one authoritative mask component.
- Minimum size: use the existing no-singleton rule as the floor; analyse larger size thresholds as predeclared ablations, not post-hoc fixes.
- Do not supply the semantic reviewer label as an inference input in the primary model. Use semantic structure only for stratified evaluation and failure analysis.

# 5. Two-label target: safety and value

V8.4 should use a two-axis target rather than a single PRINT label.

| **Target**                    | **Definition**                                                                                                                                   | **Operational role**                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| Genuine PRINT safety          | Candidate pixels correspond to genuine printed ink under the frozen truth/tolerance method, with no unacceptable false-annotation concentration. | Hard gate. A candidate cannot be selected without passing the safety head. |
| Structural preservation value | Exact preservation produces material benefit versus leaving the same candidate on the comparator path.                                           | Ranking/selection target, subject to safety.                               |
| UNCERTAIN                     | Truth instability, registration ambiguity, mixed evidence, OOD or insufficient confidence.                                                       | Fail closed to existing V8.2/V7.8 behaviour.                               |

Primary structural-value label should be based on counterfactual page outputs already defined by the project: comparator output versus exact observed-byte preservation for the candidate group. The value score should combine predeclared local MAE, ink/edge/skeleton continuity and perceptual-severity evidence without allowing any one metric to be chosen after results are seen.

# 6. Input allowlist

| **Input family**          | **Allowed representation**                                                                                                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V8.2 probability evidence | p_print values, local probability patch, band identity, calibrated margin and within-group distribution summaries.                                                                                                              |
| Observed spatial evidence | Raw and background-normalised greyscale patches centred on candidate pixels/groups.                                                                                                                                             |
| Mask geometry             | Authoritative mask patch, mask depth, distance to boundary and component context.                                                                                                                                               |
| Frozen V7.8 context       | Horizontal, vertical and compact support; preserve/remove/ambiguous route patches.                                                                                                                                              |
| Observed topology         | Edges, orientation responses, connectedness, run length, endpoints, junctions, curvature and thickness descriptors computed from observed evidence only.                                                                        |
| Multi-scale context       | Pre-register a compact local scale and one wider context scale. Recommended treatment comparison: existing 33x33 channels plus a 97x97 or 129x129 contextual branch, with architecture and crop alignment fixed before scoring. |
| Source challenge flags    | Sparse/hard-negative designation may be used for stratified reporting, not as a model input.                                                                                                                                    |

# 7. Model design

Use a controlled paired experiment so the source of improvement remains interpretable.

| **Arm**                         | **Model**                                                                                                | **Purpose**                                                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| A: probability-only baseline    | Regularised logistic or gradient-boosted model using group probability summaries and size/depth context. | Tests whether p_print distribution alone explains value.                                                                          |
| B: frozen 33x33 representation  | Compact CNN or pooled group encoder using existing observable channels.                                  | Tests whether the existing local topology can select valuable lower-probability groups.                                           |
| C: multi-scale structural model | Shared local encoder plus one wider contextual branch; group-level safety and value heads.               | Tests whether wider musical context adds value without explicit semantic rules.                                                   |
| D: semantic-oracle diagnostic   | Post-hoc stratification using Notebook 39 labels only. Never an inference model.                         | Measures the ceiling available from semantic grouping and checks whether learned selections concentrate in beneficial structures. |

# 8. Splits, leakage and calibration

- Reuse source-group separation. No source group may occur on both sides of a fold.
- Lock candidate-group construction before loading value labels.
- Generate out-of-fold safety and value scores for every eligible group.
- Cross-fit calibration by source group or page. Do not calibrate using the evaluation fold.
- Keep clean truth, counterfactual error, semantic review labels and perceptual ratings out of every inference channel.
- Retain a development audit reserve and a future untouched promotion cohort. V8.4 remains development-only.

# 9. Decision rule

A candidate group becomes V8.4 PRINT only if all conditions pass:

1. The safety probability exceeds a predeclared threshold and its false-preservation confidence bound is acceptable.
2. The predicted structural-value score is positive and exceeds a predeclared minimum material-benefit threshold.
3. The candidate is not OOD, schema-mismatched, truth-unstable or disallowed by the no-singleton/group-size rule.
4. No page-level or mask-component concentration limit is breached.
5. The final page continues to equal V7.8/V8.2 everywhere outside selected V8.4 PRINT candidates.

Do not encode rules such as "beam plus p_print >= 0.60 means PRINT". Structure type is a reporting stratum, not a deterministic permission.

# 10. Primary endpoints

| **Endpoint**                          | **Required interpretation**                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Additional genuine print preserved    | Benefit beyond the frozen comparator, reported by page, source group, threshold band and semantic structure.              |
| Additional false annotation preserved | Primary safety cost, including page/component concentration and clustered uncertainty.                                    |
| Structural-value gain                 | Counterfactual improvement in the frozen composite and each constituent metric; positive lower confidence bound required. |
| Perceptual benefit/harm               | Blinded P0-P4/B0-B4 review for selected and matched rejected candidates; diagnostic but mandatory.                        |
| Safe coverage                         | Selected eligible groups/pixels while all safety limits remain satisfied.                                                 |
| Source stability                      | No beneficial aggregate that depends on one source group; include leave-one-source-group-out recomputation.               |
| Integrity                             | Zero outside-mask drift and zero non-selected regression from the frozen comparator.                                      |

# 11. Predeclared go/no-go gates

Reuse the established D1-D9 style safety envelope, but add value-selection gates specific to V8.4. Exact numerical choices must be frozen in the machine-readable pre-registration before execution.

| **Gate**                       | **Requirement**                                                                                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V4-1 Integrity                 | Exact roster/fold/schema identity; no leakage, duplicate groups or non-finite inputs; exact replay.                                                             |
| V4-2 Immutability              | Zero outside-mask drift; zero change outside selected V8.4 PRINT groups; selected bytes equal observed bytes.                                                   |
| V4-3 Catastrophic safety       | Zero unacceptable catastrophic concentrations pooled and by source group; retain truth and perceptual severity records.                                         |
| V4-4 False preservation        | Meet the registered false/true and false-preservation ceilings with page-clustered upper bounds.                                                                |
| V4-5 Positive structural value | Pooled structural-value gain > 0 with page-clustered 95% lower bound > 0.                                                                                       |
| V4-6 Baseline dominance        | Beat probability-only and frozen 33x33 baselines without worsening safety.                                                                                      |
| V4-7 Non-trivial coverage      | Predeclare minimum selected groups, pages and source groups. Do not lower the requirement after seeing outcomes.                                                |
| V4-8 Source stability          | No negative-utility eligible source group and no single group dominates pooled benefit.                                                                         |
| V4-9 Semantic breadth          | Report results for beams, slurs/ties, noteheads/compact glyphs, staff lines and other classes; no promotion if benefit is an undisclosed single-class artefact. |
| V4-10 Visual audit             | Review all harmful candidates, top benefits, highest disagreement/OOD cases and matched rejected candidates.                                                    |

# 12. Required outputs

- 01_pre_registration.json
- 02_candidate_group_manifest.parquet
- 03_source_group_fold_map.csv
- 04_oof_safety_and_value_predictions.parquet
- 05_counterfactual_value_labels.parquet
- 06_results_by_page.csv
- 07_results_by_source_group.csv
- 08_results_by_probability_band.csv
- 09_results_by_structure.csv
- 10_catastrophic_and_harmful_groups.csv
- 11_visual_review_queue.csv
- 12_whole_page_outputs/
- 13_replay_and_integrity/
- 14_experiment_manifest.json
- 15_v8_4_verdict.json
- 16_executive_summary.txt
- upload/ standard final package with SHA-256 manifest

# 13. Decision logic

| **Outcome**                                          | **Decision**                                                                                                           |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Integrity/leakage failure                            | Invalidate run before interpreting performance.                                                                        |
| Safety fails                                         | Reject V8.4 at the frozen rule; retain comparator.                                                                     |
| Safety passes, value fails                           | Close the selected architecture as non-beneficial; do not relax the value threshold.                                   |
| Value passes but source/semantic concentration fails | Retain as scoped research only; no general V8.4 claim.                                                                 |
| Multi-scale arm does not beat 33x33                  | Conclude wider context did not add practical value under the frozen experiment.                                        |
| All development gates pass                           | Freeze one challenger in a separate step, then commission fresh untouched evidence. No automatic production authority. |

# 14. Implementation sequence

6. Freeze the V8.4 pre-registration, candidate bands, group construction, folds, inputs, model arms and gates.
7. Resolve and hash the exact V8.2/Notebook 38/39 source artefacts.
8. Build candidate groups and lock the manifest before outcome generation.
9. Create counterfactual value labels and semantic strata after the candidate lock.
10. Train paired arms under source-group OOF and write all OOF scores.
11. Calibrate cross-fitted safety/value scores and apply the frozen decision rule.
12. Assemble exact-preservation whole pages, assert immutability and lock outputs.
13. Score truth and counterfactual value; calculate clustered uncertainty and source stability.
14. Complete blinded perceptual audit and write the fail-closed verdict.
15. Package the final upload bundle and preserve the comparator and failed arms.

# 15. What V8.4 does not authorise

- No lower global PRINT threshold.
- No semantic hard-coding or deterministic continuation.
- No change to V7.8 reconstruction, routing or mask authority.
- No inference use of clean truth, Notebook 39 labels or perceptual ratings.
- No transfer of development outcomes into an untouched promotion claim.
- No automatic production deployment.

# 16. Final commissioning statement

Commission V8.4 as a learned group-level safety-and-value selector over frozen lower-probability V8.2 candidates. The primary experimental variable is whether observable local plus wider spatial context can identify candidate groups whose exact preservation is both safe and materially beneficial. The experiment must compare probability-only, existing 33x33 and one predeclared multi-scale treatment under identical source-group folds, labels, calibration and gates. Semantic structure remains an evaluation stratum, not a decision rule.

Confidence: 91%. The architecture and controls follow directly from the project evidence. Confidence is lower for the optimal candidate band, group construction and wider context size because those require pre-registration choices and have not yet been tested as one controlled experiment.

# Source basis

- Notebook38_Envelope_Relaxation_Project_Reference_FULL_DEFINITION_v1_0.docx.
- Notebook 39 upload: structure_reconstruction_sensitivity_ranking.csv; primary_075_vs_050_structure_summary.csv; structure_ranking_by_threshold_pair.csv; reviewed_region_metrics.csv; semantic_structure_review.csv.
- V8.3 closure evidence: executive_summary.txt; continuation_applicability_summary.csv; paired_050_vs_075_method_deltas.csv.
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx.
- V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx.
- V7_9_x_Research_Findings_and_Next_Steps.docx.
- V8_2_Development_Whole_Page_Experiment_Decision_Record.docx.
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx.