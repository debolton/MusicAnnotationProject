**MUSIC ANNOTATION PROJECT**

V7.9.x Research Findings  
and Recommended Next Steps

Project reference document

**Decision status: close the deterministic V7.9.x challenger series; retain frozen V7.8**

| **Project**          | Music Annotation Project                               |
| -------------------- | ------------------------------------------------------ |
| **Document purpose** | Record V7.9.x findings, decisions and next actions     |
| **Date**             | 14 October 2026                                        |
| **Status**           | Reference recommendation; not production authorisation |

Core conclusion  
The V7.9.x programme found real recoverable printed-ink signal, especially in connected-dark and vertical-continuation cases, but every proposed deterministic positive-preservation gate failed page-grouped safety and generalisation tests. V7.8 should remain the frozen incumbent. Further threshold engineering on the same development pages should stop.

# 1. Executive summary

V7.9.x investigated whether additional observed dark ink inside an authorised annotation-removal mask could be preserved without retaining unacceptable handwriting or other annotation. The research progressively tested broad connected-dark routing, safety-first deterministic gating, bidirectional continuity, and orientation-specific horizontal, vertical and compact-glyph continuation.

Recommendation  
Preserve V7.8 unchanged. Close V7.9, V7.9.2, V7.9.3 and V7.9.4 as failed but informative challengers. Continue the supervised Pilot Notebook 09. If further preservation research is commissioned, change the hypothesis materially: use a local observed-print classifier or equivalent learned pixel/patch decision layer, not another hand-built threshold gate.

- V7.9 connected-dark increased printed-ink recall, proving that genuine print was being lost by frozen V7.8 in some cases.
- The gain was not selective enough: all broad V7.9 candidates preserved materially more false annotation and reduced overall ink F1 and edge F1.
- V7.9.2 removed catastrophic cases more effectively than it identified genuinely beneficial cases; page-grouped out-of-fold utility was negative.
- V7.9.3 showed that broad bidirectional or cross-axis overlap was not sufficient positive evidence and was available in too few folds.
- V7.9.4 proved the exact NPZ added-pixel identity, separated horizontal, vertical and compact channels, and still failed multiple predeclared out-of-fold gates.
- The cumulative result is a model-selection decision, not a software failure: the incumbent remains safer, and the failed challengers should be preserved as evidence.

# 2. Decision context and governing principles

The recommendations maintain the established project architecture. Annotation confidence and correction safety are separate axes. A mark being likely annotation does not establish that removal is safe. V7.5 therefore confines intervention to the controlled mask, copies confidently observed printed geometry exactly, removes confident non-structural annotation, reconstructs only ambiguity, and leaves all pixels outside the mask unchanged.

Non-negotiable boundary  
All future challengers should remain inside the authoritative removal mask, preserve outside-mask pixels byte-for-byte, and keep clean truth out of inference and reconstruction inputs. Clean truth may be used only for post-hoc development labels, scoring and diagnostics.

The project also requires challengers to remain separate from incumbents, use page-grouped validation, exclude truth and raw coordinates from predictors, and avoid automatic promotion. Structural-safety policy changes are slower-moving and should be commissioned only when reconstruction, routing, predictors, component construction or outcome definitions materially change.

# 3. V7.9.x experimental record

| **Version** | **Hypothesis**                                                                                               | **Key outcome**                                                                                                                                                                                                                 | **Decision**                                                             |
| ----------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| V7.9        | Preserve additional observed dark ink using connected-dark, consensus or union routing.                      | Recall improved, but false annotation increased sharply and overall ink/edge F1 deteriorated. Connected-dark was the least poor option.                                                                                         | Reject current candidates; retain only as research evidence.             |
| V7.9.2      | Use deterministic component thresholds as a safety-first fallback gate for connected-dark.                   | Full-development fit appeared favourable, but page-grouped OOF produced 46 true-print pixels versus 66 false-annotation pixels, utility −218, with zero catastrophic selections.                                                | Useful as exclusion evidence, not as a final positive-preservation gate. |
| V7.9.3      | Use bidirectional continuity, including cross-axis overlap, as positive structural evidence.                 | Only two OOF components were selected; 3 true-print versus 12 false pixels, utility −45. Eight of ten folds found no eligible rule.                                                                                             | Reject cross-axis coincidence as sufficient positive evidence.           |
| V7.9.4      | Prove exact added-pixel identity, then evaluate horizontal, vertical and compact continuation independently. | Identity passed; combined OOF selected 126 true versus 130 false pixels, two catastrophic components, utility −394. Horizontal was poor; vertical showed signal but remained unsafe and concentrated; compact selected nothing. | Close deterministic V7.9.x threshold development.                        |

# 4. Findings by challenger

## 4.1 V7.9 broad route challengers

| **Candidate**  | **Recall change**        | **Extra true print** | **Extra false annotation** | **False per true** | **Overall interpretation**                                                             |
| -------------- | ------------------------ | -------------------- | -------------------------- | ------------------ | -------------------------------------------------------------------------------------- |
| Connected-dark | +10.17 percentage points | 9,030                | 16,689                     | 1.85               | Best of the V7.9 candidates, but ink F1 and edge F1 both worsened materially.          |
| Consensus      | +8.62 percentage points  | 6,529                | 16,274                     | 2.49               | Dominated by connected-dark: less true-print gain for similar false-preservation cost. |
| Union          | +15.75 percentage points | 15,311               | 35,155                     | 2.30               | Highest recall, but achieved by preserving far too much dark material; reject.         |

Connected-dark produced genuine wins on some pages, confirming that the V7.8 preserve route can miss visible print. The same route produced severe failures on sparse-print or annotation-dominated pages. The central failure was not lack of recall; it was lack of selectivity.

## 4.2 V7.9.2 deterministic fallback gate

The deterministic gate was effective at suppressing the extreme catastrophic cases, with zero catastrophic OOF selections, but it did not generalise as a positive selector. The contrast between a favourable full-development fit and negative out-of-fold performance is evidence of overfitting to the development components.

| **Measure**             | **Full development** | **Page-grouped OOF** |
| ----------------------- | -------------------- | -------------------- |
| Selected components     | 10                   | 14                   |
| Selected pages          | 8                    | 12                   |
| True print added        | 54                   | 46                   |
| False annotation added  | 10                   | 66                   |
| False/true ratio        | 0.185                | 1.435                |
| Catastrophic selections | 0                    | 0                    |
| Utility                 | +14                  | −218                 |

## 4.3 V7.9.3 bidirectional continuity

The experiment showed that low bidirectional overlap can help exclude weak candidates, but complete overlap is not sufficient positive evidence. Cross-axis coincidence can occur at annotation junctions, crowded notation, unrelated crossings and enclosed gaps. It also wrongly disadvantages valid single-axis structures such as staff lines and stems.

- Only folds 6 and 8 found an eligible rule; eight of ten folds found none.
- Both eligible folds required added_cross_axis_fraction = 1.0.
- The two OOF selections contributed 3 true-print and 12 false-annotation pixels.
- The reported conditional feature agreement was misleading without rule-bearing-fold coverage; 2/2 eligible folds agreed, but only 2/10 folds had a rule.

## 4.4 V7.9.4 orientation-specific continuation

| **Channel** | **Selected components** | **Pages** | **True print** | **False annotation** | **False/true** | **Catastrophic** | **Utility** | **Rule-bearing folds** |
| ----------- | ----------------------- | --------- | -------------- | -------------------- | -------------- | ---------------- | ----------- | ---------------------- |
| Horizontal  | 5                       | 2         | 12             | 116                  | 9.67           | 2                | −452        | 1/10                   |
| Vertical    | 6                       | 6         | 115            | 68                   | 0.59           | 1                | −157        | 3/10                   |
| Compact     | 0                       | 0         | 0              | 0                    | n/a            | 0                | 0           | 0/10                   |
| Combined    | 10                      | 6         | 126            | 130                  | 1.03           | 2                | −394        | n/a                    |

V7.9.4 resolved an important audit issue: exact added-pixel identity passed for 1,454 components across 51 pages with zero mismatches. The poor result therefore cannot be dismissed as a mismatch between recorded outcomes and analysed pixels. Vertical continuation contains a real signal, but the current evidence was neither safe nor broadly available enough to support a deterministic gate.

# 5. Technical interpretation

Principal research conclusion  
Deterministic geometry remains useful as a conservative veto and audit signal. It has not demonstrated adequate generalisation as a positive selector for additional observed print inside broad annotation masks.

1. The V7.9 candidates proved that recoverable printed ink exists beyond the V7.8 preserve route.
2. Broad darkness-based preservation is too permissive because annotation and print share the same basic visual property: dark observed ink.
3. Component-level decisions are often too coarse. A single authorised mask component can contain genuine print, annotation, blank paper and ambiguous overlap simultaneously.
4. Hand-crafted positive evidence progressively became more elaborate, but rule availability and thresholds remained unstable across page groups.
5. The vertical signal warrants retention as a future model feature, not deployment as a standalone deterministic rule.
6. The compact-glyph hypothesis was not disproved in principle; the tested topology did not produce usable OOF selections.
7. Further tuning against the same 82-page development collection would increase overfitting risk without changing the underlying information available to the rule.

# 6. Recommendations

## 6.1 Freeze and retain V7.8

- Keep V7.8 geometry_t80 and routing semantics unchanged as the incumbent.
- Do not add any V7.9.x preserve route to Notebook 09 or to a production correction path.
- Retain V7.9.x notebooks, manifests, identity tables, OOF decisions, failed verdicts and representative failure crops as research artefacts.
- Record V7.9.x as a failed challenger series with useful negative findings, not as an abandoned implementation issue.

## 6.2 Continue the supervised Pilot Notebook 09

Pilot Notebook 09 is the appropriate operational next step because it produces actual correction proposals, preserves the canonical V7.8 implementation, applies the frozen structural policy, and requires human component decisions before final approval. It does not auto-accept, retrain, recalibrate or alter the frozen threshold.

- Use the existing accept, leave unchanged, defer and mask-edit/rescore workflow.
- Where V7.8 cannot safely recover obscured print, show the proposal honestly and allow the reviewer to leave the region unchanged.
- Preserve reviewer decisions, edits, source context and routing artefacts as future evidence.

## 6.3 Stop deterministic V7.9.x threshold development

- Do not commission V7.9.5 as another grid of handcrafted darkness, geometry or continuity thresholds.
- Do not relax false-preservation constraints merely to increase recall or automation coverage.
- Do not reuse the frozen V7.8 independent collection for training or threshold selection.

## 6.4 If a new research track is commissioned, use a learned observed-print classifier

The recommended new hypothesis is a narrow local classifier that decides whether an already observed dark pixel or small patch inside the authorised removal mask represents genuine printed ink. This is not generative reconstruction and should not invent missing notation.

| **Output class** | **Treatment**                                                         |
| ---------------- | --------------------------------------------------------------------- |
| PRINT            | Copy the observed pixel exactly.                                      |
| ANNOTATION       | Route through the existing removal process.                           |
| UNCERTAIN        | Retain the existing ambiguous/review path; default to V7.8 behaviour. |

Suggested inference inputs should remain observable and truth-free: local greyscale patch, authorised mask geometry, distance to the mask boundary, V7.8 horizontal/vertical/compact support, local edges, background-normalised intensity, route context, and carefully designed orientation/topology descriptors. Clean registered pages should be used only to create development labels and to score frozen predictions.

# 7. Proposed next research experiment

## 7.1 Target and unit of analysis

- Primary target: pixel or small-patch classification of observed dark material inside the authorised mask as print, annotation or uncertain.
- Avoid making one preserve/remove decision for an entire broad mask component.
- Preserve exact observed pixels classified as print; do not synthesise their appearance.

## 7.2 Leakage and dataset controls

- Use V7.5 development pages and eligible future reviewed pilot evidence for development.
- Keep clean truth out of model inputs and reconstruction. Use clean truth only for post-hoc labels and evaluation.
- Group train/validation splits by page and, where available, by score/source group.
- Keep raw page coordinates and page identity out of the predictor schema.
- Keep the frozen V7.8 independent set outside training if it remains a permanent regression set.
- Create a new untouched holdout before any production claim.

## 7.3 Baselines and evaluation

- Compare the challenger against frozen V7.8, not against V7.9 connected-dark alone.
- Primary safety measure: false annotation preserved by the challenger beyond V7.8.
- Primary benefit measure: genuine printed ink additionally preserved beyond V7.8.
- Report pixel totals, component counts, page coverage, source-group stability, catastrophic cases and uncertainty.
- Use page-grouped out-of-fold predictions for development decisions. Do not rely on full-development fit.
- Include hard-negative and sparse-print pages, because these exposed the major V7.9 failure modes.
- Require visual review of the most beneficial, most false-heavy and most uncertain cases.

## 7.4 Predeclared promotion principles

Numerical promotion thresholds should be frozen before inspecting the new independent result. At minimum, the challenger should have zero integrity failures, zero outside-mask drift, no unacceptable catastrophic preservation cases, positive page-grouped utility, a predeclared false-preservation bound with uncertainty, stable behaviour across source groups, and non-trivial coverage. No production promotion should occur automatically.

# 8. Recommended work plan

| **Priority** | **Action**                                                                     | **Output / decision**                                                                        |
| ------------ | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| 1            | Archive V7.9.x as a concluded failed-challenger programme.                     | Reference record, manifests, verdicts, OOF tables and selected visual evidence retained.     |
| 2            | Continue supervised Pilot Notebook 09 on representative production-like pages. | Reviewed corrections, leave-unchanged cases, mask edits and audit evidence.                  |
| 3            | Define the observed-print classifier experiment before writing code.           | Frozen target definition, input allowlist, prohibited inputs, split rules and success gates. |
| 4            | Build a read-only dataset audit for pixel/patch labels.                        | Class balance, page/source coverage, registration sensitivity and ambiguous-label analysis.  |
| 5            | Train a challenger without touching V7.8.                                      | Versioned model, schema, OOF predictions and development report.                             |
| 6            | Compare the frozen challenger with frozen V7.8 on a new untouched cohort.      | Independent report including uncertainty and source-group stability.                         |
| 7            | Promote only through a documented release decision.                            | Separate detector ID, structural-policy ID, reconstruction version and audit fingerprints.   |

# 9. Actions not recommended

- Do not promote connected-dark, consensus, union, V7.9.2, V7.9.3 or V7.9.4.
- Do not tune additional deterministic thresholds against the same development outcomes.
- Do not make cross-axis overlap a positive-print rule.
- Do not treat higher recall alone as improvement when false annotation, ink F1 or edge F1 regress.
- Do not incorporate V7.9.x logic into Notebook 09 before a separately commissioned challenger passes its gates.
- Do not collapse the YOLO confidence map, authoritative annotation mask and reconstruction routing map into one artefact.
- Do not weaken structural-safety standards to increase automation coverage.

# 10. Governance and audit record

The finished record should continue to identify the annotation detector and structural-safety policy separately. A future production release should record the detector dataset and checkpoint identity, structural model and calibrator identity, ordered predictor schema, thresholds, reconstruction version, configuration fingerprints, independent validation, and approval decision.

Failed challengers should remain reproducible and discoverable. The V7.9.x outputs provide an important evidence chain: exact baseline equivalence, candidate route metrics, recovered NPZ integrity, component/pixel identity, page-grouped decisions, and fail-closed verdicts.

# 11. Evidence register

This document draws on the V7.9.x outputs produced and reviewed during the challenger programme, including the principal files below:

- V7.9.1 stage4_finalist_summary.csv, stage4_page_metrics.csv and stage4_component_metrics.csv
- V7.9.2 deterministic_gate.json, page_grouped_oof_summary.json, page_grouped_oof_component_decisions.csv and page_grouped_fold_results.csv
- V7.9.3 v7_9_3_verdict.json, page_grouped_oof_summary.json, bidirectional component table and fold results
- V7.9.4 v7_9_4_verdict.json, exact added-pixel identity tables, channel fold results and OOF decisions

The recommendations also align with the following established project references:

- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx: two-axis separation, controlled-mask boundary, exact preservation, ambiguity routing and deferred auto-acceptance.
- 10_Reconstruction_Benchmark.txt: clean truth used only after reconstruction, outside-mask immutability, crop processing and page-level caching.
- 12_V7_7_Asymmetric_Structural_Risk_Calibration.txt: frozen V7.5 candidate, page-grouped validation, predictor firewall and retrospective thresholds.
- 17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt: frozen semantics, deterministic locking, semantic hashes and independent-data controls.
- 18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt: exact stage-by-stage equivalence and fail-closed audit.
- V7_7F_Design_Record_Applying_the_Frozen_Structural_Policy_to_V7_8.docx: no refit or threshold movement on independent data; uncertainty and source-group reporting.
- Production_Retraining_Policy_YOLO_and_Structural_Safety.docx: event-driven challengers, separately versioned systems, frozen incumbents and no automatic promotion.
- 09_Pilot_Proposed_Correction_Routing_Reviewer.txt: supervised-only proposals, canonical V7.8 processing, regression gate and human accept/unchanged/defer/edit decisions.

# 12. Final decision statement

Decision  
V7.8 remains the incumbent reconstruction and routing policy. V7.9.x is closed as a completed failed-challenger research series. The project should continue the supervised Pilot Notebook 09 and, if further preservation research is justified, commission a materially different learned observed-print classifier with strict leakage controls, page/source-group validation and a new untouched holdout.

Confidence: 97% for the decision not to promote any V7.9.x deterministic gate, because multiple independent page-grouped criteria failed materially and V7.9.4 proved exact added-pixel identity. Confidence: 88% that a learned local observed-print classifier is the best next research direction, because that is a new hypothesis rather than an experimentally proven solution.

Next question: What exact training target, input allowlist, uncertainty class and predeclared independent-promotion gates should be frozen for the observed-print classifier before another notebook is built?