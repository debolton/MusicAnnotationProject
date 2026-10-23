**MUSIC ANNOTATION PROJECT**

Notebook 38  
Envelope Relaxation and Perceptual Severity Calibration

_Full Experiment Definition, Review Rubric, File Contract and Handover Record_

**STATUS: Development-only protocol. This document does not relax a production gate, freeze a threshold, authorise independent testing, or approve automatic use.**

| **Field**                    | **Definition**                                                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Document version             | 1.0                                                                                                                         |
| Reference date               | 19 October 2026                                                                                                             |
| Proposed executable notebook | 38_V8_2i_Envelope_Relaxation_and_Perceptual_Severity_Calibration.ipynb                                                      |
| Scientific role              | Post-outcome development audit of the D3–D8 envelope using frozen Notebook 37 decisions and blinded perceptual adjudication |
| Incumbent comparator         | Frozen V7.8 corrected page                                                                                                  |
| Primary source run           | Notebook 37 V8.2h downward PRINT-threshold safe-coverage frontier                                                           |
| Promotion authority          | None                                                                                                                        |
| Supersedes                   | The short initial Notebook 38 envelope-relaxation draft                                                                     |

# Document Map

- 1\. Executive decision and correction to the premise
- 2\. Background carried forward from Notebook 37
- 3\. Scientific question, estimands and scope
- 4\. Non-negotiable controls
- 5\. Experimental design and stages
- 6\. Perceptual severity and benefit rubrics
- 7\. Review presentation, blinding and adjudication
- 8\. D3–D8 envelope-relaxation analysis
- 9\. Statistical and sensitivity analysis
- 10\. Gate-specific file requirements
- 11\. Input file contract and minimum schemas
- 12\. Required outputs and final upload package
- 13\. Fail-closed rules, deviations and change control
- 14\. Decision logic and permissible conclusions
- 15\. Implementation handover for a separate Copilot chat
- 16\. Source basis and assumptions

# 1. Executive Decision and Correction to the Premise

**Notebook 38 will test whether the numerical D3–D8 development envelope is misaligned with perceptual harm. It will not begin by assuming that the envelope is too strict.**

Notebook 37 showed that the downward PRINT-threshold frontier contains genuine-print recovery, but no tested threshold met the compact D3–D8 envelope. The next question is therefore not "which lower threshold should be adopted?" It is "which gate failures correspond to material page-level harm, which are merely conservative numerical alerts, and how stable is that distinction across pages and source groups?"

The protocol separates three things that must not be collapsed:

- Truth error: clean-truth scoring identifies extra preserved pixels as PRINT, ANNOTATION or truth-UNCERTAIN.
- Perceptual severity: a blinded reviewer judges whether the resulting corrected page or local context is visibly harmed, helped or unchanged.
- Operational decision: a revised envelope may be proposed only after numerical and perceptual evidence are reconciled and tested on a source-group-separated audit reserve.

A truth-defined catastrophic component remains a required audit flag. Notebook 38 may determine that some flagged components are perceptually negligible; it must not erase the original flag, relabel truth, or retroactively declare the component safe. Both the registered and perceptual classifications remain in the record.

# 2. Background Carried Forward from Notebook 37

The following findings are carried forward as the trigger for Notebook 38 and are not recomputed by this document:

- Notebook 37 was a development-only, post-outcome threshold-sensitivity audit that reused saved V8.2 source-group OOF probabilities without retraining, refitting or recalibrating.
- It swept effective PRINT thresholds from 0.995 down to 0.500, retained the page-global minimum run size of 2 and rebuilt each candidate page from V7.8 plus exact observed bytes at final PRINT coordinates.
- No threshold satisfied the compact D3–D8 envelope.
- The highest observed point utility occurred at 0.925, but the confidence-bound, false-preservation, minimum-benefit and source-stability requirements were not jointly satisfied.
- The first truth-defined catastrophic components appeared when moving from 0.925 to 0.900, which makes that transition a mandatory review stratum rather than evidence that 0.900 is acceptable.
- Whole-page threshold ladders were useful locators, but local and wide-context transition panels are required for causal interpretation of individual components.

**Notebook 38 must reuse the frozen Notebook 37 predictions and page assemblies. It is an envelope-calibration experiment, not a new classifier-training or probability-calibration experiment.**

# 3. Scientific Question, Estimands and Scope

## 3.1 Primary question

Within the frozen Notebook 37 development frontier, how far can D3–D8 be relaxed before the gain in genuine printed ink is outweighed by perceptually material retained annotation, unstable source-group behaviour or excessive reviewer burden?

## 3.2 Primary estimands

| **Estimand**                             | **Definition**                                                                                                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Perceptually acceptable extra true print | Additional true-print pixels preserved beyond V7.8 in cases rated P0 or P1 harm and with benefit rated B1–B4.                                                      |
| Perceptually material false preservation | Additional false-annotation pixels in a case rated P2–P4 harm.                                                                                                     |
| Operationally critical failure           | Any case rated P4, or any repeated P3 mechanism that plausibly changes or obscures musical meaning across a page/source group.                                     |
| Review burden                            | Number and proportion of selected components/pages requiring P2-or-higher review intervention.                                                                     |
| Severity-adjusted utility frontier       | Trade-off between extra true print, false annotation, perceptual harm, page coverage and source-group stability across candidate thresholds and envelope variants. |
| Envelope relaxation limit                | The least conservative predeclared D3–D8 variant after which P3/P4 harm, unstable source behaviour or negative severity-adjusted utility first appears.            |

## 3.3 Scope

- Development evidence only; no untouched independent cohort is used to tune the rubric or envelope.
- Read-only use of Notebook 37 outputs and the exact source V8.2 run.
- No model training, no probability recalibration, no label reinterpretation and no post-hoc edits to candidate pages.
- No change to the authoritative corrected mask, dilation, V7.8 reconstruction/routing, source-group folds or truth firewall.
- No automatic promotion. A result may only nominate a revised envelope for later pre-registration and fresh validation.

# 4. Non-Negotiable Controls

| **Control**               | **Required behaviour**                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| D1 integrity              | Remains a hard prerequisite. Any schema, roster, hash, duplication, non-finite-data or replay failure invalidates the analysis.            |
| D2 immutability           | Remains a hard prerequisite. Outside-mask drift and non-PRINT regression from V7.8 must be zero.                                           |
| Authoritative mask        | The reviewed corrected mask remains the sole intervention boundary.                                                                        |
| V7.8 fallback             | Every candidate page equals V7.8 except at final PRINT coordinates, where the exact observed byte is copied.                               |
| Truth firewall            | Clean pages and truth-derived maps are unavailable during inference and initial blinded perceptual review.                                 |
| Source-group OOF          | Every page retains the OOF prediction created by the model that excluded its source group.                                                 |
| Original metrics retained | Registered D3–D8 results and truth-defined catastrophic flags remain unchanged and are reported beside every proposed perceptual variant.  |
| No threshold adoption     | Thresholds are analysis strata. Notebook 38 cannot authorise a lower threshold.                                                            |
| No silent exclusion       | Every missing, unreadable, unmatched or unreviewable case is written with a reason and remains in denominator reporting where appropriate. |

# 5. Experimental Design and Stages

## 5.1 Design overview

Notebook 38 is a two-phase, source-group-aware development experiment. Phase A calibrates a perceptual rubric and severity mapping using a locked calibration subset. Phase B applies that frozen mapping to a held-out audit reserve. The reserve is for development confirmation only; it is not the untouched independent promotion cohort.

| **Stage**                     | **Action**                                                                                                                                          | **Output**                                        |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 0\. Freeze protocol           | Write protocol identity, source run hashes, thresholds, rubric, assignment seed, file inventory and planned analyses before review.                 | protocol_lock.json                                |
| 1\. Resolve inputs            | Verify Notebook 37 and source V8.2 identities, hashes, page roster, threshold list and whole-page integrity.                                        | input_resolution_audit.csv                        |
| 2\. Build unique review units | Collapse repeated cumulative appearances into stable page/component/transition review units while preserving threshold lineage.                     | review_unit_manifest.csv                          |
| 3\. Allocate groups           | Assign entire source groups to rubric-calibration or audit-reserve strata. No source group appears in both.                                         | review_split_map.csv                              |
| 4\. Generate blinded material | Create randomised A/B whole-page and local/wide panels without threshold, truth colour or clean image in the initial view.                          | blinded_review_manifest.csv and image panels      |
| 5\. First-pass review         | Record visibility, preference, harm severity, benefit severity, musical-function risk and confidence.                                               | perceptual_review_round1.csv                      |
| 6\. Diagnostic review         | Reveal observed page, locator and then clean truth/overlays only after first-pass fields are locked. Record mechanism, not a replacement judgement. | diagnostic_review_round2.csv                      |
| 7\. Adjudicate                | Resolve reviewer disagreement without access to candidate threshold during first adjudication.                                                      | perceptual_adjudication.csv                       |
| 8\. Freeze mapping            | Freeze severity definitions and weighting/sensitivity matrix using calibration groups only.                                                         | severity_mapping_lock.json                        |
| 9\. Apply to audit reserve    | Calculate registered and perceptual D3–D8 variants on untouched audit-reserve groups.                                                               | audit_reserve_envelope_results.csv                |
| 10\. Frontier analysis        | Report Pareto frontiers and the first relaxation at which material harm or instability appears.                                                     | envelope_relaxation_frontier.csv                  |
| 11\. Decision record          | Choose retain/current, revise-and-retest, or reject-relaxation.                                                                                     | notebook38_verdict.json and executive_summary.txt |
| 12\. Final package            | Create curated upload folder, file list, SHA-256 manifest and ZIP.                                                                                  | notebook38_upload_bundle.zip                      |

## 5.2 Review-unit construction

- Primary unit: one stable authoritative mask component on one page at one threshold transition.
- Each unit preserves page, source group, component ID, higher threshold, lower threshold, first-appearance threshold, cumulative and newly added pixel counts, and original truth-defined catastrophic status.
- Repeated appearances of the same component at lower thresholds remain linked. Reviewers judge the transition and the cumulative state separately when both are needed.
- Whole-page context is mandatory for every P2–P4 rating and for every case initially judged "unclear".
- Negative controls include representative no-visible-difference cases and exact-equality pairs; positive controls include known visually beneficial structural completions. Control identity is hidden during review.

# 6. Perceptual Severity and Benefit Rubrics

Perceptual harm and benefit are scored separately. A component may preserve genuine notation and retain annotation at the same time. The reviewer must not compress that mixed outcome into one preference score.

## 6.1 Harm severity rubric (P0–P4)

| **Code** | **Name**                       | **Operational definition**                                                                                                                                                                                      | **Required reviewer action**                                                           |
| -------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| P0       | No perceptible effect          | No visible difference at standard whole-page inspection and no credible local difference after locator-assisted inspection.                                                                                     | Accept as perceptually null; retain truth metrics unchanged.                           |
| P1       | Perceptible but negligible     | A difference is visible after inspection but does not distract, obscure notation, resemble a musical symbol, or require correction for normal use.                                                              | Record mechanism; no page correction required.                                         |
| P2       | Reviewer-relevant local defect | Visible retained annotation or local clutter that a reviewer would normally remove, but the musical content remains readily legible and no plausible musical reading changes.                                   | Manual correction/review required; count towards review burden.                        |
| P3       | Major perceptual defect        | Prominent or repeated retained annotation, or local interference with staff/symbol/text reading, that materially reduces page quality or creates plausible ambiguity even if a competent reader may resolve it. | Fails unattended-use interpretation; escalate for adjudication and mechanism analysis. |
| P4       | Operationally catastrophic     | The change plausibly alters, creates, removes, joins, splits or obscures musically meaningful notation/text, or creates a repeated page-level failure likely to mislead performance or transcription.           | Hard stop for any proposed relaxed envelope; preserve as a critical regression case.   |

## 6.2 Harm decision anchors

- Visibility is judged first at the same whole-page scale and contrast. Zoom is used only after that first response is recorded.
- P1 versus P2 turns on whether a normal quality-control reviewer would spend time correcting the defect.
- P2 versus P3 turns on prominence, repetition and plausible ambiguity in reading staff, notehead, stem, beam, accidental, rest, articulation, dynamic, lyric, rehearsal mark, tempo or other printed instruction.
- P3 versus P4 turns on whether the defect can plausibly change musical meaning or create a repeated/systematic failure mechanism. Mere pixel count is not sufficient.
- If the reviewer cannot decide without clean truth, the initial rating is U (unclear). The diagnostic pass then records a resolved rating plus the reason; the initial U remains preserved.

## 6.3 Benefit rubric (B0–B4)

| **Code** | **Name**                  | **Definition**                                                                                                           |
| -------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| B0       | No perceptible benefit    | No visible recovery versus V7.8.                                                                                         |
| B1       | Trace benefit             | Visible only after locator-assisted inspection; no practical reading improvement.                                        |
| B2       | Local legibility benefit  | One local printed feature is clearer or more complete, but the page remains usable with V7.8.                            |
| B3       | Material notation benefit | A musically meaningful printed feature is visibly restored or made substantially easier to read.                         |
| B4       | Critical recovery         | The challenger prevents a plausible wrong reading or restores a feature whose absence materially changes interpretation. |

## 6.4 Additional structured fields

| **Field**                | **Allowed values**                                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| initial_visibility       | no difference \| visible difference \| unclear                                                                                                      |
| pair_preference          | A \| B \| neither \| mixed \| unclear                                                                                                               |
| harm_severity            | P0 \| P1 \| P2 \| P3 \| P4 \| U                                                                                                                     |
| benefit_severity         | B0 \| B1 \| B2 \| B3 \| B4 \| U                                                                                                                     |
| defect_scope             | pixel-fragment \| component-local \| staff/system \| page-repeated \| page-global                                                                   |
| musical_function_at_risk | none \| staff \| notehead \| stem/beam \| accidental \| rest \| articulation \| dynamic \| text/lyric \| barline/repeat \| other \| unclear         |
| annotation_prominence    | none \| faint \| moderate \| prominent                                                                                                              |
| reviewer_action          | accept \| accept-with-note \| manual-correct \| reject-policy-case \| unclear                                                                       |
| confidence               | low \| medium \| high                                                                                                                               |
| diagnostic_mechanism     | correct print recovery \| retained annotation \| mixed overlap \| registration ambiguity \| truth-boundary sensitivity \| display artefact \| other |

# 7. Review Presentation, Blinding and Adjudication

## 7.1 First-pass blinded presentation

- Present two corrected outputs at identical whole-page scale, crop, orientation, greyscale and contrast.
- Randomise left/right placement deterministically from the protocol seed. Conceal threshold, policy ID, truth counts, source group and whether one side is V7.8.
- Record no-difference/visible/unclear, pair preference, P severity, B severity and confidence before any locator, observed input or clean truth is shown.
- Do not use difference maps as primary quality evidence. They are locators only.
- For threshold-transition units, compare the immediately higher threshold with the lower threshold that first introduces the reviewed change.

## 7.2 Context and diagnostic sequence

- After first-pass lock, show a wide contextual panel containing the complete staff/system around the component.
- If still unresolved, show the local crop and a hollow-box whole-page locator.
- Only after the perceptual judgement is locked, reveal the observed page, clean diagnostic and truth overlay for mechanism assignment.
- Diagnostic evidence may explain a judgement but must not overwrite the original perceptual response. Any changed rating is stored as a separate adjudicated field with a reason.

## 7.3 Reviewer design

Proposed study rule: use at least two independent reviewers for every P2–P4 candidate and every original truth-defined catastrophic component. One reviewer may complete P0/P1 screening, but a deterministic audit sample must receive a second review. This is a proposed Notebook 38 control, not an existing project fact.

- Reviewers must be blind to threshold and truth during first-pass scoring.
- If ratings differ by more than one severity level, or one reviewer assigns P3/P4, adjudication is mandatory.
- Adjudication uses the same whole-page-first sequence and records the rationale in free text.
- Report raw agreement, exact agreement, within-one-level agreement and the full disagreement matrix. Do not collapse disagreement into a single average score.
- If one reviewer is available, label the result "single-reviewer exploratory" and do not nominate a revised envelope for independent testing.

# 8. D3–D8 Envelope-Relaxation Analysis

All original gate outcomes must be calculated and retained. The relaxation analysis adds parallel variants; it does not rewrite the historical specification.

## 8.1 D3 — Catastrophic safety

| **Variant** | **Definition**                                                                                                  | **Interpretation**                                                            |
| ----------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| D3-R0       | Registered rule: zero truth-defined catastrophic components pooled and by source group.                         | Control; unchanged historical gate.                                           |
| D3-R1       | Truth-defined catastrophes are permitted only when adjudicated P0 or P1; zero P2–P4 cases.                      | Tests whether the binary truth concentration rule overstates perceptual harm. |
| D3-R2       | Zero P3/P4; P2 cases reported as review burden and must not repeat as a common mechanism within a source group. | Supervised-review research tier only; not unattended use.                     |
| D3-R3       | Any P3 or P4 is a failure. P0–P2 are severity-weighted rather than binary.                                      | Broadest development sensitivity variant; cannot itself authorise promotion.  |

Required conclusion: identify the first threshold and envelope variant at which P3/P4 appears, and whether P2 burden is concentrated by page, source group, musical function or morphology.

## 8.2 D4 — False/true ratio

Calculate the registered point and page-bootstrap upper-bound tests, then a predeclared sensitivity surface. The sensitivity grid is descriptive; it does not become an acceptance limit solely because a cell looks favourable.

| **Dimension**                      | **Values to calculate**                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Observed pooled false/true ceiling | 0.10 (control), 0.15, 0.20, 0.25, 0.35, 0.50                                                           |
| Page-clustered 95% upper ceiling   | 0.20 (control), 0.25, 0.30, 0.40, 0.50, 0.75                                                           |
| Perceptual numerator variants      | All false-annotation pixels; P2–P4 case pixels; P3–P4 case pixels                                      |
| Mandatory stratification           | Page, source group, threshold band, route context, sparse/hard-negative status and perceptual severity |

## 8.3 D5 — Pixel safety

| **Dimension**                       | **Values to calculate**                                                   |
| ----------------------------------- | ------------------------------------------------------------------------- |
| Observed false-preservation ceiling | 0.50% (control), 1%, 2%, 5%, 10%                                          |
| Wilson/page-bootstrap upper ceiling | 1% (control), 2%, 5%, 10%, 15%                                            |
| Perceptual variants                 | All truth-false pixels; pixels in P2–P4 units; pixels in P3–P4 units      |
| Stop condition                      | Any P4, repeated P3 mechanism, outside-mask drift or non-PRINT regression |

The broad grid is an analysis grid, not a proposed production limit. It exists to show where perceptual harm accelerates relative to recovered print.

## 8.4 D6 — Positive utility

Retain the registered utility and add a severity-adjusted family. Do not choose a favourable weight after viewing results; report the complete predeclared matrix.

```
Registered: U = TP − 5×FP − 500×C

Severity family: U(w, λ) = TP − w×FP − Σ λ(P_i)
where w ∈ {1, 2, 3, 5, 10}; λ(P0)=0; λ(P1)∈{0,1}; λ(P2)∈{5,10,25}; λ(P3)∈{100,250,500}; P4 is a hard fail.
```

- Report the point estimate and page-clustered bootstrap interval for every predeclared utility variant.
- Report a benefit-only frontier and a review-burden frontier so utility weights do not conceal the raw trade-off.
- A revised envelope cannot be nominated if its audit-reserve utility lower bound is non-positive under the registered utility and all reasonable perceptual-weight variants.

## 8.5 D7 — Non-trivial benefit

| **Evidence dimension**   | **Sensitivity values**                                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Extra true PRINT pixels  | 100, 250, 500 (control), 750, 1,000                                                                                     |
| Benefiting pages         | 5, 8, 10 (control), 15                                                                                                  |
| Benefiting source groups | 2, 3 (control), 4, 5                                                                                                    |
| Perceptual qualification | B2+, B3+ and any-benefit variants reported separately                                                                   |
| Concentration check      | No single page or source group may supply a dominant share without explicit disclosure and leave-one-group-out analysis |

D7 is an evidence/benefit gate, not a safety gate. Relaxing D7 may classify a result as promising but underpowered; it cannot compensate for P3/P4 harm or failed D1/D2.

## 8.6 D8 — Source stability

| **Variant** | **Group-level requirement**                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| D8-R0       | Registered: source groups with at least 3 eligible pages have false/true ≤0.25 and no truth-defined catastrophe.       |
| D8-R1       | False/true ≤0.35, zero P3/P4 and non-negative severity-adjusted utility in every eligible group.                       |
| D8-R2       | False/true ≤0.50, zero P4, no repeated P3 mechanism and positive pooled plus leave-one-group-out utility.              |
| D8-R3       | Source-scoped research result: groups meeting the frozen rule are identified separately; no claim of general validity. |

- Report every source group, including groups with fewer than 3 pages, but preserve the distinction between descriptive and gate-eligible groups.
- Perform leave-one-source-group-out recomputation of pooled D3–D7 metrics.
- Flag any envelope whose apparent acceptability disappears when one beneficial group is removed or when one harmful group is included.
- Do not solve instability by silently excluding a source group. A future source-scoped policy requires a separate pre-registered experiment and fresh evidence.

# 9. Statistical and Sensitivity Analysis

- Use page-clustered bootstrap resampling for pooled utility, false/true and false-preservation summaries; preserve the Notebook 37 resample count and seed lineage unless the protocol lock specifies a new deterministic seed.
- Report Wilson 95% upper bounds for binomial safety summaries.
- Treat page and source group as the clustering/stratification units; do not present pixel counts as independent observations.
- Calculate results cumulatively by threshold and incrementally by newly admitted threshold band.
- Report registered, truth-only, perceptual-severity and mixed review-burden views side by side.
- Produce a Pareto frontier of genuine-print benefit versus truth-false pixels, P2+ burden and P3/P4 incidence.
- Run label-sensitivity views for truth-UNCERTAIN pixels without silently converting them to true or false.
- Run reviewer-sensitivity analysis using each reviewer separately and the adjudicated result.
- Run leave-one-page-out and leave-one-source-group-out influence summaries for every nominated relaxation.
- If the audit reserve is too small for a stable interval or source analysis, conclude insufficient evidence rather than relaxing the envelope.

# 10. Gate-Specific File Requirements

| **Gate**             | **Minimum files/evidence**                                                                                                                                                                                   | **Why required**                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| D1 hard prerequisite | experiment_identity.json; source_v82_record.json; protocol_lock.json; input manifests; source-group fold map; notebook code; hash manifest                                                                   | Identity, lineage, schema, leakage, deterministic replay and exact source selection.           |
| D2 hard prerequisite | audit/whole_page_integrity_and_hashes.csv; whole-page baseline/candidate files; authoritative mask identities                                                                                                | Proves zero outside-mask drift, zero non-PRINT regression and exact observed-byte restoration. |
| D3                   | catastrophic_components_all_thresholds.csv; catastrophic_component_first_appearance_inventory.csv; catastrophic visual inventory; local/wide/whole-page panels; completed perceptual review and adjudication | Links the registered concentration rule to P0–P4 perceptual severity.                          |
| D4                   | threshold_frontier_summary.csv; all eligible OOF predictions or sufficient aggregate bootstrap inputs; by-page table; incremental band table                                                                 | Recomputes pooled and clustered false/true ratios and sensitivity ceilings.                    |
| D5                   | Same prediction/aggregate inputs as D4; selected-pixel totals; Wilson denominators; severity-linked review-unit manifest                                                                                     | Recomputes false-preservation rates and perceptual numerator variants.                         |
| D6                   | True/false/uncertain counts; catastrophic lineage; severity mapping lock; review decisions; bootstrap seed/configuration                                                                                     | Calculates registered and severity-adjusted utility families.                                  |
| D7                   | threshold_frontier_by_page.csv; threshold_frontier_by_source_group.csv; representative-page selection; benefit ratings                                                                                       | Counts genuine-print benefit, benefiting pages/groups and perceptual materiality.              |
| D8                   | development score manifest; fold map; by-source-group table; review decisions with source group; leave-one-group-out outputs                                                                                 | Tests group stability, concentration and source-scoped failure mechanisms.                     |

# 11. Input File Contract and Minimum Schemas

## 11.1 Preferred transfer

Preferred: provide the complete Notebook 37 upload ZIP plus the Notebook 37 code and the source V8.2 identity/summary files. The ZIP must preserve relative paths and be checked against upload_file_list.txt and upload_manifest.csv before analysis.

## 11.2 Mandatory Notebook 37 files

- 37_V8_2h_Downward_PRINT_Threshold_Safe_Coverage_Frontier_WHOLE_PAGE_UPLOAD_AMENDED.ipynb.txt
- executive_summary.txt
- experiment_identity.json
- source_v82_record.json
- diagnostic_frontier_report.json
- upload/README.txt
- upload/upload_file_list.txt
- upload/upload_manifest.csv
- tables/threshold_frontier_summary.csv
- tables/threshold_frontier_by_page.csv
- tables/threshold_frontier_by_source_group.csv
- tables/incremental_threshold_band_summary.csv
- tables/incremental_threshold_pixels.parquet
- tables/incremental_component_review_queue_all.csv
- tables/catastrophic_components_all_thresholds.csv
- tables/catastrophic_component_first_appearance_inventory.csv
- tables/catastrophic_component_visual_inventory.csv
- tables/computable_gate_map.csv
- tables/representative_page_selection.csv
- audit/whole_page_integrity_and_hashes.csv
- audit/whole_page_contact_sheet_inventory.csv
- review/catastrophic_component_visual_review.csv
- review/threshold_increment_review_queue.csv
- review/threshold_increment_crop_inventory.csv
- review/crops/\*\*
- visual/catastrophic_components/\*\* or catastrophic_review/\*\*
- contact_sheets/\*\*
- whole_pages/&lt;page&gt;/v7_8.png
- whole_pages/&lt;page&gt;/v8_2_registered_0995.png
- whole_pages/&lt;page&gt;/p&lt;target&gt;\_threshold_candidate.png
- whole_pages/&lt;page&gt;/p&lt;target&gt;\_difference_from_registered_0995.png

## 11.3 Required source V8.2 files

- inputs/development_score_manifest.csv
- fold_map.csv
- channel_schema.json
- tables/all_eligible_oof_predictions.parquet
- tables/results_by_page.csv
- tables/results_by_source_group.csv
- tables/selected_components.csv and/or catastrophic_components.csv
- audit/source_group_split_audit.csv or equivalent no-overlap proof
- whole_pages/&lt;page&gt;/observed.png, v7_8.png and v8_2.png
- experiment_identity.json
- executive_summary.txt

## 11.4 Minimum review-unit schema

| **Column**                                                 | **Required meaning**                                                                        |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| review_unit_id                                             | Stable opaque identifier; no threshold information exposed to reviewer.                     |
| page                                                       | Page identity for analysis; hidden in review UI if it reveals source.                       |
| source_group                                               | Source-group identity for split and stability analysis; hidden in first-pass review.        |
| mask_component_id                                          | Stable authoritative component identifier.                                                  |
| higher_threshold / lower_threshold                         | Adjacent transition that created the reviewed change.                                       |
| first_appearance_threshold                                 | First threshold at which the cumulative registered catastrophic rule is met, if applicable. |
| selected / true_print / false_annotation / truth_uncertain | Frozen truth-scored counts.                                                                 |
| registered_catastrophic                                    | Original D3 flag retained unchanged.                                                        |
| whole_page_A / whole_page_B                                | Randomised corrected-page pair.                                                             |
| local_panel / wide_panel / locator                         | Review artefact paths.                                                                      |
| assignment_split                                           | rubric_calibration or audit_reserve.                                                        |
| randomisation_side                                         | Encrypted/locked A/B mapping, unavailable to reviewer.                                      |
| sha256_\*                                                  | Hash for every review image and source table row set.                                       |

## 11.5 Minimum perceptual-review schema

| **Column**               | **Requirement**                                                  |
| ------------------------ | ---------------------------------------------------------------- |
| review_unit_id           | Joins to locked review unit.                                     |
| reviewer_id              | Pseudonymous stable reviewer identity.                           |
| round                    | blind_whole_page \| blind_context \| diagnostic \| adjudication. |
| initial_visibility       | Recorded before locators or truth.                               |
| pair_preference          | A/B/neither/mixed/unclear.                                       |
| harm_severity            | P0–P4 or U.                                                      |
| benefit_severity         | B0–B4 or U.                                                      |
| defect_scope             | Structured field from rubric.                                    |
| musical_function_at_risk | Structured field from rubric.                                    |
| reviewer_action          | accept / note / manual-correct / reject / unclear.               |
| confidence               | low / medium / high.                                             |
| review_note              | Short rationale; mandatory for P2–P4, B3–B4 and U.               |
| timestamp_utc            | Review record time.                                              |
| rubric_version           | Exact frozen rubric identity.                                    |
| locked_before_diagnostic | Boolean; must be true for valid first-pass record.               |

# 12. Required Outputs and Final Upload Package

| **Directory/file**                        | **Required content**                                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| protocol_lock.json                        | Version, source hashes, rubric, split seed, sensitivity grids, planned analyses and change-control status. |
| tables/review_unit_manifest.csv           | Every review unit and lineage.                                                                             |
| tables/review_split_map.csv               | Source-group-separated calibration/audit allocation.                                                       |
| review/perceptual_review_round1.csv       | Locked blinded responses.                                                                                  |
| review/diagnostic_review_round2.csv       | Mechanism assignment after reveal.                                                                         |
| review/perceptual_adjudication.csv        | Resolved ratings and reasons.                                                                              |
| tables/reviewer_agreement.csv             | Agreement and disagreement matrix.                                                                         |
| tables/registered_D3_D8_results.csv       | Unchanged control results.                                                                                 |
| tables/perceptual_D3_D8_results.csv       | Every predeclared relaxation variant.                                                                      |
| tables/envelope_relaxation_frontier.csv   | Ordered frontier and first material-harm points.                                                           |
| tables/results_by_page.csv                | Registered and perceptual metrics by page.                                                                 |
| tables/results_by_source_group.csv        | Registered and perceptual metrics by source group.                                                         |
| tables/leave_one_group_out.csv            | Influence and stability analysis.                                                                          |
| tables/review_burden_summary.csv          | P2+ cases and manual-action burden.                                                                        |
| visuals/perceptual_frontier.png           | Benefit versus truth error and perceptual harm.                                                            |
| visuals/severity_by_threshold_band.png    | P0–P4 distribution by newly admitted band.                                                                 |
| visuals/source_group_stability.png        | Source-group trade-off view.                                                                               |
| audit/input_resolution_audit.csv          | Required file, hash and schema checks.                                                                     |
| audit/whole_page_integrity_and_hashes.csv | D2 and page identity evidence.                                                                             |
| notebook38_verdict.json                   | Machine-readable outcome and first failed principle.                                                       |
| executive_summary.txt                     | Plain-language result, limitations and next action.                                                        |
| decision_record.md or .docx               | Written project conclusion.                                                                                |
| upload/README.txt                         | Review order and interpretation boundary.                                                                  |
| upload/upload_file_list.txt               | Authoritative relative-path list.                                                                          |
| upload/upload_manifest.csv                | Byte size, SHA-256 and purpose.                                                                            |
| notebook38_upload_bundle.zip              | ZIP mirror of the curated upload folder.                                                                   |

## 12.1 Standard package layout

```
upload/
├── README.txt
├── upload_file_list.txt
├── upload_manifest.csv
├── executive_summary.txt
├── notebook38_verdict.json
├── tables/
├── review/
├── visuals/
├── audit/
└── reference/

notebook38_upload_bundle.zip
```

# 13. Fail-Closed Rules, Deviations and Change Control

- Fail before interpretation if any mandatory source identity, hash, roster, schema or whole-page integrity check fails.
- Fail if A/B randomisation can be inferred from filenames, labels or image annotations.
- Fail the perceptual study if diagnostic truth is exposed before the first-pass record is locked.
- Do not impute missing reviews or silently treat U as P0/P1.
- Do not alter P0–P4 definitions, sensitivity grids, utility weights, split assignments or decision rules after audit-reserve outcomes are inspected.
- Do not use the audit reserve to tune the rubric; a material rubric change requires a new split/version.
- Do not use Notebook 38 to alter model probabilities, calibration or the underlying PRINT decisions.
- Do not call a development relaxation "safe" without a fresh, pre-registered validation stage.
- Write failure artefacts and the incomplete status before raising an exception.
- Never overwrite Notebook 37, the source V8.2 run or an earlier Notebook 38 run.

## 13.1 Permitted pre-review amendments

Before any blinded review begins, a dated version increment may clarify filenames, fix an implementation error, add a missing mandatory output or narrow the review roster for a documented integrity reason. Preserve the previous protocol and its hash.

## 13.2 Post-review changes

After first-pass review begins, changes to rubric, weights, gates, split logic or candidate thresholds create a new experiment version. Editorial corrections that cannot influence outcomes may be documented without rerunning, but must be listed in the deviation register.

# 14. Decision Logic and Permissible Conclusions

| **Outcome**                                                                     | **Permissible conclusion**                                                         | **Next action**                                                                                                 |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| D1 or D2 failure                                                                | Implementation/data failure; no envelope conclusion.                               | Correct the implementation, create a new run and preserve the failed artefacts.                                 |
| P4 anywhere in a candidate relaxation                                           | The relaxation is not acceptable.                                                  | Retain the stricter boundary; diagnose the mechanism.                                                           |
| Repeated P3 mechanism or source concentration                                   | The relaxation is unstable or operationally unsafe.                                | Reject general relaxation; consider a new representation or separately pre-registered source-scoped hypothesis. |
| Only P0/P1 among original truth catastrophes, with stable audit-reserve metrics | The registered binary D3 definition may be over-conservative for those mechanisms. | Draft a revised D3 proposal; do not promote until fresh validation.                                             |
| P2 cases but no P3/P4                                                           | Potential supervised-review tier with measurable burden.                           | Report burden and decide whether the recovered print justifies review; no unattended-use claim.                 |
| D4/D5 relaxation improves benefit but intervals remain wide                     | Insufficient precision, not a safety pass.                                         | Acquire fresh page/source groups rather than moving limits post hoc.                                            |
| D7 alone fails while D3–D6/D8 are acceptable                                    | Potential low-coverage or underpowered result.                                     | Retain as research; gather evidence or improve discrimination.                                                  |
| Aggregate acceptable but D8 unstable                                            | No general policy supported.                                                       | Retain V7.8 generally; investigate the failure mechanism with a new pre-registered design.                      |
| One revised envelope is stable in calibration and audit reserve                 | Development nomination only.                                                       | Freeze the definition as a new challenger protocol and test on fresh untouched evidence.                        |
| No relaxation survives                                                          | Current envelope remains the development standard.                                 | Pursue better model evidence/context rather than weaker gates.                                                  |

# 15. Implementation Handover for a Separate Copilot Chat

**The next chat should treat this document as the protocol authority and the Notebook 37 package as the empirical source. It should not redesign the experiment unless a gap is explicitly recorded as a protocol amendment.**

## 15.1 Opening instruction for the next chat

```
Build Notebook 38 exactly from "Notebook38_Envelope_Relaxation_Project_Reference_FULL_DEFINITION_v1_0.docx".
Use the Notebook 37 upload bundle and source V8.2 run only.
Do not retrain, recalibrate, change labels, alter thresholds, modify V7.8, or expose clean truth during first-pass review.
First produce an input-resolution report and stop if mandatory files or hashes are missing.
Then implement source-group-separated rubric calibration and audit reserve, blinded A/B review, P0–P4/B0–B4 adjudication, registered plus perceptual D3–D8 calculations, and the standard final upload bundle.
```

## 15.2 Handover sequence

1. Read this full definition and extract every frozen constant, file name, schema and stop rule into protocol_lock.json.
2. Inventory the uploaded Notebook 37 bundle; verify upload_file_list.txt and upload_manifest.csv; report missing/extra/hash-mismatched files.
3. Resolve the exact source V8.2 run from Notebook 37 identity rather than selecting the latest run.
4. Confirm original threshold order, comparator, page roster, source groups and component IDs.
5. Build review_unit_manifest.csv and show counts by threshold transition, truth category, source group and registered catastrophic status.
6. Create source-group-separated calibration and audit-reserve assignments using a recorded deterministic seed.
7. Generate blinded whole-page-first review material with deterministic A/B randomisation.
8. Implement the exact P0–P4 and B0–B4 fields and prevent diagnostic reveal until first-pass lock.
9. Export completed review records and reviewer agreement; stop if mandatory review coverage is incomplete.
10. Freeze severity_mapping_lock.json using calibration groups only.
11. Compute all registered and perceptual D3–D8 variants on the audit reserve without changing the mapping.
12. Produce frontier, source-stability, review-burden and influence outputs.
13. Write notebook38_verdict.json using the decision table in section 14.
14. Build the final upload package and append its location/count/hash to executive_summary.txt.

## 15.3 Questions the implementing chat must answer before code generation

- Do all mandatory Notebook 37 tables and review images exist and match their manifest hashes?
- Is the full source-group roster available to create a non-overlapping audit reserve?
- Are completed human reviews already present, or must Notebook 38 provide the review interface?
- How many independent reviewers are available? If only one, the run must be labelled exploratory.
- Are sparse/hard-negative flags available in the manifest? If not, D9 remains out of scope and the omission must be explicit.
- Can each whole-page candidate be deterministically reconstructed from frozen inputs if a packaged image is absent?
- Which fresh untouched cohort, if any, will be reserved for a later validation of a nominated revised envelope? This must not be used in Notebook 38 development.

## 15.4 Minimum completion checklist

**☐** Protocol lock written before review

**☐** Input hashes and schemas verified

**☐** No source-group overlap between calibration and audit reserve

**☐** Blinded A/B mapping locked

**☐** Whole-page-first review complete

**☐** All original catastrophes reviewed

**☐** All P2–P4 cases independently reviewed/adjudicated

**☐** Original D3–D8 reproduced exactly

**☐** All predeclared relaxation variants calculated

**☐** Leave-one-group-out stability complete

**☐** Verdict and limitations written

**☐** Upload file list, manifest, hashes and ZIP verified

# 16. Source Basis and Assumptions

## 16.1 Established project constraints used

| **Project reference**                                                         | **Use in this document**                                                                                                                                                     |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx | Definitions of D3–D8, truth metrics, clustered uncertainty, frozen V7.8 comparator, authoritative mask, development/independent separation and fail-closed posture.          |
| V8_1_Full_Pixel_Whole_Page_OOF_Notebook_Specification.docx                    | Registered metrics, catastrophic rule, whole-page assembly, original D1–D12 interpretation and required source-/page-level reporting.                                        |
| V8_Next_Stage_Full_Pixel_Whole_Page_OOF_Reference.docx                        | Whole-page-first perceptual review, identical-scale presentation, difference maps as locators, and primary questions about preserved notation versus retained annotation.    |
| V8_2_Development_Whole_Page_Experiment_Decision_Record.docx                   | Development-first stage separation, durable outputs, integrity controls and standard final upload-package requirement.                                                       |
| V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx           | Controlled-comparison principle, safe-coverage frontier, source-group OOF and warning against combining multiple causal changes.                                             |
| V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx           | Separation of annotation confidence and correction safety, authoritative-mask boundary and exact preservation principle.                                                     |
| V7_9_x_Research_Findings_and_Next_Steps.docx                                  | Evidence that recoverable print exists but broad positive-preservation rules can be insufficiently selective; warning against further threshold tuning on the same evidence. |
| Notebook 37 V8.2h threshold-frontier outputs                                  | Empirical trigger, threshold transitions, truth-defined catastrophes, review artefacts and source tables used by Notebook 38.                                                |

## 16.2 New protocol choices introduced here

The P0–P4 harm rubric, B0–B4 benefit rubric, calibration/audit-reserve split, reviewer/adjudication rules, D3 relaxation variants, D4/D5 sensitivity grids, severity-adjusted utility matrix and D7/D8 sensitivity variants are proposed Notebook 38 design choices. They are not prior project facts. They must be locked before the audit reserve is reviewed and require fresh validation before any release decision.

## 16.3 Confidence and unresolved risk

Protocol-design confidence: 91%. The experiment boundary, frozen evidence lineage, D3–D8 definitions, whole-page review contract and provenance requirements are strongly grounded in the project records. Confidence is lower for the exact perceptual severity weights and relaxation grids because they are new research choices and have not yet been calibrated against independent reviewers or fresh source groups. The document therefore treats them as a predeclared sensitivity family rather than proven safety limits.

**Next question: Does the available Notebook 37 package contain enough source-group diversity and completed blinded review evidence to support a held-out audit reserve, or must Notebook 38 first be limited to rubric calibration?**