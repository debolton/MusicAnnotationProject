**MUSIC ANNOTATION PROJECT**

Residual-Mask Improvement Programme

Detailed design for Notebooks 49, 50 and 51

_Development-only research specification | 28 October 2026_

**Programme decision** Separate diagnosis, learning and causal reconstruction testing. Notebook 49 must establish whether the M3-to-M4 oracle gap contains usable observable signal. Notebook 50 may build one fail-closed M5 challenger only if Notebook 49 supports that conclusion. Notebook 51 then crosses fixed M3, M5 and oracle M4 masks with frozen restorers to test whether improved mask conditioning, rather than restorer tuning, causes the gain.

**Status: project reference and implementation specification  
Not production authorisation**

# 1. Executive decision

The next research sequence should improve the residual mask by making the reconstruction target semantically cleaner and its boundaries more informative. The programme must not begin with another fixed dilation, collar radius or hand-built threshold. It must first determine what the current M3 mask contains, then learn only where observable evidence is demonstrably informative, and finally test the resulting M5 mask with unchanged restorers.

| **Notebook** | **Role**                            | **Permitted change**                                                            | **Decision produced**                                                      |
| ------------ | ----------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 49           | Read-only oracle-gap decomposition  | No model fitting; no threshold tuning; no restorer changes                      | Whether observable evidence justifies a learned residual-mask challenger   |
| 50           | Observable residual-mask challenger | One pre-registered M5 policy, trained source-group OOF and failing closed to M3 | Whether M5 improves mask quality and is fit for frozen crossover           |
| 51           | Frozen mask-restorer crossover      | No M5 retraining; no restorer retuning                                          | Whether mask conditioning improves reconstruction across several restorers |

**Central hypothesis** As the residual mask becomes semantically purer and its boundary context becomes more trustworthy, reconstruction error should improve across multiple frozen restorers and sensitivity to the choice of restorer should contract.

# 2. Programme boundaries and inherited contracts

- Development-only. None of Notebooks 49-51 may authorise production promotion or alter the V7.8 incumbent.
- The authoritative corrected mask remains the sole boundary of permissible intervention.
- Clean reference data is prohibited from mask inference and reconstruction. It may be loaded only after observable artefacts and decisions have been written and hashed.
- Pixels outside the authoritative mask remain byte-for-byte immutable. Protected observed PRINT bytes also remain exact.
- Source-group separation is mandatory for any fitted model and for all reported stability results.
- M3 remains the fail-closed fallback. An uncertain or invalid M5 decision must reproduce M3 treatment exactly.
- The mask-building question and the restorer question remain separate. Notebook 50 changes only the mask policy; Notebook 51 freezes both masks and restorers before comparison.
- Every notebook concludes with the standard curated Final Upload Package: summary, decision tables, review images, audit and identity records, without indiscriminately copying large caches.

# 3. Conceptual residual-mask model

Notebook 45 currently produces observable arms M0-M3 and a post-lock oracle M4. The proposed programme interprets the M3 residual not as one homogeneous inpainting region, but as a mixture of four operational classes:

| **Class**   | **Meaning**                                                                                             | **Runtime treatment**                                                                      | **Primary risk**                                             |
| ----------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| PROTECT     | An observed pixel carries genuine printed information that should not be surrendered to reconstruction. | Copy the observed byte exactly and remove the coordinate from the residual mask.           | False protection can retain annotation.                      |
| REMOVE      | The coordinate is annotation/paper material that does not require reconstruction.                       | Apply the frozen removal/paper treatment and remove the coordinate from the residual mask. | False removal can erase print.                               |
| RECONSTRUCT | The coordinate represents genuinely unsupported printed content or a gap requiring estimation.          | Keep the coordinate in the residual mask.                                                  | An impure target makes the restorer invent or smear content. |
| UNCERTAIN   | Observable evidence is insufficient, inconsistent or out of distribution.                               | Retain M3 treatment exactly.                                                               | Excess uncertainty limits benefit but protects safety.       |

**Important distinction** M4 is a diagnostic oracle, not a deployable mask. Notebook 49 may use clean truth to describe the M3-to-M4 gap after the lock; Notebook 50 must predict only from observable inputs available at runtime.

# 4. Notebook 49 - Residual-Mask Oracle-Gap Decomposition

| **Field**          | **Specification**                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Proposed filename  | 49_V8_7_Residual_Mask_Oracle_Gap_Decomposition.ipynb                                                                        |
| Runtime            | CPU; GPU unnecessary                                                                                                        |
| Data role          | Read-only development diagnostic                                                                                            |
| Primary input      | Completed Notebook 45 mask run, Notebook 46 crossover identity, V8.2/V8.4 observable evidence and exact page/source records |
| Primary output     | Locked pixel/component decomposition of M3 into PROTECT, REMOVE, RECONSTRUCT and UNCERTAIN truth-side classes               |
| Learning authority | None                                                                                                                        |

## 4.1 Scientific question

Does the M3-to-M4 oracle gap contain enough stable, source-general observable signal to justify a learned M5 residual-mask challenger? The notebook is successful only if it can show that mask errors are structured and predictably associated with observable evidence. Merely showing that M3 differs from M4 is insufficient.

## 4.2 Required inputs and integrity checks

- Notebook 45 mask_lock.json, pre-registration, source_record.json, region_manifest.parquet, M0-M4 masks, mask_readiness_metrics.csv and oracle_gap_summary.csv.
- Exact observed page, V7.8 baseline, authorised mask, protected coordinates, route/support maps and source-group identity used by Notebook 45.
- Notebook 46 experiment identity and mask semantic hashes, used only to prove that the masks reviewed here are the masks crossed with the restorers.
- Clean reference pages loaded only after the complete observable manifest, region windows, predictor arrays and hashes have been written.
- Fail-closed checks for page roster identity, shape equality, unique page/source mapping, finite arrays, coordinate validity and zero pixels outside the authorised mask.

## 4.3 Two-stage lock

1. Build the complete observable candidate population from M3 without loading clean truth.
2. Write an ordered pixel manifest and a region/component manifest with hashes of all observable arrays.
3. Write candidate_lock.json containing semantic array hashes, predictor schema, page/source roster and region identity.
4. Reload the lock and verify exact replay.
5. Only then load clean truth and assign diagnostic classes and outcomes.

## 4.4 Unit of analysis

Notebook 49 should export both pixel-level and connected-region-level records. Pixel evidence is needed because broad mask components can contain print, annotation and blank material simultaneously. Region-level records remain necessary for clustered uncertainty, topology and practical review.

| **Level**                 | **Required identity**                                               | **Use**                                                               |
| ------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Pixel                     | page, source_group, y, x, M3 membership, authorised-mask membership | Class decomposition, feature distributions, future supervised samples |
| Connected residual region | stable region_id based on page and M3 connected-component identity  | Boundary, topology, composition and review unit                       |
| Page                      | page and source_group                                               | Clustered uncertainty, failure concentration and visual audit         |
| Source group              | frozen development source identity                                  | Generalisation and concentration analysis                             |

## 4.5 Truth-side class construction

| **Diagnostic class** | **Post-lock definition**                                                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PROTECT              | Observed material inside M3 corresponds to genuine printed ink in the registered clean reference and can be retained as an exact observed byte under the frozen tolerance rule. |
| REMOVE               | M3 material is annotation/non-print or paper whose correct treatment is removal/background restoration rather than structural reconstruction.                                   |
| RECONSTRUCT          | Clean print exists but is unsupported by trustworthy observed bytes; the location genuinely requires estimation from surrounding context.                                       |
| UNCERTAIN            | Registration ambiguity, mixed truth evidence, excluded borders, unstable local truth or conflicting label-quality checks prevent a reliable diagnostic assignment.              |

The exact label rule must be frozen in the Notebook 49 pre-registration. Sensitivity variants may be calculated, but the primary class definition must not be selected after inspecting separability results.

## 4.6 Observable feature families

| **Feature family**     | **Examples**                                                                                                              | **Constraint**                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Observed appearance    | raw greyscale; background-normalised greyscale; fixed gradients/edges; local contrast                                     | No clean-derived channel.                                 |
| Mask geometry          | M3 indicator; signed/unsigned boundary distance; mask depth; component size/aspect/fill; distance to protected pixels     | Derived only from authorised and locked observable masks. |
| Frozen routing/support | preserve/remove/ambiguous context; horizontal, vertical and compact support; local route fractions                        | Exact frozen V7.8 semantics and implementation identity.  |
| V8 evidence            | p_print, PRINT/ANNOTATION/UNCERTAIN decision, V8.4 safety/value scores where present                                      | OOF values only; no fitted full-development leakage.      |
| Boundary conditioning  | trusted-print contact; paper/removal contact; edge support; endpoint count; entering orientation; branch/junction proxies | Definitions frozen before truth-side analysis.            |
| Multi-scale context    | 33x33, 97x97 and, only if already durable, 193x193 observable summaries or patches                                        | Do not conduct architecture search in Notebook 49.        |
| Identity exclusions    | page name, source-group ID, raw x/y coordinates, clean pixels, truth metrics, reconstruction outcomes                     | May be retained as IDs/strata, never predictors.          |

## 4.7 Analyses

- Class prevalence by page, source group, M3 region size, mask depth and route context.
- M3 composition: fraction of pixels and regions that are PROTECT, REMOVE, RECONSTRUCT or UNCERTAIN.
- Oracle-gap decomposition by region, including whether a region contains one class or a harmful mixture.
- Univariate and multivariate feature contrasts, using effect sizes and source-group-stratified summaries rather than pooled significance alone.
- Truth-free separability screening with simple, predeclared read-only probes: for example cross-validated logistic/linear discriminability reported as diagnostic evidence only, not saved as the M5 model.
- Boundary analysis: endpoint compatibility, orientation, trusted-contact rate, maximum unsupported span, branch count and edge continuity.
- Source-group leave-one-group-out stability of any apparent separation.
- Visual queues for largest PROTECT exposure, largest REMOVE contamination, genuine RECONSTRUCT targets, mixed regions and registration/label uncertainty.

## 4.8 Notebook 49 decision rule

| **Verdict**                  | **Meaning**                                                                                                                                              | **Next action**                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| SIGNAL_SUFFICIENT_FOR_M5     | At least one operational class can be separated from RECONSTRUCT using observable evidence with stable page/source behaviour and no evidence of leakage. | Proceed to the fixed Notebook 50 pre-registration.                             |
| SIGNAL_PARTIAL_RESEARCH_ONLY | Observable signal exists but is source-concentrated, class-limited or too unstable for a general M5.                                                     | Design a narrower class- or source-scoped experiment; do not train general M5. |
| MASK_SIGNAL_INSUFFICIENT     | M3-to-M4 differences are substantial but not predictably distinguishable from observables.                                                               | Stop mask learning; reconsider data, labels or reconstruction strategy.        |
| INVALID_RUN                  | Any lock, leakage, identity, truth-firewall, roster or immutability failure.                                                                             | Correct implementation and rerun; do not interpret results.                    |

## 4.9 Required outputs

- 01_pre_registration.json
- source_record.json
- observable_candidate_lock.json
- pixel_oracle_gap.parquet
- region_oracle_gap.parquet
- feature_schema.json
- class_prevalence_by_page.csv
- class_prevalence_by_source_group.csv
- feature_contrasts.csv
- source_group_separability.csv
- boundary_conditioning_analysis.csv
- visual_audit_queue.csv
- visual contact sheets
- notebook49_verdict.json
- experiment_identity.json
- executive_summary.txt
- standard upload bundle and manifest

# 5. Notebook 50 - Observable Residual-Mask Challenger

| **Field**           | **Specification**                                              |
| ------------------- | -------------------------------------------------------------- |
| Proposed filename   | 50_V8_7_Observable_Residual_Mask_M5_Challenger.ipynb           |
| Runtime             | GPU for spatial model; CPU baselines and audit                 |
| Precondition        | Notebook 49 verdict SIGNAL_SUFFICIENT_FOR_M5                   |
| Comparator          | Frozen M3                                                      |
| Treatment           | One pre-registered M5 policy                                   |
| Fallback            | Exact M3 treatment for UNCERTAIN, invalid or unsupported cases |
| Promotion authority | None; development freeze eligibility only                      |

## 5.1 Scientific question

Can an observable-only, source-group OOF policy remove predictable PROTECT and REMOVE contamination from M3 while retaining genuine RECONSTRUCT targets and preserving the established safety and immutability contract?

## 5.2 Experimental design

**Paired causal design** Every eligible pixel/region, fold, label, observable channel, calibration procedure and evaluation rule is shared across the control and treatment. M3 is the control. M5 differs only by the frozen challenger decision layer.

| **Arm**       | **Definition**                                                                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Control M3    | The exact locked Notebook 45 M3 mask and treatment.                                                                                                                           |
| M5 challenger | M3 minus confident PROTECT pixels, minus confident REMOVE pixels and minus any pre-registered structure-aware boundary exclusions; every other coordinate remains exactly M3. |
| Oracle M4     | Diagnostic ceiling only. Never available to M5 inference or reconstruction.                                                                                                   |

## 5.3 Candidate population and target

- Population: coordinates and/or residual subregions inside frozen M3.
- Primary operational target: RECONSTRUCT versus NOT-RECONSTRUCT, with NOT-RECONSTRUCT split into PROTECT and REMOVE for treatment and error attribution.
- UNCERTAIN remains an operational class and defaults to M3.
- If Notebook 49 shows that one of PROTECT or REMOVE is not separable, Notebook 50 must narrow the target before outcomes, rather than pretending to solve all classes.

## 5.4 Model arms

The model comparison should remain deliberately bounded. The objective is not an architecture tournament.

| **Arm**               | **Representation**                                                                                            | **Purpose**                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| B0 rules baseline     | Frozen M3 and any predeclared simple observable exclusion rule                                                | Shows whether learning is necessary.      |
| B1 scalar baseline    | Regularised multinomial logistic regression or calibrated gradient boosting on engineered observable features | Interpretable baseline and leakage audit. |
| T1 spatial challenger | Compact CNN using the fixed primary patch/context design selected before training                             | Tests local topology and ownership.       |
| Optional one ablation | One predeclared additional context scale only if Notebook 49 demonstrates a specific context limitation       | No open-ended search.                     |

## 5.5 Splits, calibration and decision policy

- Outer evaluation grouped by source group where feasible; otherwise by page with explicit source-stratified reporting.
- No page or source group may contribute to both training and scored test data within a fold.
- Preprocessing, model fitting and hyperparameter selection occur inside training folds only.
- Probabilities are calibrated from OOF predictions using a cross-fitted method persisted by fold.
- The operational decision is asymmetric. A pixel is removed from M3 only when its class-specific confidence and safety conditions pass. Otherwise it remains M3.
- Isolated decisions and spatial fragments must obey one frozen connected-run/region-coherence rule applied page-globally, not batch-by-batch.
- OOD, missing, non-finite, schema-mismatched or identity-failed inputs emit UNCERTAIN and retain M3.

## 5.6 M5 construction contract

| **Decision**                          | **M5 action**                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- |
| PROTECT passes frozen rule            | Remove coordinate from reconstruction target and preserve exact observed byte.                  |
| REMOVE passes frozen rule             | Remove coordinate from reconstruction target and apply the frozen removal/background treatment. |
| RECONSTRUCT passes or remains default | Keep coordinate in residual mask.                                                               |
| UNCERTAIN / failed prerequisite       | Keep exact M3 treatment.                                                                        |
| Structure-aware boundary exclusion    | Withhold only under the pre-registered observable rule; record a distinct reason code.          |

## 5.7 Primary errors and metrics

| **Error/metric**            | **Definition**                                                                           | **Priority**                            |
| --------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------- |
| False PROTECT               | Annotation/non-print removed from M3 and copied as observed print                        | Highest safety concern.                 |
| False REMOVE                | Genuine print removed from M3 and routed to removal                                      | Highest structural concern.             |
| False RECONSTRUCT retention | Predictable non-reconstruction material left in M5                                       | Reduces benefit but is fail-closed.     |
| Missed RECONSTRUCT          | A genuinely unsupported print target is excluded from reconstruction                     | Potential structural failure.           |
| Mask purity                 | Share of M5 belonging to truth-side RECONSTRUCT                                          | Primary readiness benefit.              |
| Oracle distance             | Normalised M3-to-M4 versus M5-to-M4 distance on registered mask-quality measures         | Diagnostic, not sole objective.         |
| Boundary conditioning       | Trusted contact, endpoint compatibility, edge continuity, unsupported span and ambiguity | Primary restorer-conditioning evidence. |
| Coverage                    | Fraction of M3 safely re-routed away from reconstruction                                 | Secondary to safety.                    |
| Immutability                | Outside-mask drift and unauthorised byte change                                          | Required zero.                          |

## 5.8 Development gates

| **Gate**                 | **Required result**                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| M1 Integrity             | Zero roster, schema, leakage, non-finite or replay failures.                                                                   |
| M2 Immutability          | Zero outside-authority drift and zero unauthorised protected-byte changes.                                                     |
| M3 Catastrophic safety   | Zero predeclared catastrophic false-PROTECT or false-REMOVE regions pooled and by source group.                                |
| M4 Mask purity           | M5 improves truth-side reconstruction-target purity over M3, with page-clustered uncertainty.                                  |
| M5 Boundary readiness    | M5 improves the predeclared boundary-conditioning composite or its registered constituent metrics without a safety regression. |
| M6 Oracle opportunity    | M5 closes a non-trivial portion of the M3-to-M4 gap across more than one source group.                                         |
| M7 Source stability      | No material benefit or failure concentration hidden by pooled results.                                                         |
| M8 Conservative fallback | All UNCERTAIN/invalid cases reproduce M3 exactly.                                                                              |
| M9 Usable coverage       | Enough M3 pixels/regions are safely re-routed to justify Notebook 51; threshold frozen before definitive evaluation.           |
| M10 Visual audit         | Worst safety errors, largest gains, mixed regions, uncertain regions and all catastrophes reviewed.                            |

## 5.9 Notebook 50 verdicts

| **Verdict**                   | **Meaning**                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| M5_READY_FOR_FROZEN_CROSSOVER | All mandatory integrity/safety gates pass and mask/readiness benefit is stable enough for Notebook 51. |
| M5_LOW_COVERAGE_BASELINE      | Safety passes but benefit/coverage is insufficient. Retain as evidence; do not force thresholds lower. |
| M5_SOURCE_SCOPED_ONLY         | Pooled evidence is promising but source instability prevents general use.                              |
| M5_REJECTED                   | Safety, utility, boundary or stability gate fails.                                                     |
| INVALID_RUN                   | Leakage, lock, replay, roster, identity or immutability failure.                                       |

## 5.10 Required outputs

- 01_pre_registration.json
- dataset and source-group split audits
- frozen feature/channel schema
- OOF pixel/region probabilities and decisions
- fold model and calibrator inventories
- M3 and M5 semantic hashes
- M5 masks and treatment maps
- results_by_page.csv
- results_by_source_group.csv
- mask_purity_and_oracle_gap.csv
- boundary_conditioning_results.csv
- catastrophic_regions.csv
- visual audit queues/contact sheets
- notebook50_verdict.json
- experiment_identity.json
- replay_test.json
- executive_summary.txt
- standard upload bundle and manifest

# 6. Notebook 51 - Frozen M3/M5/M4 Reconstruction Crossover

| **Field**               | **Specification**                                                                |
| ----------------------- | -------------------------------------------------------------------------------- |
| Proposed filename       | 51_V8_7_Frozen_M3_M5_M4_Restorer_Crossover.ipynb                                 |
| Runtime                 | GPU where required by frozen learned restorer; CPU otherwise                     |
| Precondition            | Notebook 50 verdict M5_READY_FOR_FROZEN_CROSSOVER                                |
| Masks                   | Frozen M3 control, frozen M5 challenger, diagnostic oracle M4                    |
| Restorers               | Frozen subset from Notebook 46; no retraining or retuning                        |
| Primary causal question | Does M5 improve reconstruction across restorers and reduce restorer sensitivity? |

## 6.1 Experimental matrix

| **Mask arm**      | **Biharmonic** | **Telea** | **Learned U-Net** | **Optional conservative comparator** |
| ----------------- | -------------- | --------- | ----------------- | ------------------------------------ |
| M3 control        | X              | X         | X                 | X if frozen and durable              |
| M5 challenger     | X              | X         | X                 | X if frozen and durable              |
| M4 oracle ceiling | X              | X         | X                 | X if frozen and durable              |

The preferred primary restorer set is the smallest set that preserves the principal deterministic and learned regimes already evaluated in Notebook 46. Notebook 51 must not add new blends, thresholds, post-processing or architecture changes.

## 6.2 Freeze and provenance gates

- Load Notebook 50 M5 policy, models, calibrators, schema, thresholds, replay pack and semantic mask hashes by explicit manifest.
- Load Notebook 45 M3/M4 identity and Notebook 46 restorer implementations/checkpoints by explicit manifest.
- Verify page roster, source-group identity, mask shape, authorised-mask containment and semantic hashes.
- Replay a frozen multi-page sample through M5 mask construction and each restorer before the definitive run.
- Reject missing, ambiguous or mismatched artefacts. Never reconstruct a missing calibrator or silently substitute a latest run.
- Write all mask and candidate identities before clean truth is loaded for scoring.

## 6.3 Candidate assembly contract

- Start from the same frozen baseline page for every mask/restorer cell.
- Apply reconstruction only within the exact fixed residual mask for that cell.
- Preserve protected observed PRINT bytes exactly and apply frozen REMOVE treatment outside the residual target.
- Final candidates must equal the baseline everywhere except authorised treatment coordinates.
- Outside-mask drift, protected-byte drift and candidate completeness are asserted per page and per cell.

## 6.4 Primary endpoints

| **Endpoint**           | **Question**                                                                    |
| ---------------------- | ------------------------------------------------------------------------------- |
| False inserted ink     | Does M5 reduce invented or retained ink relative to M3 for the same restorer?   |
| Missing genuine ink    | Does M5 reduce lost or unrecovered print relative to M3?                        |
| Ink and edge F1        | Does M5 improve thresholded content and topology?                               |
| MAE / SSIM             | Does pixel fidelity improve? Secondary to structural and asymmetric ink errors. |
| Catastrophic regions   | Does M5 remove or create severe local failures?                                 |
| Paired page delta      | Is the M5-M3 effect consistent within the same page/restorer?                   |
| Source-group stability | Does the mask benefit generalise rather than concentrate?                       |
| Restorer spread        | Does the range/variance across restorers contract under M5?                     |
| Oracle-gap closure     | How much of M4 reconstruction performance does M5 recover?                      |

## 6.5 Primary estimands

6. For each frozen restorer, estimate the paired M5 minus M3 change on every registered endpoint.
7. For each mask, estimate between-restorer dispersion and the frequency with which restorer choice changes the page-level Pareto set.
8. Estimate the fraction of the M3-to-M4 improvement recovered by M5, separately by restorer, page and source group.
9. Test the interaction: whether the M5 benefit is similar across restorers or depends materially on one restorer.
10. Report wins, losses, ties, clustered confidence intervals and worst-page deltas. Do not rely on pooled means alone.

## 6.6 Decision logic

| **Outcome**                                                                      | **Interpretation**                                                                          | **Programme decision**                                                                     |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| M5 improves several restorers, closes part of M4 gap and reduces restorer spread | Residual-mask conditioning is the main limiting factor.                                     | Prioritise mask refinement and favour the simplest restorer meeting quality/runtime needs. |
| M5 helps only one restorer                                                       | Mask and restorer interact materially.                                                      | Do not generalise M5; investigate the interaction with a new pre-registration.             |
| M5 improves mask metrics but not reconstruction                                  | Notebook 50 optimised the wrong mask qualities or M4 is not an adequate operational oracle. | Return to diagnostic design; do not tune restorers post hoc.                               |
| M5 worsens catastrophic or source-specific outcomes                              | Mask challenger fails reconstruction safety despite mask-only promise.                      | Reject M5 and retain M3.                                                                   |
| M5 approaches M4 but restorer spread remains large                               | Both mask and restorer matter.                                                              | Proceed to a separately designed restorer experiment only after closing Notebook 51.       |
| Integrity or replay failure                                                      | Performance is uninterpretable.                                                             | Invalidate run and correct implementation.                                                 |

## 6.7 Required outputs

- experiment contract and resolved artefact inventory
- mask/restorer/page completion matrix
- whole-page candidates and semantic hashes
- results_by_page_restorer_mask.csv
- results_by_source_group.csv
- paired_M5_vs_M3_deltas.csv
- oracle_gap_closure.csv
- restorer_dispersion_by_mask.csv
- Pareto membership by page and source group
- catastrophic/failure cases
- immutability and protected-byte audits
- review contact sheets
- notebook51_verdict.json
- experiment_identity.json
- executive_summary.txt
- standard upload bundle and manifest

# 7. Stage gates and hand-offs

| **From**         | **Required hand-off**                                                                                                 | **Stop condition**                                                                                |
| ---------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 49 to 50         | Locked observable dataset, class definitions, feature schema, source-group analysis, verdict SIGNAL_SUFFICIENT_FOR_M5 | No stable separability, source concentration, leakage or invalid lock                             |
| 50 to 51         | Frozen M5 bundle, exact replay, M3 fallback proof, M5 masks/hashes, verdict M5_READY_FOR_FROZEN_CROSSOVER             | Any safety, catastrophe, immutability, stability or low-evidence failure                          |
| 51 to later work | Causal paired evidence on mask effect, oracle-gap closure and restorer dispersion                                     | Do not optimise restorers unless Notebook 51 shows residual restorer sensitivity remains material |

# 8. Failure taxonomy and required response

| **Failure**                          | **Required response**                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------- |
| Truth leakage                        | Invalidate the run. Rebuild from a clean observable-only path; do not salvage metrics. |
| Roster/hash mismatch                 | Write discrepancy report and stop before modelling or reconstruction.                  |
| M5 decision outside M3/authority     | Invalidate page and run; the challenger has exceeded its intervention boundary.        |
| Outside-mask or protected-byte drift | Invalidate candidate immediately.                                                      |
| Source-group concentration           | Report scoped evidence; do not claim general mask improvement.                         |
| Low coverage                         | Retain as conservative baseline; never lower thresholds merely to create volume.       |
| Wide uncertainty                     | Report insufficient evidence rather than a pass.                                       |
| Post-hoc rule change                 | Create a new versioned experiment and fresh OOF evaluation.                            |
| Unexpected M4 behaviour              | Treat M4 as a diagnostic model requiring review, not unquestionable truth.             |
| Partial/corrupt checkpoint           | Fail closed and rebuild only the affected page if deterministic identity permits.      |

# 9. Standard upload package for each notebook

| **Folder/file**              | **Contents**                                                               |
| ---------------------------- | -------------------------------------------------------------------------- |
| README.txt                   | Purpose, run identity, verdict, key caveats and underlying run location.   |
| upload_manifest.csv          | Relative path, role, byte size and SHA-256 for every packaged artefact.    |
| tables/                      | Curated decision, page, source-group, failure and integrity tables.        |
| review/                      | Contact sheets and representative/worst-case review imagery.               |
| audit/                       | Identity, leakage, replay, immutability, completion and exception reports. |
| notebookXX_upload_bundle.zip | Curated package only; large caches remain referenced by manifests.         |

# 10. Implementation checklist

## Before coding

- Freeze names, scope, primary question and explicit non-authorisations.
- Name exact source runs rather than resolving broadly by latest modification time.
- Define primary labels, masks, metrics, catastrophic cases and verdicts before outcomes.
- Separate observable construction, lock creation, truth unlock and scoring cells visibly.

## During execution

- Write failures before raising errors.
- Checkpoint per page with content fingerprints and corruption checks.
- Persist every learned object immediately.
- Use deterministic ordering for pages, regions and pixels.
- Record software versions, seeds, constants and timings.

## Before interpretation

- Verify complete page x mask x restorer matrix.
- Verify zero outside-mask and protected-byte drift.
- Verify source-group separation and exact OOF assignment.
- Review concentrated failures and not only favourable means.
- Regenerate summaries from scored metrics, not inventories.

## Before reference upload

- Generate executive summary and verdict JSON.
- Produce the curated Final Upload Package.
- Check that manifests point to all heavyweight underlying artefacts.
- Retain failed challengers and their evidence without overwriting the incumbent.

# 11. Source basis

This specification consolidates and extends the established project contracts below. The proposed details for Notebooks 49-51 are programme design recommendations; they are not claimed as previously executed facts.

| **Project reference**                                                                            | **Use in this specification**                                                                                                                       |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| V8_6_Residual_Reconstruction_Readiness_Programme.docx                                            | M0-M4 mask roles, M3 collar, mask-readiness endpoints, oracle gap and frozen restorer crossover principle.                                          |
| 45_V8_6_Residual_Mask_Readiness_and_Oracle_Audit.ipynb.txt                                       | Actual M0-M3 construction, post-lock M4 construction, semantic locks and current readiness metrics.                                                 |
| 46R6_V8_6_FAIL_CLOSED_PAGE_CHECKPOINTED_RESUMABLE_Crossover_with_Whole_Page_Candidates.ipynb.txt | Fail-closed, checkpointed fixed-mask/frozen-restorer crossover architecture.                                                                        |
| V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx                              | Preserve/remove/ambiguous architecture, observable geometry evidence, controlled mask and exact outside-mask identity.                              |
| Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx                    | Observable-only predictor firewall, PRINT/ANNOTATION/UNCERTAIN fail-closed semantics, grouped validation, asymmetric safety and audit requirements. |
| V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx                              | Controlled scalar-versus-spatial comparison and fixed 33x33 topology representation.                                                                |
| V7_9_x_Research_Findings_and_Next_Steps.docx                                                     | Evidence against further hand-built positive threshold gates and requirement for local learned decisions.                                           |
| 17_V7_8_Rebuilt_Independent_Data_Builder_Cached_Registration.txt                                 | Canonical support/routing arrays, deterministic semantic locking and predictor construction.                                                        |
| 18_V7_8_2_Rebuilt_Builder_Equivalence_Audit.txt                                                  | Stage-by-stage equivalence and first-failure semantics.                                                                                             |
| V8_2_Development_Whole_Page_Experiment_Decision_Record.docx                                      | Development first, freeze second, independent test third; self-contained durable outputs and standard upload package.                               |
| 10_Reconstruction_Benchmark.txt                                                                  | Clean truth after reconstruction only, outside-mask immutability and paired page-level evaluation.                                                  |

# 12. Confidence, unresolved choices and next decision

**Confidence: 92%** Confidence is high in the three-stage separation, truth firewall, fail-closed M3 fallback, grouped evaluation and frozen crossover because those principles are explicit throughout the project record. Confidence is lower for the exact Notebook 50 target granularity, architecture and numerical promotion gates because Notebook 49 has not yet quantified the M3-to-M4 class balance and observable separability. Those items must be frozen only after Notebook 49, without using Notebook 50 outcomes.

## Unresolved choices to freeze after Notebook 49

- Pixel-level, region-level or hierarchical M5 decision unit.
- Whether the primary target is four-class, three-class or RECONSTRUCT versus NOT-RECONSTRUCT with secondary routing.
- One primary spatial context size and whether one outer-context ablation is justified.
- Class-specific confidence thresholds and spatial coherence rules.
- Numerical catastrophic definitions and minimum evidence/coverage gates.
- Whether boundary exclusion is learned jointly or remains a separate frozen observable rule.

**Next question** What exact outputs from the completed Notebook 45 and Notebook 46 runs are available to populate Notebook 49 without recomputing upstream masks or restorations?