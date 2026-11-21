**MUSIC ANNOTATION PROJECT**

Notebook 87

**Production-Profile Paired HITL A1 Cleanup Experiment**

**Decision in one sentence**

Notebook 87 closes threshold-only cleanup as a production route: reducing the annotation-confidence threshold buys additional cleaning only by sacrificing increasingly large amounts of genuine musical print. A1 remains the preferred baseline, and the next challenger should be a contextual A1-based safe-cleanup model rather than another probability threshold sweep.

| **Candidate** | **Mean extra annotation cleanup** | **Mean genuine-print loss** | **Decision**                                                   |
| ------------- | --------------------------------- | --------------------------- | -------------------------------------------------------------- |
| C090_P010     | 1.71%                             | 0.23%                       | Reject for production: too little gain for non-zero print risk |
| C0875_P010    | 4.66%                             | 3.80%                       | Reject                                                         |
| C085_P010     | 13.99%                            | 7.18%                       | Strongly reject                                                |

**Document status:** development decision record and A5 commissioning direction; not production authorisation.

# 1. Purpose and scope

Notebook 87 was designed to answer a specific unresolved question: whether the simple high-annotation-confidence / low-print-confidence cleanup idea remained safe when evaluated using production-style, human-reviewed HITL masks rather than the historically thin paired-development masks.

- The paired annotated pages were passed through the production YOLO draft and confidence-map workflow.
- The masks were reviewed blind to clean truth and retained in a separate PAIRED_HITL_DEVELOPMENT namespace.
- Reviewed-mask and candidate-output locks were closed before clean truth was introduced.
- Clean truth was registered into the frozen Notebook 07 observed coordinate system before pixelwise scoring.
- The experiment compared A1 with three predeclared extra-cleanup policies; it did not tune thresholds after outcomes were viewed.

**What the experiment tested**

Starting from A1, should an AMBIGUOUS pixel be changed from its exact observed byte to the frozen local paper estimate when YOLO annotation confidence is high and calibrated print probability is below 0.10?

# 2. Candidate definitions

| **Arm**     | **Additional cleanup rule applied to A1-preserved AMBIGUOUS pixels** |
| ----------- | -------------------------------------------------------------------- |
| A1_BASELINE | No additional cleanup. AMBIGUOUS remains the exact observed byte.    |
| C090_P010   | p(annotation) >= 0.900 and p_print &lt; 0.10 -&gt; paper             |
| C0875_P010  | p(annotation) >= 0.875 and p_print &lt; 0.10 -&gt; paper             |
| C085_P010   | p(annotation) >= 0.850 and p_print &lt; 0.10 -&gt; paper             |

The burden of proof remained asymmetric. A1 was the fail-closed output; every candidate had to reproduce A1 byte-for-byte outside its selected cleanup set, and no candidate was permitted to change pixels outside the reviewed authority.

# 3. Integrity and population handling

Notebook 87 ultimately required several implementation corrections before the result became interpretable. These corrections are part of the experiment record because they affect the validity of the coordinate-level truth comparison.

| **Issue**                          | **Resolution**                                                                                                                                            | **Why it mattered**                                                                           |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Population contamination risk      | All paired HITL drafts, confidence maps, reviews, masks and locks were written under paired_hitl_development/.                                            | Prevented the paired development set from entering the historical real-world HITL population. |
| Observed/mask page shift           | Corrected masks were assigned one-to-one to pages using spatial agreement against the page-specific YOLO draft, rather than trusting shifted audit paths. | Prevented one page from being scored with another page's reviewed mask.                       |
| OpenCV inpaint mask error          | The prepared mask remained uint8 0/255 through Telea; Boolean authority was created only for logical operations.                                          | Restored the frozen reconstruction contract and eliminated dtype-induced failures.            |
| Zero M3 / zero authority           | Pages with no relevant candidate population were explicitly skipped and recorded.                                                                         | Kept zero-exposure controls out of the candidate-effect denominator.                          |
| Clean/observed resolution mismatch | Clean pages were registered after prediction lock into the Notebook 07 observed coordinate system, with validity masks and scale-aware truth tolerance.   | Made pixelwise truth labels spatially meaningful.                                             |
| One-pixel mask canvas differences  | Only narrow border discrepancies were reconciled with zero-authority padding and audit records.                                                           | Avoided resizing or extending intervention authority.                                         |

# 4. Principal quantitative result

The production-profile frontier is monotonic and unattractive. Lowering the annotation-confidence threshold increases annotation cleanup, but genuine-print loss rises sharply. No tested threshold offers meaningful cleanup while remaining effectively non-inferior to A1 on print safety.

| **Candidate** | **Mean cleanup** | **Mean print loss** | **Interpretation**                                                             |
| ------------- | ---------------- | ------------------- | ------------------------------------------------------------------------------ |
| C090_P010     | 1.71%            | 0.23%               | Small cosmetic benefit; still sacrifices genuine print on a minority of pages. |
| C0875_P010    | 4.66%            | 3.80%               | Material print damage for modest additional cleanup.                           |
| C085_P010     | 13.99%           | 7.18%               | Large cleanup, but destructive to print and unsuitable for restoration.        |

The pooled pixel counts lead to the same conclusion:

| **Candidate** | **Truth annotation cleaned** | **Truth print lost** | **Annotation cleaned per print pixel lost** |
| ------------- | ---------------------------- | -------------------- | ------------------------------------------- |
| C090_P010     | 630                          | 346                  | 1.82                                        |
| C0875_P010    | 4,987                        | 8,894                | 0.56                                        |
| C085_P010     | 16,685                       | 19,375               | 0.86                                        |

**Interpretation**

C090_P010 is the least damaging tested rule, but it does not clear the project objective. Removing 630 additional annotation pixels at a cost of 346 genuine-print pixels is not an acceptable default for a preservation-first music restorer. The lower thresholds are decisively worse.

# 5. Page-level evidence

The aggregate trade-off is not caused by uniform weak performance. The same fixed rule can be excellent on one page and severely destructive on another. That is the strongest evidence that safe cleanup is contextual rather than threshold-defined.

| **Pattern**                              | **Representative pages**                    | **Meaning**                                                                                             |
| ---------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| High cleanup with no detected print loss | Page023, Page051, Page039, Page028, Page005 | The annotation and print probabilities contain useful signal in favourable local contexts.              |
| Severe print loss under the same rule    | Page003, Page049, Page004, Page055, Page036 | High annotation confidence plus low p_print is not sufficient evidence that changing the pixel is safe. |

At C085_P010, examples included Page003 cleaning 1,009 annotation pixels while losing 5,364 print pixels, Page049 cleaning 955 while losing 3,214, Page004 cleaning 266 while losing 2,623, and Page055 cleaning 664 while losing 2,095. Conversely, several pages achieved hundreds of annotation pixels cleaned with zero detected print loss. The model required is therefore one that can distinguish these contexts, not a more permissive global threshold.

# 6. What Notebook 87 established about HITL masks

The production-style paired masks were materially different from the historical thin paired masks. Human review was not a cosmetic adjustment to the YOLO drafts; on some pages it added or erased a large proportion of the final authority. This validates the decision to rerun the truth experiment through the real HITL workflow.

- The new result is more relevant to production-style reviewed masks than Notebook 86.
- The reviewed mask must remain an authority artefact separate from the raw YOLO confidence map.
- Low YOLO confidence cannot be interpreted as evidence of non-annotation because reviewers may add genuine annotation that YOLO missed.
- Broad or manually expanded masks change the contextual problem by including more print and reducing visible outside-mask structural evidence.
- The existing reviewed masks should not be mechanically thinned. If particular sources required extensive human rescue, that is evidence for a future YOLO retraining challenge, not authority for post-review erosion.

# 7. Decisions

| **Decision**                     | **Status**      | **Rationale**                                                                      |
| -------------------------------- | --------------- | ---------------------------------------------------------------------------------- |
| Retain A1 as the design baseline | ADOPT           | A1 remains the safest current visual-restoration behaviour.                        |
| Promote C090_P010                | REJECT          | Non-zero print harm for a small cleanup gain.                                      |
| Promote C0875_P010               | REJECT          | Material print loss.                                                               |
| Promote C085_P010                | STRONGLY REJECT | Clearly destructive.                                                               |
| Continue threshold sweeping      | STOP            | The frontier shows a context problem, not a missing global threshold.              |
| Proceed with contextual A5       | RECOMMENDED     | Notebook 87 supplies contrasting safe and unsafe examples for supervised learning. |
| Automatic production promotion   | NOT AUTHORISED  | The pages are development-exposed and the experiment was diagnostic.               |

# 8. Way forward: A5 contextual safe-cleanup model

**A5 objective**

Start from exact A1. Change an A1-preserved AMBIGUOUS pixel only when an observable-only contextual model supplies strong positive evidence that the pixel is safely removable annotation. Every uncertain, unsupported, invalid or out-of-distribution case must remain exact A1.

## 8.1 Target and treatment

| **Truth-side class**                     | **Required A5 treatment**                                                |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| Genuine printed music                    | KEEP_A1. Any cleanup is a structural false positive.                     |
| Annotation / non-print                   | Eligible for SAFE_ANNOTATION_CLEANUP if the frozen decision rule passes. |
| Truth-uncertain / registration-sensitive | KEEP_A1 and report separately.                                           |

## 8.2 Observable evidence

| **Evidence family** | **Recommended representation**                                         | **Purpose**                                                                          |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Observed image      | 33x33 local patch plus wider context such as 97x97                     | Distinguish local ownership and broader musical structure.                           |
| Annotation evidence | YOLO p(annotation) and neighbourhood context                           | Identify likely handwriting without treating confidence as correction authority.     |
| Print evidence      | Calibrated p_print plus horizontal, vertical and compact support       | Protect staff lines, stems, glyphs and print crossings.                              |
| Mask geometry       | Depth, distance to boundary, thickness and component context           | Account for the broad reviewed HITL mask profile.                                    |
| Paper context       | Paper estimate, surrounding blank fraction, distance to credible print | Target annotation away from print, where A1 leaves the largest cosmetic opportunity. |
| Topology            | Edges, continuity, junctions and local line responses                  | Differentiate handwriting from printed musical structure.                            |

## 8.3 Training and evaluation design

1. Train only on the A1-preserved AMBIGUOUS population from the production-profile paired HITL development pages.
2. Use source-group-separated outer folds and page-grouped fitting/calibration. Pixels from the same page must not cross train/test boundaries.
3. Exclude truth-uncertain pixels from positive cleanup training or treat them as mandatory KEEP_A1, not annotation.
4. Generate source-group out-of-fold probabilities and whole-page A5 outputs before any final operating threshold is selected.
5. Use conservative calibrated thresholds and a page-global spatial filter; isolated cleanup decisions should fail back to A1.
6. Compare A5 primarily with A1. A3 and V8.9 remain diagnostic comparators rather than the success baseline.
7. Report PAGE_DOMINANT, NEAR_PRINT, deep-mask, source-group and reviewer-added-mask strata separately.
8. If development gates pass, freeze one A5 bundle and test once on a fresh, untouched paired cohort processed blindly through the same HITL workflow.

## 8.4 Predeclared success criteria

| **Gate**              | **Required outcome**                                                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Print non-inferiority | Page-clustered A5-minus-A1 print harm must remain within a small predeclared margin; the 95% upper harm bound must pass.             |
| Annotation benefit    | Positive page-clustered lower confidence bound for annotation cleaned beyond A1, with gains across multiple pages and source groups. |
| Concentrated failure  | Zero catastrophic print-loss components under a predeclared connected-component rule.                                                |
| A1 identity           | A5 must equal A1 exactly outside valid SAFE_ANNOTATION_CLEANUP decisions.                                                            |
| Authority             | Zero pixels changed outside the reviewed prepared mask.                                                                              |
| Transport             | Benefit must remain credible across broad/deep masks and reviewer-added regions.                                                     |
| Independent evidence  | No promotion until a fresh paired HITL-reviewed cohort passes without threshold movement, refitting or exclusion changes.            |

# 9. Separate detector and mask-authoring workstream

Notebook 87 also indicates that the upstream YOLO detector and the downstream safe-cleanup model should remain separate workstreams.

- Measure draft-to-reviewed additions and erasures by source group, including Pomp and other sources requiring heavy human intervention.
- Use repeated reviewer additions or erasures as evidence for a separately governed YOLO retraining challenger.
- Do not train the correction model to compensate silently for a poor detector, and do not thin authoritative reviewed masks merely to resemble the old paired masks.
- Retain the continuous confidence map, draft mask, reviewed mask, prepared authority, routing map and correction decision as separate audit artefacts.

# 10. Evidence status and limitations

- Notebook 87 uses development-exposed paired pages. It is valid for development and model commissioning, not independent promotion.
- Truth registration required upscaling and homography into the Notebook 07 coordinate system. Validity masks and scale-aware tolerance reduce, but do not eliminate, registration-label uncertainty.
- The experiment demonstrates that the tested deterministic policies fail; it does not demonstrate that every contextual model will succeed.
- The numerical conclusions should be tied to the exact Notebook 87 run ID, prediction lock, review lock, registration lock and output-table hashes in the final A5 pre-registration.

**Final conclusion**

Notebook 87 closes the simple probability-band route. Annotation confidence contains useful positive evidence, but correction safety is context-dependent. The correct next step is A5: a conservative, contextual, observable-only safe-cleanup model built on A1, with A1 retained exactly whenever the model lacks strong evidence.

# 11. Reference basis

- Notebook 87 - Paired HITL Development Draft, Blind Review, Freeze, and A1 Cleanup Truth Experiment.
- Music Annotation Project - Recent Experimental Outcomes and Objectives for A5.
- V7.5 Design Record - Annotation Confidence and Correction Safety.
- V7.7 Asymmetric Structural-Risk Calibration.
- V7.8 Rebuilt Independent-Data Builder and V7.8.2 Equivalence Audit.
- Observed-Print Classifier Pre-Registered Commissioning Specification v1.1.
- V8 Full-Pixel Whole-Page OOF Reference and V8.1 Specification.
- Production Retraining Policy - YOLO and Structural Safety.
- Notebook 85 HITL Annotation-Probability Heatmaps and Band-Expansion Atlas.
- Notebook 86 Paired-Truth Annotation x Print Confidence Experiment.

**Confidence: 97%.** The threshold-rejection conclusion is supported by the production-profile paired truth frontier, pooled counts and page-level contrast. Confidence is below 100% because registration-based truth labels remain subject to residual alignment uncertainty, and A5 has not yet been trained or independently validated.

**Next question:** Does the A5 contextual model achieve a positive page-clustered annotation-cleanup gain over A1 while keeping the upper confidence bound on print harm within the predeclared non-inferiority margin and producing zero catastrophic print-loss components?