Notebook 59R3

**151-Page 59R2-Retained Morphology and Failure-Family Audit**

_Experiment record, conclusion and recommended next steps | 14 November 2026_

**Decision statement:** 59R3 identifies two broad retained-region regimes and several widespread, overlapping practical failure families. The evidence does not support one global threshold treatment. It supports a family-specific challenger programme with authoritative semantic review before retraining.

# 1. Executive summary

Notebook 59R3 examined coherent regions retained by the 59R2 observed-print decision layer across the complete 151-page annotated-only cohort. The experiment was prompted by recurring visible cases including top-right part or catalogue identifiers such as "D32", dark pen-like notes, pale pencil near the operating threshold, and long glyphs or arrows that remain on corrected pages.

The unsupervised feature space separated cleanly into two clusters. However, the notebook's automatic label "ONE_DOMINANT_REPEATABLE_MORPHOLOGY" overstates the practical conclusion. The dominant cluster is a statistical regime of numerous small fragments, not a single semantic morphology. A second cluster contains far fewer but much larger, darker, lower-probability regions and represents a material operational problem.

| **Headline result**            | **Value**                                                      |
| ------------------------------ | -------------------------------------------------------------- |
| Cohort                         | 151 pages; 12,653,698 scored pixels                            |
| Coherent retained regions      | 21,908 regions on 150 pages                                    |
| Selected unsupervised solution | 2 clusters; silhouette 0.8689                                  |
| Cluster 0                      | 21,538 small regions; 71.16% of pixels; present on 150 pages   |
| Cluster 1                      | 370 large dark regions; 28.84% of pixels; present on 100 pages |
| Interpretive conclusion        | Two-scale regime with multiple overlapping failure families    |

# 2. Scientific question

The experiment asked:

Across the 151-page cohort, are the largest coherent 59R2-retained annotation-like regions dominated by one repeatable dark-ink morphology, or do pale near-threshold, dark-pen, print-overlap, top-right identifier and long-glyph failures form distinct families requiring different treatments?

The experiment remained inside the project's established separation between annotation identity and correction safety. A reviewed annotation mask authorises where intervention may occur, while downstream route and classifier evidence determines how safely the region can be changed. High confidence that a mark is annotation does not, by itself, establish safe removal.

_Project basis: V7.5 Design Record, sections 1-5 and guardrails; intervention remains inside the authoritative mask and outside-mask pixels remain unchanged._

# 3. Experiment design

## 3.1 Population and retained-region construction

- Input population: all saved 59R2 scored coordinates from the completed 151-page whole-page cohort.
- Retained candidates: observed dark pixels grouped into five p_remove bands: 0.993-0.995, 0.985-0.993, 0.950-0.985, 0.900-0.950 and below 0.900.
- Unit of analysis: coherent connected regions within each probability band, rather than whole annotation-mask components.
- Minimum retained-region size: 12 exact scored pixels.
- No clean truth was used. Regions are annotation-like candidates and descriptive hypotheses, not semantic labels.

## 3.2 Observable features

| **Feature family**           | **Examples**                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| Scale and geometry           | Pixel count, width/height fractions, fill, aspect ratio, major/minor axis, linearity |
| Stroke topology              | Skeleton-to-pixel ratio, endpoints, junctions, orientation                           |
| Intensity and model response | Observed intensity quantiles; p_remove mean, median, range and variation             |
| Local context                | Dark density and edge density in a surrounding ring                                  |
| Diagnostic families          | Top-right identifier, long glyph, pale near-threshold, dark pen and dense-dark flags |

Raw page position was not included in the unsupervised clustering. Top-right placement was evaluated afterwards as a diagnostic family. This reduces the risk that the clustering result merely reproduces a position rule.

## 3.3 Clustering and interpretation

- Robust scaling and median imputation were applied to observable morphology/context features.
- K-means solutions for k=2 through k=8 were compared using silhouette score and inertia.
- Cluster names were assigned only after examining aggregate profiles and remained descriptive hypotheses.
- Dedicated recurrence tables and contact sheets were produced for top-right identifiers and long glyphs.

# 4. Results

## 4.1 Cluster selection

| **k** | **Silhouette score** | **Interpretation**      |
| ----- | -------------------- | ----------------------- |
| 2     | 0.8689               | Clear best separation   |
| 3     | 0.7654               | Still strong, but worse |
| 4     | 0.5882               | Substantially weaker    |
| 5     | 0.5704               | Substantially weaker    |
| 6-8   | 0.3352-0.3637        | Poorer separation       |

The large gap between k=2 and the alternatives indicates two strong broad regimes in the feature space. It does not establish that there are only two semantic error types.

## 4.2 Cluster profiles

| **Measure**          | **Cluster 0: small-fragment regime** | **Cluster 1: large-dark regime** |
| -------------------- | ------------------------------------ | -------------------------------- |
| Regions              | 21,538                               | 370                              |
| Pages                | 150                                  | 100                              |
| Retained pixels      | 1,326,053                            | 537,345                          |
| Pixel share          | 71.16%                               | 28.84%                           |
| Region share         | 98.31%                               | 1.69%                            |
| Median region pixels | 26                                   | 1,128                            |
| Median observed grey | 133                                  | 52                               |
| Median p_remove      | 0.9682                               | 0.2761                           |
| Median aspect ratio  | 1.71                                 | 1.64                             |
| Dark-pen flag rate   | 33.8%                                | 84.1%                            |

Cluster 0 dominates counts and total pixels because very many small retained fragments occur across nearly every page. Cluster 1 is operationally important despite its low region count: its median region is more than forty times larger, much darker, and assigned far lower removal probability.

**Corrected interpretation:** the experiment found a dominant small-fragment regime plus a distinct large-dark retained-region regime. It did not demonstrate one homogeneous morphology.

## 4.3 Special failure families

| **Diagnostic family**          | **Regions** | **Pages** | **Cohort prevalence** | **Median observed** | **Median p_remove** | **Median aspect** |
| ------------------------------ | ----------- | --------- | --------------------- | ------------------- | ------------------- | ----------------- |
| Top-right identifier candidate | 2,487       | 123       | 81.5%                 | 131.5               | 0.9559              | 1.58              |
| Long-glyph candidate           | 1,767       | 138       | 91.4%                 | 81.0                | 0.8741              | 6.75              |
| Pale near-threshold            | 3,293       | 134       | 88.7%                 | 166.0               | 0.9858              | 1.67              |
| Dark-pen candidate             | 7,585       | 147       | 97.4%                 | 28.0                | 0.5796              | 1.80              |
| Dense-dark candidate           | 5,499       | 142       | 94.0%                 | 6.5                 | 0.7991              | 1.88              |

These flags overlap and should not be summed as mutually exclusive classes. Their value is prevalence and treatment diagnosis: the same page or region may be dark, elongated and top-right.

## 4.4 Top-right identifiers

The top-right family is clearly recurrent. Candidate regions were found on 123 pages. The recurrence fingerprint includes substantial populations in the uppermost 5% of the page and the rightmost 10-20% of page width. The two largest recurring bins included:

- 62 regions across 38 pages, x=0.8-0.9 of page width, y=top 5%, size 200-600 pixels, median p_remove approximately 0.3003.
- 59 regions across 34 pages, x=0.9-1.0, y=top 5%, size 75-200 pixels, median p_remove approximately 0.3245.

This supports the observed D32-type failure. It also shows at least two subtypes: large, dark, very low-removal-score identifiers and smaller top-right fragments closer to the threshold frontier. Top-right position is useful diagnostic context, but must not become an automatic removal shortcut without semantic validation.

## 4.5 Long glyphs and arrows

Long-glyph candidates occurred on 138 pages, with median aspect ratio 6.75, median observed grey 81 and median p_remove 0.8741. This is materially different from pale near-threshold material, whose median observed grey was 166 and median p_remove was 0.9858.

The result supports the hypothesis that arrows and sweeping marks are not merely calibration errors. A fixed 33x33 local patch can truncate the feature that defines the mark, so a larger or multi-scale context is a plausible treatment hypothesis.

# 5. Conclusion

Notebook 59R3 provides strong evidence that 59R2 under-removal is recurrent across the cohort and is not confined to Page258. The evidence is clearest for dark pen-like regions, top-right part/catalogue identifiers, long glyphs and pale near-threshold material. These phenomena occur on large fractions of the 151-page cohort.

The best summary is not "one dominant repeatable morphology". The correct conclusion is:

Two broad statistical scales exist, and several overlapping practical failure families recur across the cohort. The small-fragment regime dominates counts and pixels, while a distinct large-dark regime contains fewer but much larger, visually salient retained regions. Different families are unlikely to be solved safely by one global threshold.

The experiment therefore supports a new challenger investigation, but it does not authorise threshold promotion, retraining from unreviewed predictions, or production deployment. The 151-page population is failure-discovery and regression evidence. It is not automatically supervised truth.

# 6. Treatment implications

| **Failure family**       | **Likely mechanism**                                                             | **Recommended research treatment**                                                                     |
| ------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Pale near-threshold      | Calibration/operating-point uncertainty near probability plateaus                | Test calibration and policy separately, with safety scoring and no post-hoc threshold promotion        |
| Dark pen / dense dark    | Insufficient annotation-versus-print ownership discrimination                    | Add authoritative dark-ink annotation examples to a spatial challenger                                 |
| Top-right identifiers    | Local morphology plus page-role/layout context; recurring first-page identifiers | Test a broader-context or page-role branch; prohibit raw coordinates as an unvalidated shortcut        |
| Long glyphs / arrows     | Defining topology exceeds the 33x33 field of view                                | Test one pre-registered larger or multi-scale context while holding labels, folds, loss and rule fixed |
| Print-overlap candidates | True print and annotation interleave within a few pixels                         | Retain PRINT / ANNOTATION / UNCERTAIN semantics and conservative fallback                              |

_Project basis: V8.0 controlled design found that darkness, orientation and continuation alone were insufficient; the central problem is local ownership where annotation and print overlap. The 33x33 experiment deliberately deferred multi-scale context._

# 7. Limitations

- No clean truth or human ownership labels were used. Family and cluster names are hypotheses.
- All outputs reported one source group, indicating unresolved or collapsed source-group identity. Cross-source stability was not established.
- Connected regions were created within probability bands. A physical mark split across bands may appear as multiple regions.
- Top-right detection is positional and intentionally broad; it includes true identifiers, fragments and unrelated material.
- K-means optimises geometric separation, not semantic usefulness or review priority.
- The automatic conclusion rule placed excessive weight on dominant pixel share and page presence and insufficient weight on region size, visual salience and distinct treatment needs.

# 8. Recommended next steps

## 8.1 Immediate: authoritative semantic review

Create a page-diversified review set of approximately 500 regions:

1. 100 top-right identifier candidates, ensuring repeated D32-type placements and varied page titles/editions.
2. 100 long-glyph candidates, including arrows, sweeping marks and elongated handwriting.
3. 100 large dark Cluster 1 regions.
4. 100 pale near-threshold Cluster 0 regions.
5. 100 high-context or apparent print-overlap regions.

Use the ownership classes ANNOTATION, GENUINE PRINT, MIXED/OVERLAP, UNCERTAIN and MASK ERROR. Report purity by cluster, diagnostic family, page and true source group. This review determines whether each family is a valid modelling target and whether families require separate treatment.

## 8.2 Repair provenance before modelling

- Resolve the actual source_group for every page rather than UNKNOWN.
- Preserve the exact 151-page run, region table, contact sheets and hashes as a permanent failure-discovery/regression record.
- Do not consume the full cohort into training if the cohort is intended to remain a regression set.

## 8.3 Pre-register one controlled representation experiment

If semantic review confirms the families, run a controlled 59R3/59R4 challenger comparison in which the only material experimental variable is context representation:

- Control: current 33x33 observable spatial representation.
- Treatment: one larger or multi-scale context, for example 33x33 plus 97x97, with identical centres, labels, page/source-group folds, calibration and decision rule.
- Keep raw page coordinates, page number, score title and source identity out of the primary model.
- Test page-role or top-right context only as a separately declared ablation, not as a silent shortcut.
- Retain the three-way operational outcome: PRINT, ANNOTATION and UNCERTAIN, with V7.8 fallback for uncertainty.

_Project basis: Observed-Print Classifier pre-registration allows 33x33 primary patches, requires page/source-group separation and prohibits raw x/y, page identity and source identifiers. V8.0 deferred larger context to a later controlled ablation._

## 8.4 Evaluation and promotion discipline

- Use page- and source-group-grouped out-of-fold predictions.
- Compare against frozen V7.8/59R2, not only against a lowered threshold.
- Report family-specific annotation purity, false preservation, true removal, catastrophic concentrations and review burden.
- Keep an untouched independent cohort for any promotion claim.
- Promote no model or threshold automatically; preserve the incumbent until a documented challenger decision passes integrity, regression and safety gates.

_Project basis: production retraining is event-driven; unreviewed outputs are not training truth; frozen regression sets should be preserved; promotion is never automatic._

# 9. Decision record

| **Decision item**                                     | **59R3 outcome**                                                                                                                       |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Is under-removal isolated?                            | No. Candidate families occur across most of the cohort.                                                                                |
| Is there one homogeneous morphology?                  | No. There are two broad scales and multiple overlapping practical families.                                                            |
| Should the global threshold simply be reduced?        | No. Lower thresholds mix pale, dark, elongated and overlap cases and do not address representation limits.                             |
| Is new learning justified?                            | Yes, as a controlled challenger investigation after authoritative family review.                                                       |
| Can the 151 pages be used directly as training truth? | No. Preserve them as diagnostic/regression evidence unless individual regions receive authoritative labels.                            |
| Preferred next model hypothesis                       | Spatial ownership with one additional pre-registered context scale, plus a separate layout-context ablation for top-right identifiers. |

# 10. Confidence and next question

**Confidence: 97%.** The numerical clustering, prevalence and recurrence outputs are internally consistent and the two-cluster separation is strong. Confidence is lower for semantic interpretation because no human ownership truth was used and source groups were not resolved.

**Next question:** When a page-diversified reviewer labels representative regions, what is the ANNOTATION purity of each cluster and special family, and does a 33x33 plus 97x97 context improve long-glyph and top-right-identifier discrimination without increasing false removal of genuine print?

# Appendix A. Evidence and project references

| **Reference**                                   | **Use in this report**                                                                                                                                                                                                    |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Notebook 59R3 outputs                           | cluster_model_selection.csv; cluster_profiles.csv; cluster_prevalence.csv; special_family_summary.csv; top_right_recurrence_fingerprints.csv; executive_summary.txt; notebook59R3_result.json; notebook59R3_verdict.json. |
| V7.5 Design Record                              | Separate annotation confidence from correction safety; authoritative mask boundary; exact preservation and outside-mask immutability.                                                                                     |
| V7.9.x Research Findings                        | Reject further deterministic threshold engineering; use a learned local ownership classifier with PRINT / ANNOTATION / UNCERTAIN.                                                                                         |
| Observed-Print Classifier Pre-Registration v1.1 | Truth firewall, allowed observable inputs, prohibited raw coordinates, page/source grouping and fail-closed promotion gates.                                                                                              |
| V8.0a/V8.0b Controlled Experiment Design        | 33x33 spatial topology experiment and deferred multi-scale context.                                                                                                                                                       |
| Production Retraining Policy                    | Event-driven challenger creation, authoritative labels only, frozen regression sets and non-automatic promotion.                                                                                                          |