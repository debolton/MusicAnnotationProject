**V7.5 Design Record**

_From annotation detection to risk-aware correction_

Music Annotation Project | 11 October 2026

**Core decision: keep annotation confidence and correction safety as separate axes. High confidence that a mark is annotation does not imply high confidence that it can be removed safely.**

# 1. Why this design changed

The reconstruction work initially concentrated on what to do inside a known removal mask. V7.4.1 showed that unrestricted inpainting can damage observable printed music, while tonal suppression preserves structure but may retain dark pencil. That led to V7.5: preserve confidently observed printed geometry exactly, remove confidently non-structural annotation, and restrict reconstruction to ambiguous areas.

A subsequent design review clarified a second distinction. The trained YOLO model answers where annotation is likely to exist. The surrounding-area decision tree answers whether a proposed correction is safe in that local musical context. These are related inputs, but they are not interchangeable.

# 2. Existing project building blocks

- YOLO segmentation training produces a best checkpoint for annotation detection.
- The HITL confidence generator saves both a binary draft mask and a continuous pixel confidence map.
- The CPU reviewer displays confidence as a heatmap and allows additions and erasures before saving a corrected mask.
- Notebook 09 can prefer corrected masks and can fall back to YOLO mask generation, while keeping pixels outside the removal mask unchanged.
- The reconstruction benchmark separates background, staff, symbol and mixed regions and prohibits clean-target leakage into reconstruction.

# 3. The two-axis framework

| **Axis**              | **Question**                                                              | **Evidence family**                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Annotation confidence | How certain are we that the proposed area is annotation?                  | YOLO pixel confidence; instance confidence; overlap/tile agreement; hard-negative behaviour; later model calibration.                                           |
| Correction safety     | How certain are we that changing this area will not damage printed music? | Staff, stem, bar-line and compact-glyph continuity; geometry crossing; local ink density; mask depth; component shape; ambiguity and reconstruction dependence. |

# 4. Behavioural matrix for the future GUI

| **Annotation confidence** | **Correction safety** | **Proposed treatment**                                                                           |
| ------------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| High                      | High                  | Auto-accept only after thresholds have been validated on independent data.                       |
| High                      | Low or ambiguous      | Review. The system probably found annotation, but removal may cross musical structure.           |
| Low or medium             | High                  | Quick accept/reject. The correction itself is simple, but annotation identity remains uncertain. |
| Low                       | Low                   | Explicit review; candidate for future training data.                                             |

Important: V7.5 does not implement auto-acceptance. It records surrounding-area evidence so that later work can validate which factors predict safe correction. Any combined score or threshold remains a future, evidence-based step.

# 5. V7.5 routing principle

1. Use an existing controlled annotation-removal mask to define where intervention is permitted.
2. Inside the mask, copy confidently observed printed geometry byte-for-byte.
3. Replace confidently light, non-structural annotation pixels with a local paper estimate.
4. Use reconstruction only for the explicit ambiguous class.
5. Leave all pixels outside the removal mask byte-for-byte unchanged.
6. Record the surrounding-area evidence for each mask component and route, without yet converting it into a production confidence score.

# 6. Surrounding-area evidence now collected in V7.5

| **Evidence**              | **Interpretation**                                                    | **Purpose**                                                                                                               |
| ------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Horizontal support        | Dark horizontal geometry continues around or through the component.   | Evidence of staff lines or printed horizontal rules.                                                                      |
| Vertical support          | Narrow vertical geometry is present around or through the component.  | Evidence of stems, bar lines or vertical glyph strokes.                                                                   |
| Compact connected support | A compact dark component connects from outside the removal mask.      | Evidence of noteheads and small printed glyphs.                                                                           |
| Visible-edge density      | Edge density in a ring immediately outside the component.             | Measures structural complexity surrounding the correction.                                                                |
| Dark-ink density          | Fraction of nearby ring pixels below a dark threshold.                | Measures how much printed or dark material surrounds the proposed correction.                                             |
| Staff proximity           | Distance from the component to detected horizontal musical structure. | Flags correction close to staff geometry.                                                                                 |
| Mask depth                | Maximum distance from the component boundary into the mask.           | Large/deep masks contain less observable local evidence.                                                                  |
| Border contact            | Whether the component reaches a page boundary.                        | Boundary cases may have weaker surrounding evidence.                                                                      |
| Component geometry        | Area, bounding-box dimensions, aspect ratio and fill ratio.           | Supports later distinction between compact marks, strokes and broad masks.                                                |
| Routing fractions         | Preserve, remove and ambiguous proportions for each component.        | Shows how dependent the correction is on exact copying, whitening or reconstruction.                                      |
| Clean-truth diagnostics   | Post-hoc printed-ink fractions for validation only.                   | Labels which evidence patterns actually correspond to printed structure, without leaking clean truth into reconstruction. |

# 7. Data products added to V7.5

- \`surrounding_area_evidence_by_component.csv\`: one record per mask component and candidate threshold, including geometry, context and routing evidence.
- \`surrounding_area_evidence_summary.csv\`: aggregate distributions by candidate threshold.
- \`surrounding_evidence_truth_diagnostics.csv\`: post-hoc clean-target diagnostics used only to assess evidence quality.
- Existing routing maps and route truth summaries remain available for visual and quantitative validation.

# 8. Guardrails

- Corrected or controlled masks remain the boundary of permissible intervention. Surrounding-area analysis does not search the whole page for grey pixels to remove.
- YOLO confidence must not override a reviewed corrected mask.
- Clean targets are used only after reconstruction for scoring and diagnostics.
- No combined decision-confidence score is emitted in V7.5.
- No auto-accept threshold is set until independent validation demonstrates acceptable false-positive and structural-risk behaviour.
- Improved YOLO models should move more cases through fixed safety gates; the safety standard should not be weakened to increase automation.

# 9. Future Notebook 09 direction

The eventual visual Notebook 09 should present proposed corrections by exception rather than requiring a full-page manual pass. A likely progression is:

7. Current: predict, manually correct the mask, reconstruct, then inspect.
8. Intermediate: predict and reconstruct, auto-group regions by annotation confidence and correction safety, then show uncertain or risky regions for accept/correct review.
9. Mature: auto-accept independently validated high/high regions, queue exceptional regions, and preserve manual work primarily for difficult cases and model-training feedback.

The eventual audit record should preserve the YOLO confidence map, authoritative annotation mask, surrounding-area evidence/routing map, proposed corrected image, model and settings fingerprints, and the human decision where review occurred.

# 10. Open questions for later validation

- Which surrounding-area evidence variables best predict damage to staff, stem, notehead and glyph structure?
- Should evidence be calibrated at pixel, connected-component or proposed-correction-region level?
- How should overlapping-tile agreement be retained alongside maximum YOLO confidence?
- What independent test set is large and diverse enough to justify auto-acceptance?
- What failure rate is acceptable for plain-paper corrections versus notation-crossing corrections?
- When should low correction safety lead to "leave unchanged" rather than reconstruction?

# 11. Current decision

Run the revised V7.5 as a geometry-only reconstruction benchmark with richer evidence logging. Do not combine the evidence with YOLO confidence or use it for automatic acceptance yet. Use the collected component-level evidence to design and validate the later two-axis policy.