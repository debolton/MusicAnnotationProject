V8.2 / V8.2a Findings and V8.2b Decision Record

Music Annotation Project | Development research record | 17 October 2026

**DECISION: Build V8.2b as a read-only spatial probability-band localisation and route/support audit. Reuse the exact saved V8.2 OOF probabilities and V8.2a evidence; do not retrain or recalibrate. Treat 0.90 as an exploratory automatic-candidate tier and 0.80–0.90 as review-only evidence, not as a promoted policy.**

# 1. Background

V8.2 was commissioned as a clean development-only whole-page experiment after V8.1 mixed development testing with later promotion-grade artefact reconstruction. V8.2 retrained the unchanged 33×33 CNN under source-group out-of-fold evaluation, persisted fold models and isotonic calibrators, scored all eligible held-out pixels, and generated whole corrected pages against V7.8.

V8.2a then reused the saved V8.2 OOF probabilities without retraining. It copied and hash-inventoried the V8.2 record, swept coordinated threshold/margin pairs, retained the optional connected-run rule, and added red/green/amber decision overlays.

# 2. V8.2 outcome

V8.2 was operationally safe but too conservative: only 124 PRINT pixels were emitted from 219,624 eligible pixels. Of these, 75 were PRINT truth, 0 were ANNOTATION truth, and 49 were truth-UNCERTAIN. Benefit was concentrated in two pages and two source groups. The whole-page comparisons therefore appeared almost unchanged from V7.8.

The integrity controls passed: zero outside-mask drift, zero non-PRINT V7.8 regression, zero catastrophic components, source-group OOF separation, and durable model/calibrator persistence.

# 3. Why the V8.2a overlays changed the diagnosis

The decision overlays showed extensive red and amber regions but almost no green. This established that the lack of page-scale difference was caused by very low PRINT coverage, not by a rendering problem. Lowering the primary exploratory threshold from 0.995 to 0.980 produced almost no additional coverage.

The threshold sweep showed that the issue was not simply a slightly over-conservative threshold. Material coverage appeared only as the effective threshold approached 0.90 and below, where truth-UNCERTAIN and false-annotation selections also increased.

# 4. Threshold evidence

| Threshold | Selected | PRINT truth | ANNOTATION truth | UNCERTAIN truth | Benefiting pages | Benefiting groups | Utility |
| --------- | -------- | ----------- | ---------------- | --------------- | ---------------- | ----------------- | ------- |
| 0.98      | 127      | 75          | 2                | 50              | 2                | 2                 | 65      |
| 0.95      | 193      | 98          | 2                | 93              | 2                | 2                 | 88      |
| 0.90      | 470      | 249         | 14               | 207             | 5                | 3                 | 179     |
| 0.85      | 906      | 412         | 54               | 440             | 8                | 5                 | 142     |
| 0.80      | 2,525    | 1,107       | 156              | 1,262           | 14               | 9                 | 327     |

Interpretation: 0.98 is effectively unchanged from the registered operating point. The 0.95 band adds mostly truth-UNCERTAIN material and does not broaden benefiting pages/source groups. The 0.90 band is the first region with broader genuine benefit. At 0.85 and 0.80, coverage grows strongly but false annotation and truth-UNCERTAIN selections also rise.

# 5. Probability-distribution finding

The saved quantiles show meaningful discrimination rather than a completely ineffective classifier. PRINT-truth probability has median approximately 0.497, 75th percentile approximately 0.825, 90th percentile 0.850, 95th percentile approximately 0.923, and 97.5th percentile approximately 0.962. Annotation truth remains much lower for most pixels, but truth-UNCERTAIN occupies much of the 0.80–0.95 region.

The principal unresolved question is spatial ownership: whether candidates in the 0.80–0.95 range form continuous printed structures supported by V7.8 route/topology evidence, or are dominated by registration-boundary and mixed-evidence locations.

# 6. Singleton rule

The page-global connected-run rule should be retained. At 0.80, disabling it gained only 22 additional true PRINT pixels while adding 20 false annotation and 120 truth-UNCERTAIN pixels, reducing utility from 327 to 249. The rule is therefore not the principal cause of low green coverage.

# 7. V8.2b experiment contract

- Reuse the exact V8.2 OOF raw and calibrated probabilities embedded in the completed V8.2a record.
- Do not retrain the CNN, refit isotonic calibration, alter folds, change channels, or reinterpret labels.
- Copy the V8.2 models, calibrators, inputs, audit files and prediction tables into the V8.2b run and verify recorded hashes.
- Partition probabilities into bands A (>=0.995), B (0.95–0.995), C (0.90–0.95), D (0.85–0.90), E (0.80–0.85), F (0.50–0.80), and G (<0.50).
- Generate an exploratory 0.90 corrected page while retaining V7.8 everywhere except final connected 0.90+ candidate pixels.
- Present 0.80–0.90 pixels as a review-only tier; do not copy them into the corrected page.
- Join each candidate to authoritative-mask component identity, mask depth, V7.8 preserve/remove/ambiguous route, horizontal/vertical/compact support and connected band-run size.
- Produce page, source-group, route, support and component summaries plus confidence heatmaps and post-hoc truth overlays.
- Keep truth overlays clearly marked as post-hoc development diagnostics and never use them for inference or page generation.

# 8. V8.2b principal outputs

- pixel_probability_band_diagnostics.parquet / .csv
- band_summary_global.csv, band_summary_by_page.csv, band_summary_by_source_group.csv
- route_context_band_analysis.csv and support_map_band_analysis.csv
- component_band_analysis.csv and review_candidates_ge_080.parquet
- threshold_policy_summary.csv and incremental_threshold_bands.csv
- probability_heatmap.png, threshold_band_overlay.png and tiered_policy_overlay.png per page
- posthoc_truth_overlay.png, route_context_overlay.png and support_context_overlay.png per page
- new_at_090_vs_095.png, new_at_085_vs_090.png and new_at_080_vs_085.png per page
- auto090_most_beneficial, false-heavy, uncertain-heavy and review-tier contact sheets
- experiment_identity.json, source_records.json and copied_source_artifact_inventory.csv

# 9. Interpretation boundary

V8.2b is retrospective development diagnosis. It does not freeze 0.90, authorise automatic correction, or constitute independent promotion evidence. If V8.2b reveals a coherent generalisable morphology that can resolve uncertainty without weakening the registered safety envelope, that finding should motivate one separately declared controlled challenger experiment.

# 10. Working-practice conclusion

The V8.2 lineage now follows the intended order: development experiment, focused diagnostic audit, then—only if justified—one controlled new challenger. Every run is self-contained enough to prevent future iteration chasing: its source record, models, calibrators, schema, predictions, policies and generated evidence are copied or content-addressed inside the new run.

# 11. Source basis

- 28_V8_2_Development_Whole_Page_Experiment.ipynb / text export
- 29_V8_2a_Policy_Sensitivity_Threshold_Sweep_and_Overlay_Audit.ipynb / text export
- V8_2_Development_Whole_Page_Experiment_Decision_Record.docx
- V8_Next_Stage_Full_Pixel_Whole_Page_OOF_Reference.docx
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx
- V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx
- probability_quantiles.csv and threshold_sweep_summary.csv from the completed V8.2a run

Confidence: 97%. The pooled threshold composition, quantiles and singleton comparison are explicit in the completed V8.2a outputs. The remaining uncertainty is spatial, which is exactly the question V8.2b is designed to resolve.