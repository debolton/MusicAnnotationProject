**MUSIC ANNOTATION PROJECT**

Future Direction of Travel

_Generative residual reconstruction after the V8.x observed-print programme_

**STATUS  
**Future research resource only. This document does not commission a model, alter V7.8/V8.x, authorise generative reconstruction, or assert that FLUX.1-Fill is suitable for deployment.

Project reference document | 20 October 2026

# 1. Proposal in one sentence

After the current V8.x observed-print thread is concluded, consider a separately governed generative residual-reconstruction feasibility study, with FLUX.1-Fill as one possible candidate, restricted to genuinely unsupported residual gaps after trusted observed print has been identified and preserved exactly.

**Recommended positioning  
**Keep V8.x and future generative reconstruction as separate programmes. V8.x decides which existing observed bytes are trustworthy PRINT and copies those bytes exactly. A later generative strand would synthesise only the residual area for which trusted observed evidence is unavailable.

# 2. Why this is a later, separate strand

The current project architecture is built around an authoritative intervention mask, exact observed-byte preservation for PRINT, conservative fallback for ANNOTATION and UNCERTAIN, clean truth used only after inference, and byte-for-byte immutability outside the mask. A generative model changes the scientific problem from classification/preservation to synthesis. Combining those questions in V8.4 would prevent clear attribution and weaken the existing regression contract.

| **Current V8.x question**                                                                                  | **Future generative question**                                                                                                    |
| ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Which observed pixels inside an authorised mask are genuine printed music and should be preserved exactly? | Where no trustworthy observed evidence remains, can a model reconstruct missing notation faithfully rather than merely plausibly? |
| Output is an exact copy of an observed byte at valid PRINT coordinates.                                    | Output is newly synthesised content within a residual reconstruction mask.                                                        |
| Primary risk is false annotation preservation.                                                             | Primary risk is hallucinated, omitted or topologically altered musical content.                                                   |
| Comparator and fallback are frozen V7.8 geometry_t80.                                                      | Comparator set should include frozen V7.8, Telea, biharmonic and the strongest non-generative restoration baseline.               |

# 3. Candidate technology and factual caveat

FLUX.1-Fill is proposed here only as a named candidate for later investigation. This project record does not verify a particular checkpoint, licence, hosting arrangement, API, weight format, reproducibility guarantee or deployment right. Those details can change and must be verified from authoritative documentation at the point when the study is commissioned.

- Do not treat the model name as a fixed technical specification.
- Freeze the exact model identifier, weights or service version, licence, inference implementation, scheduler/settings, random seed handling and software environment before evaluation.
- If deterministic replay cannot be demonstrated, treat output variability as part of the safety problem rather than hiding it through best-of-N selection.

# 4. Research hypothesis

**Primary hypothesis  
**A context-aware generative fill model may reconstruct curved, non-local or semantically constrained printed structures more effectively than local PDE or biharmonic inpainting when the model receives a tightly controlled residual mask and surrounding trusted musical context.

This is a proposal, not an established result. V8.3 showed that smaller residual gaps did not improve mean local MAE for the tested legacy methods. That negative result justifies considering a materially different model class, but it does not establish that a general generative model will be faithful enough for music notation.

# 5. Proposed programme boundary

| **Element**        | **Proposed rule**                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| Working title      | V9.0 Generative Residual Reconstruction Feasibility, or another new major strand clearly separated from V8.x. |
| Input authority    | Annotated/aligned image, authoritative mask, frozen observed-print decisions and trusted unchanged context.   |
| Generation mask    | Only the final residual gap after exact preservation decisions. Never the whole annotation mask by default.   |
| Protected pixels   | Outside-mask pixels and all trusted observed PRINT pixels remain byte-for-byte immutable.                     |
| Truth role         | Clean page used only for label construction and post-output scoring, never as an inference input.             |
| Operational status | Research-only. No automatic replacement of V7.8 and no production output authority.                           |
| Human review       | Mandatory for every generated candidate in the feasibility stage.                                             |

# 6. Core safety concern: plausibility is not fidelity

For sheet music, a visually convincing completion can still be materially wrong. An invented notehead, accidental, augmentation dot, articulation, beam connection, stem, barline or slur changes musical information. The principal evaluation question is therefore not "does the image look natural?" but "does the generated image recover the specific printed content supported by truth and context without inventing notation?"

- Photorealism, aesthetic preference and generic perceptual-image scores cannot be the primary safety endpoints.
- A clean-looking but musically altered result must be classified as failure, even if ordinary SSIM or visual preference improves.
- The evaluation must separate genuine recovery, harmless tonal variation, omitted print, invented print and topological reconnection errors.

# 7. Proposed phased feasibility design

1. Phase A - benchmark definition. Freeze candidate models, comparators, inference settings, mask rules, metrics, failure taxonomy and review protocol before outcome inspection.
2. Phase B - synthetic occlusion test. Apply controlled masks to clean pages so the exact missing content is known. Include staff, stem/barline, beam, notehead/compact glyph, slur/tie, hairpin, mixed and background strata.
3. Phase C - real-mask counterfactual test. Use authoritative annotation masks and registered clean truth, while preserving the same truth firewall and intervention boundary used by the existing project.
4. Phase D - repeatability and sensitivity. Repeat frozen inputs across declared seeds or deterministic settings, mask perturbations and context-window conditions. Report variation rather than selecting the most favourable sample.
5. Phase E - blinded musical review. Compare candidate reconstruction with frozen V7.8 and non-generative baselines without initially exposing model identity or clean truth.
6. Phase F - fail-closed decision. Continue only if explicit musical-content, catastrophic-error, immutability, source-stability and reproducibility gates pass.

# 8. Required comparators

| **Comparator**                                  | **Reason for inclusion**                                                                |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| Frozen V7.8 geometry_t80                        | Incumbent correction and fallback contract.                                             |
| Telea                                           | Established local inpainting baseline used by the project.                              |
| Biharmonic                                      | Established smooth-completion baseline.                                                 |
| Best historical hybrid                          | Tests whether routing plus conventional reconstruction remains competitive.             |
| Strong non-generative learned restoration model | Separates the value of learned restoration from the extra risk of generative synthesis. |
| No-change / leave unresolved                    | Essential safety baseline where synthesis is worse than retaining uncertainty.          |

# 9. Evaluation framework

| **Evidence family**        | **Required measures**                                                                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Pixel fidelity             | MAE/PSNR/SSIM within the authorised residual region, reported as secondary rather than decisive measures.                                      |
| Printed-ink fidelity       | Ink precision, recall and F1 with the frozen tolerance method.                                                                                 |
| Topology                   | Edge F1, skeleton recall/precision, connected components, endpoints, junctions, line continuity and thickness.                                 |
| Musical-content errors     | Unsupported noteheads, accidentals, dots, articulations, beams, stems, barlines, ties/slurs and connections; omissions of the same structures. |
| Catastrophic concentration | Component/page events where invented or omitted content is musically consequential, regardless of favourable average scores.                   |
| Immutability               | Zero changes outside the residual generation mask and zero changes to protected observed PRINT pixels.                                         |
| Repeatability              | Output variation across declared seeds/runs, including topological and musical-content disagreement.                                           |
| Source stability           | Results by page, score/source group, scan style, sparse-print/hard-negative status and semantic structure.                                     |
| Human judgement            | Blinded side-by-side review of actual corrected pages and focused crops, with "leave unresolved" allowed as the preferred result.              |

# 10. Suggested error taxonomy

| **Class**                     | **Interpretation**                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| G0 Faithful recovery          | Missing printed structure is restored without unsupported notation or protected-pixel change.                 |
| G1 Tonal-only deviation       | Geometry/content preserved; limited greyscale or paper-tone difference.                                       |
| G2 Benign geometric deviation | Minor non-musical shape variation with no change to encoded notation; requires explicit reviewer agreement.   |
| G3 Printed omission           | Expected printed pixels or connectivity remain missing.                                                       |
| G4 Unsupported insertion      | Generated printed-looking mark has no clean-truth support.                                                    |
| G5 Topological alteration     | Wrong connection, disconnection, junction, beam grouping, staff crossing, slur/tie path or symbol attachment. |
| G6 Boundary violation         | Any change outside the authorised residual mask or to protected observed PRINT.                               |
| G7 Unstable generation        | Material output disagreement across frozen repeated runs/seeds.                                               |
| G8 Indeterminate              | Registration or truth ambiguity prevents a reliable content judgement; fail closed.                           |

# 11. Proposed go/no-go principles

Exact numerical thresholds should be pre-registered only when a specific model, cohort and label method have been verified. The following are directional commissioning principles, not established project facts.

- Zero boundary violations and zero protected-PRINT changes.
- Zero unacceptable G4/G5 catastrophic musical-content events in pooled results and every eligible source group.
- Positive page-clustered benefit against frozen V7.8 and the best non-generative comparator.
- No aggregate result driven by a single page, source group or semantic category.
- Declared repeatability standard met without best-of-N cherry-picking.
- Non-trivial coverage across pages, sources and structural classes.
- Complete blinded review of all catastrophic candidates, top benefits, top regressions and unstable cases.
- Any insufficient sample or wide uncertainty interval yields "insufficient evidence", not a pass.

# 12. Prompting and conditioning restrictions

Prompt design can itself leak semantic assumptions or encourage invention. The primary experiment should minimise text prompting and use a frozen neutral instruction, if the selected implementation requires text at all. Do not supply the clean answer, human semantic label, page title, score identity or a description of the expected missing symbol.

- All context must come from observable page pixels, the authorised residual mask and frozen project artefacts.
- If multiple prompt formulations are compared, that is a separate pre-registered arm, not informal prompt tuning after failures are seen.
- Negative prompts or guidance settings must be frozen and replayable.

# 13. Governance and artefacts

- 01_pre_registration.json: exact models, licences, inference stack, cohort, masks, prompts/settings, seeds, comparators, metrics and gates.
- 02_model_and_licence_record/: downloaded or service artefact identity, SHA-256 where available, terms and access date.
- 03_input_lock.json: page/source roster, authoritative masks, residual masks and protected-pixel hashes.
- 04_generation_manifest.parquet: one row per page/mask/model/run/seed with complete settings and output hash.
- 05_reconstruction_outputs/: all candidate and comparator pages, with no silent best-sample selection.
- 06_quantitative_results/: pixel, ink, topology, content-error, source and uncertainty tables.
- 07_blinded_review/: queue, reviewer decisions and adjudication.
- 08_replay_and_immutability/: exact boundary checks and repeated-run comparison.
- 09_verdict.json and executive_summary.txt: first failed gate and explicit continuation/closure decision.
- upload/: curated final evidence package with file list and SHA-256 manifest.

# 14. Decision sequence

| **Outcome**                                               | **Decision**                                                                                       |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Model/version/licence or reproducibility cannot be frozen | Do not start the scientific comparison; record a feasibility block.                                |
| Any boundary or protected-pixel violation                 | Invalidate the implementation before interpreting image quality.                                   |
| Musical hallucination or topological catastrophe appears  | Reject that frozen candidate/rule. Do not hide it behind average image metrics.                    |
| Visually attractive but no reliable fidelity advantage    | Retain as an image-generation demonstration, not a correction method.                              |
| Better pixel metrics but worse musical-content errors     | Reject for project use.                                                                            |
| Safe but negligible coverage or benefit                   | Retain as low-coverage research; do not weaken gates.                                              |
| All feasibility gates pass                                | Freeze one candidate and commission fresh untouched evidence before any supervised pilot.          |
| Fresh evidence passes                                     | Eligible only for bounded supervised comparison. Production authority remains a separate decision. |

# 15. Relationship to the current roadmap

The recommended order is: complete the current V8.x preservation programme; retain V7.8 as fallback; freeze the meaning of trusted observed PRINT and unsupported residual gaps; then, if resources justify it, open a new major reconstruction strand. The later strand should not reopen V8.3 legacy continuation tuning and should not change V8.4 candidate selection.

**Roadmap guardrail  
**Preserve first; generate only where evidence is genuinely absent; evaluate musical truth rather than visual plausibility; fail closed to V7.8 or no-change when confidence is inadequate.

# 16. Discussion and caveats retained for future reviewers

- The reason to consider a generative model is not that V8.3 almost succeeded. V8.3 was negative. The reason is that a materially different model class may use wider learned context unavailable to local legacy methods.
- FLUX.1-Fill may prove unsuitable. A specialist document or notation restoration model may offer better faithfulness, repeatability or governance.
- Synthetic occlusions are necessary but not sufficient. They may be cleaner than real annotation overlap and must not substitute for real-mask evidence.
- Clean truth may itself contain registration uncertainty. Ambiguous labels must be excluded or assigned an explicit indeterminate class.
- A generative system can trade omissions for unsupported insertions. Those error types require asymmetric treatment because inserted notation can change musical meaning.
- Whole-page review matters. A locally plausible completion may create a wrong relationship to adjacent measures, voices, staffs or systems.
- General-purpose model updates can change behaviour. A service alias or product name is not a durable experiment identity.
- Licensing, privacy, data residency, model-access and cost issues require separate confirmation when the exact implementation is selected.

# 17. Recommendation

Retain this proposal as a future direction of travel, not an active V8.x task. Revisit the proposal only after the observed-print selector and residual-mask semantics are stable. At commissioning, begin with a model-and-governance verification note, then a pre-registered controlled benchmark. Include FLUX.1-Fill only if the exact available version can be frozen, lawfully used, replayed or its variability fully characterised, and evaluated without allowing synthesis outside the final residual mask.

Confidence: 87%. Confidence is high that generative residual reconstruction must remain separate from V8.x and must be evaluated for musical hallucination, topology and strict mask immutability. Confidence is lower that FLUX.1-Fill specifically will be appropriate, because this document intentionally does not verify a fixed model version, licence, deployment mode or domain-specific performance.

# 18. Source basis

- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx - authoritative-mask boundary, exact preservation, reconstruction only for ambiguity and separate audit axes.
- 10_Reconstruction_Benchmark.txt - clean-truth firewall, outside-mask immutability and conventional reconstruction comparators.
- V7_9_x_Research_Findings_and_Next_Steps.docx - deterministic continuation closure, learned observed-print direction and no generative reconstruction in the preservation layer.
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx - PRINT/ANNOTATION/UNCERTAIN rule, truth-free inference, source/page separation and fail-closed gates.
- V8_0a_Visual_Learning_and_V8_0b_Controlled_Experiment_Design 1.docx - controlled isolation of representation changes and explicit deferral of multi-scale redesign.
- V8_Next_Stage_Full_Pixel_Whole_Page_OOF_Reference.docx - whole-page outputs, exact V7.8 fallback and statement that V8 is not generative reconstruction.
- V8_2_Development_Whole_Page_Experiment_Decision_Record.docx - development/freeze/independent-stage separation and durable output packaging.
- Production_Retraining_Policy_YOLO_and_Structural_Safety.docx - frozen incumbent, challenger governance, event-driven change and preservation of failed experiments.
- V8.3 Residual-Gap Continuation Closure Record and Notebook 40 upload - negative legacy-reconstruction finding motivating a materially different future hypothesis.

# 19. Next question

What frozen evaluation set and notation-level error ontology should be created now, while the project still has clean registered truth, so that any future generative reconstruction model can be assessed without retrospectively inventing its safety criteria?