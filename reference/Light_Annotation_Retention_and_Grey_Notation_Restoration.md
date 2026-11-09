**MUSIC ANNOTATION PROJECT**

Light Annotation Retention and Grey Notation Restoration

_Separate research and quality record | 9 November 2026_

**CORE DECISION**  
Capture two distinct quality problems outside the current V8.8 operationalisation scope: (1) faint annotation that remains visible after correction, and (2) genuine grey musical notation whose original tone should be restored or preserved rather than darkened. These must be evaluated separately from structural safety and separately from each other.

Status: project research backlog and commissioning reference. This document does not modify the frozen SAFE_REMOVE policy, M5_REMOVE, reconstruction semantics, or the V8.8 release gates.

# 1. Clarification of the requirement

The issue is broader than preventing damage to dark printed structure. A corrected page can be structurally safe and still be visually unsatisfactory in two opposite ways:

- Residual light annotation: pale pencil, erased strokes, smudges or low-contrast marks remain perceptible when the intended result is local paper.
- Grey notation tone loss: authentic printed notation is present but light or degraded; correction makes it darker, heavier or more uniform than the source evidence supports.

The desired treatment of genuine grey notation is tonal fidelity. Where trustworthy observed notation exists, preserve or reconstruct its appropriate grey appearance. Do not equate restoration with converting it to black.

# 2. Why this is a separate document

The active V8.8 programme is an engineering qualification of the frozen REMOVE-only mechanism. It expressly excludes policy retraining, SAFE_PROTECT re-entry and changes to reconstruction semantics. The final V8.7 record also parks SAFE_PROTECT and defers new research routes until V8.8 has established the operational baseline.

- Adding faint-residue cleanup now would change the removal-quality objective and possibly the treatment applied to SAFE_REMOVE regions.
- Adding grey-tone restoration now would change the reconstruction or preservation objective and could alter pixels beyond the registered REMOVE-only causal comparison.
- Combining both would obscure attribution because annotation suppression and notation tone restoration have different desired outputs and different failure costs.

# 3. Proposed three-axis quality model

| **Axis**                | **Question**                                                     | **Primary failure**                                                | **Preferred evidence**                                                             |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Structural safety       | Did correction preserve genuine musical content?                 | Missing, broken or altered notation.                               | Ink/edge/topology metrics, component review and clean truth after lock.            |
| Annotation cleanliness  | Did unwanted annotation remain visibly present?                  | Faint pencil, smudge or tonal residue survives.                    | Residual-contrast maps, local paper deviation, blinded perceptual ratings.         |
| Notation tonal fidelity | Was genuine light or grey notation reproduced at the right tone? | Darkening, thickening, flattening or bleaching authentic notation. | Observed/clean tonal agreement, stroke-profile fidelity and notation-aware review. |

# 4. Research stream A: light-annotation retention

Objective: detect and reduce perceptually visible annotation residue after the structurally safe correction has been produced, without whitening genuine low-contrast print or altering unaffected paper texture.

- Population: pixels or small connected regions inside the authoritative intervention mask that follow a removal/background treatment but remain measurably different from trustworthy local paper.
- Primary endpoint: perceptually material residual annotation, assessed by blinded page/crop review and a local contrast or paper-deviation measure.
- Safety endpoint: genuine printed ink incorrectly treated as residue. This remains an asymmetric, fail-closed error.
- Comparator: frozen V8.8 output. The experiment changes only the residual-cleanup layer.
- Default action: if the residue classifier or tonal estimate is uncertain, retain frozen V8.8 output and route the case to review.

## Candidate measurements

- Difference from a locally estimated paper surface, measured in absolute intensity and local contrast.
- Residual connected-component size, edge coherence and persistence across neighbouring pixels.
- Blinded severity categories such as invisible, acceptable trace, noticeable residue, distracting residue and musically confusing residue.
- Page and source-group concentration, because scanner tone and paper ageing may mimic light annotation.

# 5. Research stream B: grey musical notation

Objective: preserve or restore authentic low-contrast notation with tonal fidelity. The success criterion is not maximum darkness. It is faithful recovery of the notation as supported by observed evidence and, during evaluation only, clean truth.

- Exact preservation case: where trustworthy grey notation is directly observed, copy its original byte or local tonal profile rather than binarising or darkening it.
- Reconstruction case: where notation is partially obscured, estimate missing tone consistently with neighbouring stroke intensity and paper background, while preserving topology.
- Primary failure: a genuine grey line or symbol becomes blacker, thicker, sharper or more uniform than the source notation.
- Opposite failure: authentic notation is over-whitened or left incomplete because it is mistaken for annotation residue.
- Default action: uncertainty retains the frozen incumbent treatment or routes to review. No invented dark stroke is preferable to unsupported tonal enhancement.

## Candidate measurements

- Mean and distribution of intensity error on genuine notation pixels, not only binary ink recall.
- Stroke cross-section agreement: centre tone, edge tone, width and gradient compared with adjacent visible notation.
- Local contrast ratio relative to surrounding paper and nearby notation of the same staff or symbol.
- Topology and musical-structure continuity, retained alongside tonal metrics so a visually plausible grey patch cannot hide a structural error.
- Blinded reviewer judgement: faithful tone, too dark, too light, thickened, blurred, structurally wrong or no material difference.

# 6. Experimental separation and order

| **Phase** | **Experiment**                          | **Rule**                                                                                                                |
| --------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 1         | Audit the frozen V8.8 output            | Create a labelled error taxonomy and quantify how often light residue and grey-tone errors occur. No correction change. |
| 2         | Light-residue cleanup challenger        | Change only the post-removal cosmetic treatment. Hold structural preservation and reconstruction fixed.                 |
| 3         | Grey-notation tonal-fidelity challenger | Change only tonal preservation/restoration for trusted notation. Hold annotation cleanup fixed.                         |
| 4         | Joint page-level evaluation             | Combine only challengers that separately pass. Re-run full structural, cleanliness and tonal-fidelity gates.            |
| 5         | Independent and supervised validation   | Use fresh role-locked sources, blinded review and explicit rollback. No threshold changes after unblinding.             |

# 7. Required audit artefacts

- Frozen V8.8 corrected page and identity.
- Authoritative mask, M3, SAFE_REMOVE and M5_REMOVE as separate artefacts.
- Light-residue probability or severity map and any cleanup decision map.
- Grey-notation confidence, tonal target/estimate and final tone-change map.
- Before/after intensity profiles for selected notation strokes.
- Difference maps separated by purpose: annotation cleanup, tone restoration and structural reconstruction.
- Blinded review record, page/source-group summaries, failure queues, software identity and rollback information.

# 8. Non-negotiable guardrails

- No pixel outside the authoritative intervention boundary changes.
- No clean image or truth-derived value enters inference. Clean truth is evaluation-only after outputs are written and locked.
- No global contrast enhancement, thresholding or blackening of notation is permitted as a shortcut.
- Annotation cleanliness cannot be improved by erasing uncertain grey structure.
- Tonal fidelity cannot be improved by retaining annotation because it is also grey.
- Any combined system must preserve separate probabilities, decisions and reason codes for removal cleanliness and notation tone.
- A pass on structural safety does not imply a pass on cosmetic cleanliness or tonal fidelity.

# 9. Immediate recommended action

Add a read-only review stage after the V8.8 supervised pilot outputs are available. The first notebook or audit should not modify pages. It should identify representative examples, assign the two error types separately, quantify frequency and severity by page/source group, and produce a pre-registration proposal for the first single-variable challenger.

**Decision statement**  
Retain V8.8 unchanged as the operational baseline. Commission future annotation-cleanliness and grey-notation tonal-fidelity work as two separately pre-registered research streams. Their shared principle is fidelity, not darkness: remove residual annotation when supported, but preserve or restore genuine light notation at the tone justified by the evidence.

Confidence: 96%. The need to separate these objectives follows directly from the current project architecture: V8.8 is REMOVE-only operationalisation, structural safety and cosmetic quality are distinct, and observed print preservation is defined as exact copying rather than enhancement. The remaining uncertainty is empirical: the prevalence and severity of these two failure classes must be measured on the completed V8.8 pilot outputs before numerical gates are frozen.

Next question: What severity rubric and fresh review cohort should be frozen for the read-only audit of residual light annotation and grey-notation tonal fidelity?

# Source basis

- V8_7_Final_Decision_and_V8_8_Direction.docx: V8.8 is REMOVE-only operationalisation; SAFE_PROTECT is parked; new research routes are deferred and isolated.
- V8_8_Operationalisation_Workplan.docx: no retraining or reconstruction change in V8.8; exact fallback and separate audit artefacts are required.
- V7_5_Design_Record_Annotation_Confidence_and_Correction_Safety.docx: preserve observed print exactly, remove confident non-structural annotation, reconstruct ambiguity only.
- 12_V7_7_Asymmetric_Structural_Risk_Calibration 1.txt: structural safety and cosmetic residue are separate outcomes.
- Observed_Print_Classifier_PreRegistered_Commissioning_Specification_v1.1.docx and V8 whole-page records: exact observed-byte preservation, truth firewall, fail-closed uncertainty and page/source-group evaluation.