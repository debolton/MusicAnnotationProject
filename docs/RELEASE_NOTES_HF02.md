# HF-02 release notes
- Frozen predecessor: HF-01, upload → YOLO → draft/confidence → V7.8 proposal.
- Added Notebook 07 stage before correction: human mask correction and explicit authorisation.
- Added Notebook 09 stage: V7.7F hash/replay verification, canonical V7.8 proposal, green/amber/red evidence, accept/leave/defer and final approval.
- Auto-accept remains disabled.

## Deployment caveat
Hugging Face ephemeral process memory means in-progress review state is session-scoped. Durable cross-restart audits require HF persistent storage or external object storage. The original Notebook 07 custom JavaScript canvas is not byte-identical here; the backend semantics and workflow ordering are preserved using Gradio ImageEditor.
