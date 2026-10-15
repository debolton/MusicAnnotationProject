# pilot_app.py

import json
from pathlib import Path

from modules.pilot.policy_bundle import PolicyBundle
from modules.pilot.hitl_resolution import HITLResolver
from modules.pilot.proposal_builder import ProposalBuilder
from modules.pilot.decision_manager import DecisionManager
from modules.pilot.reviewer_backend import ReviewerBackend
from modules.pilot.reviewer_ui import ReviewerUI
from modules.pilot.regression_gate import RegressionGate

CONFIG_PATH = 'project_config.json'


def load_config():
    path = Path(CONFIG_PATH)
    if not path.exists():
        raise RuntimeError(f'Missing configuration file: {path}')
    return json.loads(path.read_text())


def main():
    config = load_config()

    root = Path(config['project_root'])

    policy = PolicyBundle(
        project_root=root
    ).load()

    run_dir = root / 'production_pilot' / 'runs' / 'pilot_runtime'
    run_dir.mkdir(parents=True, exist_ok=True)

    gate = RegressionGate(
        project_root=root,
        run_dir=run_dir,
        structural_block_sha256='cce24d65e8866b086956855b55e9a5104383c8b936629c5035354968c929b04c'
    )

    gate.execute()

    resolver = HITLResolver(
        review_audit_dir=root / 'human_in_the_middle' / 'audit',
        draft_audit_dir=root / 'human_in_the_middle' / 'draft_audit'
    )

    proposals_dir = run_dir / 'proposals'
    masks_dir = run_dir / 'masks'
    routing_dir = run_dir / 'routing'
    audit_dir = run_dir / 'audit'
    decisions_dir = run_dir / 'decisions'
    edits_dir = run_dir / 'edited_masks'
    final_dir = run_dir / 'final'

    for p in [proposals_dir, masks_dir, routing_dir, audit_dir, decisions_dir, edits_dir, final_dir]:
        p.mkdir(parents=True, exist_ok=True)

    builder = ProposalBuilder(
        resolver=resolver,
        policy_bundle=policy,
        proposals_dir=proposals_dir,
        mask_output_dir=masks_dir,
        routing_dir=routing_dir,
        audit_dir=audit_dir
    )

    incoming = root / 'production_pilot' / 'incoming'

    state = {}

    for source in sorted(incoming.iterdir()):
        if source.is_file():
            state[source.name] = builder.build(source)

    decision_manager = DecisionManager(
        policy_id=policy['policy']['policy_id'],
        threshold=policy['threshold'],
        decisions_dir=decisions_dir,
        protected_prefix_sha256='6cf870d5ef07964951decdab89df119d43c781f0b595ac993c561eec6d8f34e2'
    )

    backend = ReviewerBackend(
        state=state,
        policy_bundle=policy,
        decision_manager=decision_manager,
        edited_masks_dir=edits_dir
    )

    ui = ReviewerUI(
        backend=backend,
        final_dir=final_dir,
        public_link=True
    )

    ui.launch()


if __name__ == '__main__':
    main()
