# modules/pilot/pilot_reviewer.py

from pathlib import Path

from modules.pilot.policy_bundle import PolicyBundle
from modules.pilot.regression_gate import RegressionGate
from modules.pilot.hitl_resolution import HITLResolver
from modules.pilot.proposal_builder import ProposalBuilder
from modules.pilot.decision_manager import DecisionManager
from modules.pilot.reviewer_backend import ReviewerBackend
from modules.pilot.reviewer_ui import ReviewerUI


class PilotReviewer:

    def __init__(
        self,
        config
    ):
        self.config = config

    def build(self):

        policy = PolicyBundle(
            self.config.root
        ).load()

        gate = RegressionGate(
            project_root=self.config.root,
            run_dir=self.config.current_run,
            canonical_structural_block_sha256=
            self.config.canonical_hash
        )

        regression_result = gate.execute()

        resolver = HITLResolver(
            self.config.review_audit_dir,
            self.config.draft_audit_dir
        )

        builder = ProposalBuilder(
            resolver=resolver,
            policy_bundle=policy,
            proposals_dir=self.config.proposals_dir,
            mask_output_dir=self.config.mask_output_dir,
            routing_dir=self.config.routing_dir,
            audit_dir=self.config.audit_dir
        )

        pages = {}

        for source in self.config.incoming_pages():

            pages[source.name] = (
                builder.build(source)
            )

        decision_manager = (
            DecisionManager(
                policy_id=
                policy["policy"]["policy_id"],

                threshold=
                policy["threshold"],

                decisions_dir=
                self.config.decisions_dir,

                protected_prefix_sha256=
                self.config.protected_prefix_sha256
            )
        )

        backend = ReviewerBackend(
            state=pages,
            policy_bundle=policy,
            decision_manager=
            decision_manager,
            edited_masks_dir=
            self.config.edits_dir
        )

        return ReviewerUI(
            backend,
            self.config.final_dir
        )
