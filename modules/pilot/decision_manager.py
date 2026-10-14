# modules/pilot/decision_manager.py

import json
import hashlib

import numpy as np

from pathlib import Path
from datetime import datetime, timezone


class DecisionManager:

    def __init__(
        self,
        policy_id,
        threshold,
        decisions_dir,
        protected_prefix_sha256,
    ):

        self.policy_id = policy_id
        self.threshold = threshold

        self.decisions_dir = Path(
            decisions_dir
        )

        self.protected_prefix_sha256 = (
            protected_prefix_sha256
        )

    # --------------------------------------------------
    # Component identity
    # --------------------------------------------------

    @staticmethod
    def uid(
        page,
        revision,
        component
    ):

        return (
            f"{Path(page).stem}:"
            f"r{revision}:"
            f"c{int(component)}"
        )

    # --------------------------------------------------
    # Array signatures
    # --------------------------------------------------

    @staticmethod
    def array_signature(array):

        return hashlib.sha256(

            np.ascontiguousarray(
                array
            ).tobytes()

            +

            str(array.shape).encode()

            +

            str(array.dtype).encode()

        ).hexdigest()

    # --------------------------------------------------
    # Component signatures
    # --------------------------------------------------

    def component_signature(
        self,
        labels,
        component_id,
        proposal,
        routes,
    ):

        mask = (
            labels
            == int(component_id)
        )

        ys, xs = np.where(mask)

        if not len(xs):

            return None

        x0 = xs.min()
        x1 = xs.max() + 1

        y0 = ys.min()
        y1 = ys.max() + 1

        arrays = [

            mask[
                y0:y1,
                x0:x1
            ].astype(np.uint8),

            proposal[
                y0:y1,
                x0:x1
            ],

            routes["preserve"][
                y0:y1,
                x0:x1
            ].astype(np.uint8),

            routes["remove"][
                y0:y1,
                x0:x1
            ].astype(np.uint8),

            routes["ambiguous"][
                y0:y1,
                x0:x1
            ].astype(np.uint8),
        ]

        return (

            int(x0),
            int(y0),
            int(x1),
            int(y1),

            *[
                self.array_signature(a)
                for a in arrays
            ]

        )

    # --------------------------------------------------
    # New decision state
    # --------------------------------------------------

    def fresh_decisions(
        self,
        frame,
        page,
        revision,
    ):

        decisions = {}

        for _, row in frame.iterrows():

            component = int(
                row.component
            )

            decisions[
                self.uid(
                    page,
                    revision,
                    component,
                )
            ] = {

                "status":
                    "unreviewed",

                "notes":
                    "",

                "history":
                    [],
            }

        return decisions

    # --------------------------------------------------
    # Snapshots
    # --------------------------------------------------

    def snapshot_components(
        self,
        state,
        page,
    ):

        result = {}

        revision = state[
            "revision"
        ]

        for _, row in (

            state[
                "components"
            ].iterrows()

        ):

            component = int(
                row.component
            )

            signature = (
                self.component_signature(
                    state["labels"],
                    component,
                    state["proposal"],
                    state["routes"],
                )
            )

            key = self.uid(
                page,
                revision,
                component,
            )

            result[
                signature
            ] = (

                component,

                state[
                    "decisions"
                ][key].copy()

            )

        return result

    # --------------------------------------------------
    # Preservation logic
    # --------------------------------------------------

    def preserve_decisions(
        self,
        page,
        old_state,
        new_frame,
        new_labels,
        new_proposal,
        new_routes,
        new_revision,
    ):

        old_snapshot = (
            self.snapshot_components(
                old_state,
                page,
            )
        )

        new_decisions = (
            self.fresh_decisions(
                new_frame,
                page,
                new_revision,
            )
        )

        superseded = []

        matched = set()

        preserved = 0

        for _, row in (
            new_frame.iterrows()
        ):

            component = int(
                row.component
            )

            signature = (
                self.component_signature(
                    new_labels,
                    component,
                    new_proposal,
                    new_routes,
                )
            )

            if (
                signature
                not in old_snapshot
            ):
                continue

            (
                old_component,
                old_decision,
            ) = old_snapshot[
                signature
            ]

            key = self.uid(
                page,
                new_revision,
                component,
            )

            new_decisions[
                key
            ] = {

                **old_decision,

                "history":

                    list(
                        old_decision.get(
                            "history",
                            [],
                        )
                    )

                    +

                    [{

                        "event":
                            "preserved_after_mask_edit",

                        "from_revision":
                            old_state[
                                "revision"
                            ],

                        "from_component":
                            old_component,

                        "utc":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),
                    }]
            }

            preserved += 1

            matched.add(
                signature
            )

        for (
            signature,
            value,
        ) in old_snapshot.items():

            if (
                signature
                in matched
            ):
                continue

            (
                old_component,
                decision,
            ) = value

            superseded.append({

                "uid":
                    self.uid(
                        page,
                        old_state[
                            "revision"
                        ],
                        old_component,
                    ),

                **decision,

                "status":
                    "superseded",

                "superseded_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
            })

        return {

            "decisions":
                new_decisions,

            "superseded":
                superseded,

            "preserved":
                preserved,
        }

    # --------------------------------------------------
    # Counts
    # --------------------------------------------------

    @staticmethod
    def counts(
        decisions
    ):

        values = [

            value["status"]

            for value

            in decisions.values()

        ]

        return {

            "unreviewed":
                values.count(
                    "unreviewed"
                ),

            "accepted":
                values.count(
                    "accepted"
                ),

            "left_unchanged":
                values.count(
                    "left_unchanged"
                ),

            "deferred":
                values.count(
                    "deferred"
                ),
        }

    # --------------------------------------------------
    # Decisions
    # --------------------------------------------------

    def set_decision(
        self,
        state,
        page,
        component,
        status,
        notes,
    ):

        revision = state[
            "revision"
        ]

        key = self.uid(
            page,
            revision,
            component,
        )

        record = state[
            "decisions"
        ][key]

        record.update({

            "status":
                status,

            "notes":
                notes,

            "utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        })

        record[
            "history"
        ].append({

            "status":
                status,

            "notes":
                notes,
        })

        return record

    # --------------------------------------------------
    # Persist review state
    # --------------------------------------------------

    def persist(
        self,
        page,
        state,
        event,
    ):

        output = {

            "page":
                page,

            "revision":
                state[
                    "revision"
                ],

            "policy_id":
                self.policy_id,

            "threshold":
                self.threshold,

            "protected_prefix_sha256":
                (
                    self
                    .protected_prefix_sha256
                ),

            "decisions":
                state[
                    "decisions"
                ],

            "superseded":
                state.get(
                    "superseded",
                    [],
                ),

            "context":
                state.get(
                    "context",
                    {},
                ),

            "event":
                event,

            "updated_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        output_path = (

            self.decisions_dir
            /
            f"{Path(page).stem}"
            "_review_state.json"

        )

        output_path.write_text(

            json.dumps(
                output,
                indent=2,
                default=str,
            )

        )

        return output_path
