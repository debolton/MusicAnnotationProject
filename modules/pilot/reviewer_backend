# modules/pilot/reviewer_backend.py

import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from modules.pilot.structural import (
    prepare_mask,
    telea_reconstruct,
    geometry_t80,
    surrounding_features,
)


class ReviewerBackend:

    def __init__(
        self,
        state,
        policy_bundle,
        decision_manager,
        edited_masks_dir,
        min_component_area=20,
    ):

        self.state = state

        self.policy = policy_bundle

        self.model = policy_bundle["model"]

        self.calibrator = (
            policy_bundle["calibrator"]
        )

        self.predictors = (
            policy_bundle["predictors"]
        )

        self.threshold = float(
            policy_bundle["threshold"]
        )

        self.decision_manager = (
            decision_manager
        )

        self.edited_masks_dir = Path(
            edited_masks_dir
        )

        self.min_component_area = (
            min_component_area
        )

    # --------------------------------------------------
    # Review state helpers
    # --------------------------------------------------

    def counts(self, page):

        return (
            self.decision_manager.counts(
                self.state[page][
                    "decisions"
                ]
            )
        )

    def first_unreviewed(
        self,
        page
    ):

        state = self.state[page]

        for _, row in (
            state["components"]
            .iterrows()
        ):

            cid = int(
                row.component
            )

            key = (
                self.decision_manager.uid(
                    page,
                    state["revision"],
                    cid,
                )
            )

            if (
                state["decisions"][key][
                    "status"
                ]
                == "unreviewed"
            ):

                return cid

        return None

    def queue_choices(
        self,
        page
    ):

        state = self.state[page]

        result = []

        for _, row in (
            state["components"]
            .iterrows()
        ):

            cid = int(
                row.component
            )

            key = (
                self.decision_manager.uid(
                    page,
                    state["revision"],
                    cid,
                )
            )

            status = (
                state["decisions"][key][
                    "status"
                ]
            )

            result.append(

                (
                    f"C{cid} [{status}]",
                    cid
                )

            )

        return result

    # --------------------------------------------------
    # Current reviewed page
    # --------------------------------------------------

    def current(
        self,
        page
    ):

        state = self.state[page]

        output = (
            state["observed"]
            .copy()
        )

        for _, row in (
            state["components"]
            .iterrows()
        ):

            cid = int(
                row.component
            )

            key = (
                self.decision_manager.uid(
                    page,
                    state["revision"],
                    cid,
                )
            )

            if (
                state["decisions"][key][
                    "status"
                ]
                == "accepted"
            ):

                component_mask = (
                    state["labels"]
                    == cid
                )

                output[
                    component_mask
                ] = state[
                    "proposal"
                ][
                    component_mask
                ]

        assert np.array_equal(

            output[
                state["mask"] == 0
            ],

            state["observed"][
                state["mask"] == 0
            ]

        )

        return output

    # --------------------------------------------------
    # Visualisation helpers
    # --------------------------------------------------

    @staticmethod
    def tint(
        base,
        layers
    ):

        if base.ndim == 2:

            rgb = cv2.cvtColor(
                base,
                cv2.COLOR_GRAY2RGB
            )

        else:

            rgb = base.copy()

        overlay = rgb.copy()

        alpha = np.zeros(
            rgb.shape[:2],
            np.float32
        )

        for mask, colour, opacity in layers:

            overlay[mask] = colour

            alpha[mask] = np.maximum(
                alpha[mask],
                opacity
            )

        return (

            rgb
            * (
                1
                - alpha[..., None]
            )

            +

            overlay
            * alpha[..., None]

        ).astype(np.uint8)

    def outlined_image(
        self,
        page,
        image,
        category="all",
        status="all",
    ):

        state = self.state[page]

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        colours = {

            "green":
                (40, 180, 65),

            "amber":
                (245, 165, 0),

            "red":
                (225, 40, 40),
        }

        for _, row in (
            state["components"]
            .iterrows()
        ):

            cid = int(
                row.component
            )

            key = (
                self.decision_manager.uid(
                    page,
                    state["revision"],
                    cid,
                )
            )

            component_status = (
                state["decisions"][key][
                    "status"
                ]
            )

            if (
                category != "all"
                and row.pilot_category
                != category
            ):
                continue

            if (
                status != "all"
                and component_status
                != status
            ):
                continue

            colour = colours[
                row.pilot_category
            ]

            cv2.rectangle(
                rgb,
                (
                    int(row.x),
                    int(row.y)
                ),
                (
                    int(
                        row.x
                        + row.bbox_w
                    ),
                    int(
                        row.y
                        + row.bbox_h
                    )
                ),
                colour,
                2,
            )

        return rgb

    def overview(
        self,
        page,
        category="all",
        status="all",
        view="current",
    ):

        state = self.state[page]

        base = {

            "current":
                self.current(page),

            "proposal":
                state[
                    "proposal"
                ],

            "original":
                state[
                    "observed"
                ],
        }[view]

        return self.outlined_image(
            page,
            base,
            category,
            status,
        )

    # --------------------------------------------------
    # Component diagnostics
    # --------------------------------------------------

    def component_diagnostic(
        self,
        page,
        component_id,
    ):

        state = self.state[page]

        if component_id is None:

            return (
                self.overview(
                    page
                ),
                "Select component",
            )

        row = state[
            "components"
        ]

        row = row[
            row.component.astype(int)
            == int(component_id)
        ]

        if row.empty:

            return (
                self.overview(
                    page
                ),
                "Stale component",
            )

        row = row.iloc[0]

        component_mask = (
            state["labels"]
            == int(component_id)
        )

        ys, xs = np.where(
            component_mask
        )

        x0 = max(
            0,
            xs.min() - 90
        )

        y0 = max(
            0,
            ys.min() - 90
        )

        x1 = min(
            state[
                "observed"
            ].shape[1],
            xs.max() + 90
        )

        y1 = min(
            state[
                "observed"
            ].shape[0],
            ys.max() + 90
        )

        original = cv2.cvtColor(

            state["observed"][
                y0:y1,
                x0:x1
            ],

            cv2.COLOR_GRAY2RGB

        )

        proposal = cv2.cvtColor(

            state["proposal"][
                y0:y1,
                x0:x1
            ],

            cv2.COLOR_GRAY2RGB

        )

        comparison = np.hstack(

            [
                original,
                np.full(
                    (
                        original.shape[0],
                        8,
                        3
                    ),
                    230,
                    np.uint8
                ),
                proposal,
            ]

        )

        key = (
            self.decision_manager.uid(
                page,
                state["revision"],
                component_id,
            )
        )

        status = (
            state["decisions"][key][
                "status"
            ]
        )

        info = (
            f"C{component_id} | "
            f"{row.pilot_category.upper()} | "
            f"{status} | "
            f"p_safe="
            f"{row.prob_structural_safe:.4f}"
        )

        return comparison, info

    # --------------------------------------------------
    # Decision handling
    # --------------------------------------------------

    def decide(
        self,
        page,
        component,
        status,
        notes,
    ):

        self.decision_manager.set_decision(
            self.state[page],
            page,
            component,
            status,
            notes,
        )

        self.decision_manager.persist(
            page,
            self.state[page],
            {
                "event":
                    "decision",

                "component":
                    int(component),

                "status":
                    status,
            },
        )

        return (
            f"C{component}: "
            f"{status}"
        )

    # --------------------------------------------------
    # Table
    # --------------------------------------------------

    def table(
        self,
        page
    ):

        state = self.state[page]

        rows = []

        for _, row in (
            state["components"]
            .iterrows()
        ):

            cid = int(
                row.component
            )

            key = (
                self.decision_manager.uid(
                    page,
                    state["revision"],
                    cid,
                )
            )

            rows.append({

                "component":
                    cid,

                "category":
                    row.pilot_category,

                "status":
                    state[
                        "decisions"
                    ][key][
                        "status"
                    ],

                "area_pixels":
                    int(row.area),

                "p_structural_safe":
                    round(
                        float(
                            row
                            .prob_structural_safe
                        ),
                        4,
                    ),
            })

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------
    # Approval
    # --------------------------------------------------

    def approve(
        self,
        page,
        output_path,
        notes=""
    ):

        counts = self.counts(
            page
        )

        if (
            counts["unreviewed"]
            or counts["deferred"]
        ):

            raise RuntimeError(

                "Page not ready for "
                "approval."

            )

        final_image = (
            self.current(page)
        )

        output_path = Path(
            output_path
        )

        cv2.imwrite(
            str(output_path),
            final_image
        )

        self.decision_manager.persist(
            page,
            self.state[page],
            {
                "event":
                    "final_approval",

                "notes":
                    notes,

                "path":
                    str(output_path),
            },
        )

        return final_image
