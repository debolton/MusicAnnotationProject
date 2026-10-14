# modules/pilot/proposal_builder.py

import json
import cv2
import numpy as np
import pandas as pd

from pathlib import Path

from modules.pilot.structural import (
    prepare_mask,
    telea_reconstruct,
    geometry_t80,
    surrounding_features,
)


class ProposalBuilder:

    LEGACY_ALIASES = {
        "component_pixels_x": "component_pixels",
        "component_pixels_y": "component_pixels",
    }

    def __init__(
        self,
        resolver,
        policy_bundle,
        proposals_dir,
        mask_output_dir,
        routing_dir,
        audit_dir,
        min_component_area=20,
    ):

        self.resolver = resolver

        self.policy = policy_bundle

        self.model = policy_bundle["model"]
        self.calibrator = policy_bundle["calibrator"]

        self.predictors = (
            policy_bundle["predictors"]
        )

        self.threshold = float(
            policy_bundle["threshold"]
        )

        self.proposals_dir = Path(
            proposals_dir
        )

        self.mask_output_dir = Path(
            mask_output_dir
        )

        self.routing_dir = Path(
            routing_dir
        )

        self.audit_dir = Path(
            audit_dir
        )

        self.min_component_area = (
            min_component_area
        )

    # --------------------------------------------------
    # Predictor preparation
    # --------------------------------------------------

    def apply_predictor_aliases(
        self,
        frame,
    ):

        for old_name, new_name in (
            self.LEGACY_ALIASES.items()
        ):

            if (
                old_name
                in self.predictors
                and old_name
                not in frame.columns
            ):

                frame[old_name] = (
                    frame[new_name]
                )

        missing = [

            predictor

            for predictor

            in self.predictors

            if predictor
            not in frame.columns

        ]

        if missing:

            raise RuntimeError(

                "Missing frozen "
                f"predictors: {missing}"

            )

        return frame

    # --------------------------------------------------
    # Scoring
    # --------------------------------------------------

    def score_components(
        self,
        frame,
    ):

        frame = (
            self.apply_predictor_aliases(
                frame
            )
        )

        features = (
            frame[
                self.predictors
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        raw_scores = (
            self.model.predict_proba(
                features
            )[:, 1]
        )

        calibrated = (
            self.calibrator.predict(
                raw_scores
            )
        )

        frame[
            "raw_structural_safe_score"
        ] = raw_scores

        frame[
            "prob_structural_safe"
        ] = calibrated

        frame[
            "passes_frozen_structural_threshold"
        ] = (
            calibrated
            >= self.threshold
        )

        red_rule = (

            (
                frame[
                    "ambiguous_fraction"
                ] > 0.50
            )

            |

            (

                (
                    frame[
                        "preserve_fraction"
                    ]
                    > 0.20
                )

                &

                (
                    ~frame[
                        "passes_frozen_structural_threshold"
                    ]
                )

            )

        )

        frame["pilot_category"] = np.where(

            frame[
                "passes_frozen_structural_threshold"
            ],

            "green",

            np.where(
                red_rule,
                "red",
                "amber",
            ),
        )

        return frame

    # --------------------------------------------------
    # Component extraction
    # --------------------------------------------------

    def extract_components(
        self,
        page_name,
        observed,
        labels,
        stats,
        routes,
        confidence,
    ):

        rows = []

        component_count = len(stats)

        for cid in range(
            1,
            component_count,
        ):

            area = stats[
                cid,
                cv2.CC_STAT_AREA,
            ]

            if (
                area
                < self.min_component_area
            ):
                continue

            component = (
                labels
                == cid
            )

            row = {

                "page":
                    page_name,

                "component":
                    cid,

                "x":
                    int(
                        stats[
                            cid,
                            0
                        ]
                    ),

                "y":
                    int(
                        stats[
                            cid,
                            1
                        ]
                    ),

                "bbox_w":
                    int(
                        stats[
                            cid,
                            2
                        ]
                    ),

                "bbox_h":
                    int(
                        stats[
                            cid,
                            3
                        ]
                    ),

                "annotation_confidence_mean":
                    float(
                        confidence[
                            component
                        ].mean()
                    )
                    if component.any()
                    else 0.0,

                "annotation_confidence_max":
                    float(
                        confidence[
                            component
                        ].max()
                    )
                    if component.any()
                    else 0.0,
            }

            row.update(

                surrounding_features(
                    observed,
                    component,
                    routes,
                )

            )

            rows.append(row)

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------
    # Outputs
    # --------------------------------------------------

    def save_outputs(
        self,
        source,
        mask,
        telea,
        proposal,
        routes,
    ):

        stem = source.stem

        cv2.imwrite(

            str(
                self.proposals_dir
                / f"{stem}.png"
            ),

            proposal,
        )

        cv2.imwrite(

            str(
                self.mask_output_dir
                /
                f"{stem}_authoritative.png"
            ),

            mask,
        )

        route_export = {

            f"g80_{key}":

                value.astype(
                    np.uint8
                )

            for key, value
            in routes.items()

            if key in {
                "preserve",
                "remove",
                "ambiguous",
                "horizontal",
                "vertical",
                "compact",
            }
        }

        np.savez_compressed(

            self.routing_dir
            / f"{stem}.npz",

            mask=mask,

            telea=telea,

            geometry_t80=proposal,

            **route_export,

        )

    # --------------------------------------------------
    # Resolution audit
    # --------------------------------------------------

    def write_resolution_audit(
        self,
        source,
        resolved,
    ):

        payload = {

            "source_original":
                str(source),

            "source_md5":
                resolved[
                    "source_md5"
                ],

            "observed_path":
                str(
                    resolved[
                        "observed_path"
                    ]
                ),

            "mask_path":
                str(
                    resolved[
                        "mask_path"
                    ]
                ),

            "confidence_path":
                (
                    str(
                        resolved[
                            "confidence_path"
                        ]
                    )
                    if resolved[
                        "confidence_path"
                    ]
                    else None
                ),

            "review_audit_path":
                resolved[
                    "review_audit_path"
                ],

            "draft_audit_path":
                resolved[
                    "draft_audit_path"
                ],

            "mask_status":
                resolved[
                    "mask_status"
                ],

            "observed_sha256":
                resolved[
                    "observed_sha256"
                ],

            "authoritative_mask_sha256":
                resolved[
                    "authoritative_mask_sha256"
                ],

            "confidence_sha256":
                resolved[
                    "confidence_sha256"
                ],
        }

        (
            self.audit_dir
            /
            f"{source.stem}_input_resolution.json"
        ).write_text(
            json.dumps(
                payload,
                indent=2,
            )
        )

    # --------------------------------------------------
    # Main build operation
    # --------------------------------------------------

    def build(
        self,
        source,
    ):

        source = Path(source)

        resolved = (
            self.resolver.resolve(
                source
            )
        )

        observed = (
            resolved["observed"]
        )

        seed = (
            resolved["seed"]
        )

        confidence = (
            resolved["confidence"]
        )

        mask = prepare_mask(
            seed,
            dilation=2,
        )

        telea = (
            telea_reconstruct(
                observed,
                mask,
            )
        )

        proposal, routes = (
            geometry_t80(
                observed,
                mask,
                telea,
            )
        )

        (
            count,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            (
                mask > 0
            ).astype(
                np.uint8
            ),
            8,
        )

        frame = (
            self.extract_components(
                source.stem,
                observed,
                labels,
                stats,
                routes,
                confidence,
            )
        )

        frame = (
            self.score_components(
                frame
            )
        )

        self.save_outputs(
            source,
            mask,
            telea,
            proposal,
            routes,
        )

        self.write_resolution_audit(
            source,
            resolved,
        )

        return {

            "source":
                source,

            "source_md5":
                resolved[
                    "source_md5"
                ],

            "observed":
                observed,

            "mask":
                mask,

            "proposal":
                proposal,

            "routes":
                routes,

            "labels":
                labels,

            "components":
                frame,

            "mask_source":
                str(
                    resolved[
                        "mask_path"
                    ]
                ),

            "confidence_source":
                (
                    str(
                        resolved[
                            "confidence_path"
                        ]
                    )
                    if resolved[
                        "confidence_path"
                    ]
                    else None
                ),

            "observed_source":
                str(
                    resolved[
                        "observed_path"
                    ]
                ),

            "mask_status":
                resolved[
                    "mask_status"
                ],

            "review_audit":
                resolved[
                    "review_audit_path"
                ],

            "draft_audit":
                resolved[
                    "draft_audit_path"
                ],
        }
