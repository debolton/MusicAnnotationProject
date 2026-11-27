from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
A5_V1_BUNDLE = ROOT / "frozen" / "a5_v1"


EXPECTED_CHANNELS = [
    "obs",
    "norm",
    "mask",
    "depth",
    "horizontal",
    "vertical",
    "compact",
    "preserve",
    "remove",
    "ambiguous",
    "edge",
]

EXPECTED_PATCH_SIZE = 33
EXPECTED_POLICY_ID = "A5-v1-print-p0980-margin0960-run2"

EXPECTED_POLICY = {
    "annotation_action": "KEEP_A1_BYTE",
    "annotation_threshold": 0.995,
    "clean_image_permitted_at_runtime": False,
    "connectivity": 8,
    "development_policy_selected_post_outcome": True,
    "effective_print_threshold": 0.980,
    "fallback": "A1",
    "independent_validation_completed": False,
    "minimum_print_run": 2,
    "outside_authoritative_mask": "IMMUTABLE_OBSERVED",
    "policy_id": EXPECTED_POLICY_ID,
    "print_action": "COPY_EXACT_OBSERVED_BYTE",
    "print_margin": 0.960,
    "print_threshold": 0.980,
    "production_authorised": False,
    "raw_page_coordinates_as_model_inputs": False,
    "runtime_truth_permitted": False,
    "status": "FROZEN_CANDIDATE_FOR_INDEPENDENT_TEST",
    "truth_codes": {
        "0": "ANNOTATION",
        "1": "PRINT",
        "2": "UNCERTAIN",
    },
    "uncertain_action": "KEEP_A1_BYTE",
}

ALLOWED_DECISIONS = {
    "ANNOTATION",
    "PRINT",
    "UNCERTAIN",
}


class A5Operator:
    """
    Frozen A5-v1 observed-print operator.

    Candidate population:
        authoritative mask
        AND observed byte < 200
        AND not already PRESERVE-routed by frozen V7.8

    Input contract:
        Notebook 87 A1 is supplied by A1Operator.
        A5 must not construct, replace or modify A1.

    Output contract:
        PRINT      -> copy the exact observed byte
        ANNOTATION -> retain the exact Notebook 87 A1 byte
        UNCERTAIN  -> retain the exact Notebook 87 A1 byte

    Integrity contract:
        outside authoritative mask -> exact observed byte
        outside final PRINT mask    -> exact A1 byte
        every changed byte          -> exact observed byte

    This operator is not a CLEAN-to-paper operator.
    """

    a5_policy_id = EXPECTED_POLICY_ID

    def __init__(
        self,
        device: str | None = None,
        bundle_dir: Path | str = A5_V1_BUNDLE,
        run_startup_replay: bool = True,
    ):
        self.bundle_dir = Path(bundle_dir)

        if not self.bundle_dir.is_dir():
            raise RuntimeError(
                "Missing frozen A5-v1 bundle directory: "
                f"{self.bundle_dir}"
            )

        runtime_path = (
            self.bundle_dir
            / "inference_runtime.py"
        )

        if not runtime_path.is_file():
            raise RuntimeError(
                "Missing frozen A5-v1 inference runtime: "
                f"{runtime_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "a5_v1_frozen_runtime",
            runtime_path,
        )

        if (
            spec is None
            or spec.loader is None
        ):
            raise RuntimeError(
                "Unable to create module specification for "
                f"{runtime_path}"
            )

        runtime = importlib.util.module_from_spec(
            spec
        )

        spec.loader.exec_module(
            runtime
        )

        self.rt = runtime

        self._assert_runtime_api()

        loaded = self.rt.load_bundle(
            self.bundle_dir,
            device=device,
        )

        if (
            not isinstance(loaded, tuple)
            or len(loaded) != 5
        ):
            raise RuntimeError(
                "A5-v1 load_bundle() must return exactly "
                "(model, calibrator, policy, schema, device)"
            )

        (
            self.model,
            self.calibrator,
            self.policy,
            self.schema,
            self.device,
        ) = loaded

        self._assert_bundle_contract()

        self.bundle_id = str(
            self.policy.get("bundle_id")
            or self.policy.get(
                "candidate_lock_id"
            )
            or self.policy["policy_id"]
        )

        self.model_sha256 = self.policy.get(
            "model_sha256"
        )

        self.calibrator_sha256 = self.policy.get(
            "calibrator_sha256"
        )

        self.startup_replay = None

        if run_startup_replay:
            self.startup_replay = (
                self._run_startup_replay()
            )

    # ------------------------------------------------------------------
    # Frozen bundle validation
    # ------------------------------------------------------------------

    def _assert_runtime_api(
        self,
    ) -> None:
        required = [
            "load_bundle",
            "make_padded",
            "score_coordinates",
            "decisions",
            "assemble",
        ]

        missing = [
            name
            for name in required
            if not callable(
                getattr(
                    self.rt,
                    name,
                    None,
                )
            )
        ]

        if missing:
            raise RuntimeError(
                "Frozen A5-v1 runtime API is incomplete: "
                f"{missing}"
            )

    def _assert_bundle_contract(
        self,
    ) -> None:
        if not isinstance(
            self.policy,
            dict,
        ):
            raise RuntimeError(
                "Frozen A5-v1 policy must be a dictionary"
            )

        if not isinstance(
            self.schema,
            dict,
        ):
            raise RuntimeError(
                "Frozen A5-v1 schema must be a dictionary"
            )

        channels = self.schema.get(
            "channels"
        )

        if channels != EXPECTED_CHANNELS:
            raise RuntimeError(
                "A5-v1 channel order differs from the frozen "
                f"11-channel contract: {channels!r}"
            )

        patch = self.schema.get(
            "patch",
            self.schema.get(
                "patch_size"
            ),
        )

        if int(
            patch
            if patch is not None
            else -1
        ) != EXPECTED_PATCH_SIZE:
            raise RuntimeError(
                "A5-v1 patch size differs from the frozen "
                f"{EXPECTED_PATCH_SIZE}x"
                f"{EXPECTED_PATCH_SIZE} contract: "
                f"{patch!r}"
            )

        missing = sorted(
            set(EXPECTED_POLICY)
            - set(self.policy)
        )

        if missing:
            raise RuntimeError(
                "Frozen A5-v1 policy is missing fields: "
                f"{missing}"
            )

        for key, expected in (
            EXPECTED_POLICY.items()
        ):
            actual = self.policy[key]

            if isinstance(
                expected,
                float,
            ):
                try:
                    actual_number = float(
                        actual
                    )

                except (
                    TypeError,
                    ValueError,
                ) as exc:
                    raise RuntimeError(
                        "A5-v1 policy field "
                        f"{key!r} is not numeric: "
                        f"{actual!r}"
                    ) from exc

                if actual_number != expected:
                    raise RuntimeError(
                        "A5-v1 policy mismatch for "
                        f"{key}: {actual_number!r} "
                        f"!= {expected!r}"
                    )

            elif actual != expected:
                raise RuntimeError(
                    "A5-v1 policy mismatch for "
                    f"{key}: {actual!r} "
                    f"!= {expected!r}"
                )

        effective = max(
            float(
                self.policy[
                    "print_threshold"
                ]
            ),
            (
                1.0
                + float(
                    self.policy[
                        "print_margin"
                    ]
                )
            )
            / 2.0,
        )

        if effective != float(
            self.policy[
                "effective_print_threshold"
            ]
        ):
            raise RuntimeError(
                "A5-v1 effective PRINT threshold is "
                "internally inconsistent: calculated "
                f"{effective!r}, recorded "
                f"{self.policy['effective_print_threshold']!r}"
            )

        self.a5_policy_id = str(
            self.policy["policy_id"]
        )

    # ------------------------------------------------------------------
    # Startup replay
    # ------------------------------------------------------------------

    def _run_startup_replay(
        self,
    ) -> Any:
        """
        Run the frozen bundle's own replay gate.

        This proves the internal frozen model, calibrator,
        schema, policy and assembly runtime remain replayable.

        The application-level A1 fallback contract is checked
        separately by run_from_a1() and main.py.
        """

        runtime_replay = getattr(
            self.rt,
            "run_replay",
            None,
        )

        if callable(runtime_replay):
            replay = runtime_replay(
                self.bundle_dir,
                self.model,
                self.calibrator,
                self.policy,
                self.schema,
                self.device,
            )

        else:
            try:
                from .hf043_replay import (
                    run_ardross_replay,
                )

            except ImportError as exc:
                raise RuntimeError(
                    "No A5-v1 startup replay is available. "
                    "Provide runtime.run_replay() or "
                    "app.hf043_replay."
                    "run_ardross_replay()."
                ) from exc

            replay = run_ardross_replay(
                self
            )

        if replay is False:
            raise RuntimeError(
                "Frozen A5-v1 startup replay failed"
            )

        if isinstance(
            replay,
            dict,
        ):
            if replay.get(
                "passed"
            ) is False:
                raise RuntimeError(
                    "Frozen A5-v1 startup replay failed: "
                    f"{replay}"
                )

            replay_policy_id = replay.get(
                "policy_id"
            )

            if (
                replay_policy_id is not None
                and str(replay_policy_id)
                != self.a5_policy_id
            ):
                raise RuntimeError(
                    "A5-v1 replay policy mismatch: "
                    f"{replay_policy_id!r} != "
                    f"{self.a5_policy_id!r}"
                )

            zero_checks = [
                "outside_mask_drift",
                "outside_authoritative_mask_drift",
                "non_print_baseline_regression",
                "non_print_v78_regression",
                "print_not_equal_observed",
                "changed_byte_not_equal_observed",
                "unexpected_changed_byte",
            ]

            for key in zero_checks:
                value = replay.get(
                    key
                )

                if (
                    value is not None
                    and int(value) != 0
                ):
                    raise RuntimeError(
                        "Frozen A5-v1 startup replay "
                        f"failed {key}: {value}"
                    )

        return replay

    # ------------------------------------------------------------------
    # Public page API
    # ------------------------------------------------------------------

    def run_from_a1(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        a1_result: dict,
        page: str | None = None,
        batch: int = 512,
    ) -> dict:
        """
        Apply frozen A5-v1 decisions using Notebook 87 A1 as the
        non-PRINT fallback.

        Operational lineage:

            reviewed authoritative seed
                -> frozen V7.8 preparation and routing
                -> frozen V8.9 corrected page
                -> Notebook 87 A1
                -> A5-v1

        A5-v1 assembly:

            final PRINT      -> exact observed byte
            final ANNOTATION -> exact A1 byte
            final UNCERTAIN  -> exact A1 byte
            outside mask     -> exact observed byte

        A5 does not construct, replace or modify A1.
        """

        del page

        if not isinstance(
            a1_result,
            dict,
        ):
            raise RuntimeError(
                "a1_result must be a dictionary"
            )

        required = {
            "observed",
            "reviewed_seed",
            "mask",
            "telea",
            "paper",
            "normalised",
            "horizontal",
            "vertical",
            "compact",
            "preserve",
            "remove",
            "ambiguous",
            "routes",
            "geometry_t80",
            "v8_9_corrected",
            "a1",
        }

        missing = sorted(
            required - set(a1_result)
        )

        if missing:
            raise RuntimeError(
                "A1 result omitted fields required "
                f"by A5-v1: {missing}"
            )

        observed_u8 = self._require_uint8_image(
            observed,
            "observed page",
        )

        shape = observed_u8.shape

        seed = self._require_uint8_image(
            reviewed_seed,
            "reviewed authoritative seed",
            shape,
        )

        a1_observed = self._require_uint8_image(
            a1_result["observed"],
            "A1 observed page",
            shape,
        )

        a1_seed = self._require_uint8_image(
            a1_result["reviewed_seed"],
            "A1 reviewed seed",
            shape,
        )

        if not np.array_equal(
            a1_observed,
            observed_u8,
        ):
            raise RuntimeError(
                "A1 observed page differs from the "
                "page supplied to A5"
            )

        if not np.array_equal(
            a1_seed > 0,
            seed > 0,
        ):
            raise RuntimeError(
                "A1 reviewed seed differs from the "
                "seed supplied to A5"
            )

        authority = self._require_bool_mask(
            a1_result["mask"],
            "authoritative mask",
            shape,
        )

        telea = self._require_uint8_image(
            a1_result["telea"],
            "Telea diagnostic",
            shape,
        )

        paper = self._require_uint8_image(
            a1_result["paper"],
            "paper estimate",
            shape,
        )

        normalised = self._require_uint8_image(
            a1_result["normalised"],
            "background-normalised page",
            shape,
        )

        geometry_t80 = self._require_uint8_image(
            a1_result["geometry_t80"],
            "frozen V7.8 geometry_t80",
            shape,
        )

        v8_9_corrected = self._require_uint8_image(
            a1_result["v8_9_corrected"],
            "frozen V8.9 corrected page",
            shape,
        )

        a1 = self._require_uint8_image(
            a1_result["a1"],
            "Notebook 87 A1",
            shape,
        )

        routes = a1_result["routes"]

        if not isinstance(
            routes,
            dict,
        ):
            raise RuntimeError(
                "a1_result['routes'] must be a dictionary"
            )

        required_routes = {
            "preserve",
            "remove",
            "ambiguous",
        }

        missing_routes = sorted(
            required_routes - set(routes)
        )

        if missing_routes:
            raise RuntimeError(
                "A1 result omitted V7.8 routes "
                f"required by A5-v1: {missing_routes}"
            )

        preserve = self._require_bool_mask(
            a1_result["preserve"],
            "V7.8 PRESERVE route",
            shape,
        )

        remove = self._require_bool_mask(
            a1_result["remove"],
            "V7.8 REMOVE route",
            shape,
        )

        ambiguous = self._require_bool_mask(
            a1_result["ambiguous"],
            "V7.8 AMBIGUOUS route",
            shape,
        )

        horizontal = self._require_bool_mask(
            a1_result["horizontal"],
            "V7.8 horizontal support",
            shape,
        )

        vertical = self._require_bool_mask(
            a1_result["vertical"],
            "V7.8 vertical support",
            shape,
        )

        compact = self._require_bool_mask(
            a1_result["compact"],
            "V7.8 compact support",
            shape,
        )

        # ----------------------------------------------------------
        # Direct fields and route-dictionary fields must agree
        # ----------------------------------------------------------

        for name, direct in (
            ("preserve", preserve),
            ("remove", remove),
            ("ambiguous", ambiguous),
        ):
            route_value = self._require_bool_mask(
                routes[name],
                f"routes[{name!r}]",
                shape,
            )

            if not np.array_equal(
                direct,
                route_value,
            ):
                raise RuntimeError(
                    f"A1 direct {name} array differs from "
                    f"routes[{name!r}]"
                )

        # ----------------------------------------------------------
        # V7.8 route partition
        # ----------------------------------------------------------

        if np.any(
            preserve & remove
        ):
            raise RuntimeError(
                "V7.8 PRESERVE and REMOVE routes overlap"
            )

        if np.any(
            preserve & ambiguous
        ):
            raise RuntimeError(
                "V7.8 PRESERVE and AMBIGUOUS routes overlap"
            )

        if np.any(
            remove & ambiguous
        ):
            raise RuntimeError(
                "V7.8 REMOVE and AMBIGUOUS routes overlap"
            )

        routed = (
            preserve
            | remove
            | ambiguous
        )

        if not np.array_equal(
            routed,
            authority,
        ):
            missing_authority = int(
                np.sum(
                    authority
                    & ~routed
                )
            )

            escaped_route = int(
                np.sum(
                    routed
                    & ~authority
                )
            )

            raise RuntimeError(
                "V7.8 routes do not exactly partition "
                "the authoritative mask: "
                f"missing={missing_authority}, "
                f"outside={escaped_route}"
            )

        # ----------------------------------------------------------
        # Stage-boundary integrity
        # ----------------------------------------------------------

        for name, image in (
            (
                "V7.8 geometry_t80",
                geometry_t80,
            ),
            (
                "V8.9 corrected page",
                v8_9_corrected,
            ),
            (
                "Notebook 87 A1",
                a1,
            ),
            (
                "Telea diagnostic",
                telea,
            ),
        ):
            if np.any(
                image[~authority]
                != observed_u8[~authority]
            ):
                raise RuntimeError(
                    f"{name} changed pixels outside "
                    "the authoritative mask"
                )

        if np.any(
            a1[ambiguous]
            != observed_u8[ambiguous]
        ):
            raise RuntimeError(
                "Notebook 87 A1 does not contain the "
                "exact observed bytes at every "
                "V7.8 AMBIGUOUS coordinate"
            )

        if np.any(
            a1[~ambiguous]
            != v8_9_corrected[~ambiguous]
        ):
            raise RuntimeError(
                "Notebook 87 A1 differs from V8.9 "
                "outside the V7.8 AMBIGUOUS route"
            )

        actual_a1_delta = (
            a1 != v8_9_corrected
        )

        expected_a1_delta = (
            ambiguous
            & (
                observed_u8
                != v8_9_corrected
            )
        )

        if not np.array_equal(
            actual_a1_delta,
            expected_a1_delta,
        ):
            unexpected = int(
                np.sum(
                    actual_a1_delta
                    & ~expected_a1_delta
                )
            )

            missing_effective = int(
                np.sum(
                    expected_a1_delta
                    & ~actual_a1_delta
                )
            )

            raise RuntimeError(
                "Notebook 87 A1 delta from V8.9 does "
                "not equal effective AMBIGUOUS-to-"
                "observed restoration: "
                f"unexpected={unexpected}, "
                f"missing={missing_effective}"
            )

        # ----------------------------------------------------------
        # Construct the frozen A5 11-channel input
        #
        # These definitions match the V8.2/A5 channel contract.
        # No clean or truth-derived input is used.
        # ----------------------------------------------------------

        depth = cv2.distanceTransform(
            authority.astype(np.uint8),
            cv2.DIST_L2,
            5,
        ).astype(np.float32)

        edge = (
            cv2.Canny(
                observed_u8,
                70,
                170,
            )
            > 0
        )

        channels = {
            "obs": observed_u8,
            "norm": normalised,
            "mask": authority,
            "depth": depth,
            "horizontal": horizontal,
            "vertical": vertical,
            "compact": compact,
            "preserve": preserve,
            "remove": remove,
            "ambiguous": ambiguous,
            "edge": edge,
        }

        if list(
            channels
        ) != EXPECTED_CHANNELS:
            raise RuntimeError(
                "Constructed A5 channel order differs "
                "from the frozen schema: "
                f"{list(channels)!r}"
            )

        if list(
            self.schema["channels"]
        ) != EXPECTED_CHANNELS:
            raise RuntimeError(
                "Loaded A5 schema does not match the "
                "expected frozen channel order"
            )

        # ----------------------------------------------------------
        # Run A5 from the exact supplied A1 fallback
        # ----------------------------------------------------------

        result = self.process_arrays(
            arrays=channels,
            baseline=a1,
            authoritative_mask=authority,
            batch=batch,
        )

        a5 = self._require_uint8_image(
            result["corrected"],
            "A5-v1 corrected page",
            shape,
        )

        eligible_a5_v1 = (
            self._require_bool_mask(
                result["eligible"],
                "A5-v1 eligibility",
                shape,
            )
        )

        print_mask = (
            self._require_bool_mask(
                result["print_mask"],
                "A5-v1 PRINT mask",
                shape,
            )
        )

        p_print = np.asarray(
            result["probability"]
        )

        provisional = np.asarray(
            result["provisional"]
        )

        final = np.asarray(
            result["final"]
        )

        for name, value in (
            (
                "p_print",
                p_print,
            ),
            (
                "provisional",
                provisional,
            ),
            (
                "final",
                final,
            ),
        ):
            if value.shape != shape:
                raise RuntimeError(
                    f"{name} shape mismatch: "
                    f"expected {shape}, "
                    f"found {value.shape}"
                )

        # ----------------------------------------------------------
        # Explicit A1 fallback contract
        # ----------------------------------------------------------

        if np.any(
            a5[print_mask]
            != observed_u8[print_mask]
        ):
            raise RuntimeError(
                "A5-v1 PRINT output does not equal "
                "the exact observed byte"
            )

        if np.any(
            a5[~print_mask]
            != a1[~print_mask]
        ):
            raise RuntimeError(
                "A5-v1 non-PRINT output differs from "
                "Notebook 87 A1"
            )

        if np.any(
            a5[~authority]
            != observed_u8[~authority]
        ):
            raise RuntimeError(
                "A5-v1 changed pixels outside the "
                "authoritative mask"
            )

        actual_delta_from_a1 = (
            a5 != a1
        )

        expected_delta_from_a1 = (
            print_mask
            & (
                observed_u8 != a1
            )
        )

        if not np.array_equal(
            actual_delta_from_a1,
            expected_delta_from_a1,
        ):
            unexpected = int(
                np.sum(
                    actual_delta_from_a1
                    & ~expected_delta_from_a1
                )
            )

            missing_effective = int(
                np.sum(
                    expected_delta_from_a1
                    & ~actual_delta_from_a1
                )
            )

            raise RuntimeError(
                "A5-v1 delta from A1 does not equal "
                "effective PRINT restoration: "
                f"unexpected={unexpected}, "
                f"missing={missing_effective}"
            )

        if np.any(
            actual_delta_from_a1
            & (
                a5 != observed_u8
            )
        ):
            raise RuntimeError(
                "A5-v1 changed byte does not equal observed"
            )

        # ----------------------------------------------------------
        # Return the complete lineage without overwriting A1
        # ----------------------------------------------------------

        return {
            **a1_result,

            "mask": authority,
            "telea": telea,
            "paper": paper,
            "normalised": normalised,

            "horizontal": horizontal,
            "vertical": vertical,
            "compact": compact,

            "preserve": preserve,
            "remove": remove,
            "ambiguous": ambiguous,

            "channels": channels,
            "depth": depth,
            "edge": edge,

            "routes": {
                **routes,
                "preserve": preserve,
                "remove": remove,
                "ambiguous": ambiguous,
            },

            "geometry_t80": geometry_t80,
            "v8_9_corrected": v8_9_corrected,

            "a1": a1,
            "a5": a5,

            "eligible_a5_v1": eligible_a5_v1,

            "p_print": p_print,
            "provisional": provisional,
            "final": final,
            "print_mask": print_mask,

            "a5_ys": result["y"],
            "a5_xs": result["x"],

            "a5_delta_from_a1": (
                actual_delta_from_a1
            ),

            "a5_delta_from_v7_8": (
                a5 != geometry_t80
            ),

            "a1_policy_id": a1_result.get(
                "a1_policy_id",
                "NOTEBOOK87_A1_FROM_FROZEN_V8_9",
            ),

            "v89_operator_sha256": (
                a1_result[
                    "v89_operator_sha256"
                ]
            ),

            "a5_policy_id": (
                self.a5_policy_id
            ),

            "bundle_id": (
                self.bundle_id
            ),

            "model_sha256": (
                self.model_sha256
            ),

            "calibrator_sha256": (
                self.calibrator_sha256
            ),

            "a5_action": (
                "COPY_EXACT_OBSERVED_BYTE"
            ),

            "a5_fallback": (
                "Notebook87 A1"
            ),
        }

    def run(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        page: str | None = None,
        batch: int = 512,
    ) -> dict:
        """
        Disabled legacy entry point.

        A5 now requires a precomputed Notebook 87 A1 result.
        """

        del observed
        del reviewed_seed
        del page
        del batch

        raise RuntimeError(
            "A5Operator.run() is disabled. "
            "A5-v1 requires a precomputed Notebook 87 A1 "
            "result from A1Operator. Call run_from_a1()."
        )

    # ------------------------------------------------------------------
    # Array-level API
    # ------------------------------------------------------------------

    def process_arrays(
        self,
        arrays: dict,
        baseline: np.ndarray,
        authoritative_mask: np.ndarray | None = None,
        batch: int = 512,
    ) -> dict:
        """
        Score the frozen A5-v1 eligible population and assemble a
        corrected page from the supplied fallback baseline.

        Operationally the fallback baseline is Notebook 87 A1.
        This method does not construct, infer or replace A1.
        """

        if not isinstance(
            arrays,
            dict,
        ):
            raise RuntimeError(
                "A5-v1 arrays must be a dictionary"
            )

        if (
            not isinstance(batch, int)
            or batch < 1
        ):
            raise ValueError(
                "batch must be a positive integer"
            )

        required_channels = list(
            self.schema["channels"]
        )

        missing = [
            channel
            for channel in required_channels
            if channel not in arrays
        ]

        if missing:
            raise RuntimeError(
                "A5-v1 arrays omit channels: "
                f"{missing}"
            )

        observed = self._require_uint8_image(
            arrays["obs"],
            "observed",
        )

        baseline = self._require_uint8_image(
            baseline,
            "A5 non-PRINT fallback baseline",
            observed.shape,
        )

        schema_mask = self._require_bool_mask(
            arrays["mask"],
            "channel mask",
            observed.shape,
        )

        if authoritative_mask is None:
            mask = schema_mask

        else:
            mask = self._require_bool_mask(
                authoritative_mask,
                "authoritative mask",
                observed.shape,
            )

            if not np.array_equal(
                mask,
                schema_mask,
            ):
                raise RuntimeError(
                    "Authoritative mask and "
                    "channel mask differ"
                )

        preserve = self._require_bool_mask(
            arrays["preserve"],
            "preserve route",
            observed.shape,
        )

        remove = self._require_bool_mask(
            arrays["remove"],
            "remove route",
            observed.shape,
        )

        ambiguous = self._require_bool_mask(
            arrays["ambiguous"],
            "ambiguous route",
            observed.shape,
        )

        routed = (
            preserve
            | remove
            | ambiguous
        )

        if np.any(
            preserve
            & remove
        ):
            raise RuntimeError(
                "PRESERVE and REMOVE routes overlap"
            )

        if np.any(
            preserve
            & ambiguous
        ):
            raise RuntimeError(
                "PRESERVE and AMBIGUOUS routes overlap"
            )

        if np.any(
            remove
            & ambiguous
        ):
            raise RuntimeError(
                "REMOVE and AMBIGUOUS routes overlap"
            )

        if not np.array_equal(
            routed,
            mask,
        ):
            missing_route = int(
                np.sum(
                    mask
                    & ~routed
                )
            )

            escaped_route = int(
                np.sum(
                    routed
                    & ~mask
                )
            )

            raise RuntimeError(
                "V7.8 routes do not exactly partition "
                "the authoritative mask: "
                f"missing={missing_route}, "
                f"outside={escaped_route}"
            )

        if np.any(
            baseline[~mask]
            != observed[~mask]
        ):
            raise RuntimeError(
                "A5 fallback baseline has outside-mask drift"
            )

        # Frozen A5-v1 candidate rule:
        #
        #   inside prepared authoritative mask
        #   AND observed byte below 200
        #   AND not already PRESERVE-routed by V7.8

        eligible = (
            mask
            & (
                observed < 200
            )
            & ~preserve
        )

        expected_eligible = (
            mask
            & (
                observed < 200
            )
            & ~preserve
        )

        if not np.array_equal(
            eligible,
            expected_eligible,
        ):
            raise RuntimeError(
                "A5-v1 eligibility does not equal "
                "the frozen candidate rule"
            )

        if np.any(
            eligible
            & ~mask
        ):
            raise RuntimeError(
                "A5-v1 eligibility escapes the "
                "authoritative mask"
            )

        if np.any(
            eligible
            & preserve
        ):
            raise RuntimeError(
                "A5-v1 eligibility includes PRESERVE pixels"
            )

        if np.any(
            eligible
            & (
                observed >= 200
            )
        ):
            raise RuntimeError(
                "A5-v1 eligibility includes non-dark pixels"
            )

        y, x = np.where(
            eligible
        )

        y = y.astype(
            np.int64
        )

        x = x.astype(
            np.int64
        )

        probability_map = np.full(
            observed.shape,
            np.nan,
            dtype=np.float32,
        )

        provisional_map = np.full(
            observed.shape,
            "INELIGIBLE",
            dtype=object,
        )

        final_map = np.full(
            observed.shape,
            "INELIGIBLE",
            dtype=object,
        )

        if len(y) == 0:
            print_mask = np.zeros(
                observed.shape,
                dtype=bool,
            )

            corrected = baseline.copy()

            self._assert_output_contract(
                observed=observed,
                baseline=baseline,
                mask=mask,
                eligible=eligible,
                print_mask=print_mask,
                corrected=corrected,
            )

            return {
                "eligible": eligible,
                "probability": probability_map,
                "provisional": provisional_map,
                "final": final_map,
                "y": y,
                "x": x,
                "print_mask": print_mask,
                "corrected": corrected,
            }

        padded = self.rt.make_padded(
            arrays,
            self.schema,
        )

        probability = (
            self.rt.score_coordinates(
                self.model,
                self.calibrator,
                padded,
                self.schema,
                y,
                x,
                self.device,
                batch=batch,
            )
        )

        probability = np.asarray(
            probability,
            dtype=np.float64,
        )

        if probability.shape != (
            len(y),
        ):
            raise RuntimeError(
                "A5-v1 PRINT probability shape mismatch: "
                f"expected {(len(y),)}, "
                f"found {probability.shape}"
            )

        if not np.isfinite(
            probability
        ).all():
            raise RuntimeError(
                "A5-v1 PRINT probability contains "
                "non-finite values"
            )

        if (
            np.any(
                probability < 0.0
            )
            or np.any(
                probability > 1.0
            )
        ):
            raise RuntimeError(
                "A5-v1 PRINT probability lies outside 0..1"
            )

        decision_result = (
            self.rt.decisions(
                probability,
                self.policy,
                observed.shape,
                y,
                x,
            )
        )

        provisional, final = (
            self._normalise_decision_result(
                decision_result=decision_result,
                probability_count=len(y),
                image_shape=observed.shape,
                y=y,
                x=x,
            )
        )

        probability_map[
            y,
            x,
        ] = probability.astype(
            np.float32
        )

        provisional_map[
            y,
            x,
        ] = provisional

        final_map[
            y,
            x,
        ] = final

        corrected, print_mask = (
            self.rt.assemble(
                observed,
                baseline,
                mask,
                y,
                x,
                final,
            )
        )

        corrected = (
            self._require_uint8_image(
                corrected,
                "A5-v1 corrected page",
                observed.shape,
            )
        )

        print_mask = (
            self._require_bool_mask(
                print_mask,
                "A5-v1 PRINT mask",
                observed.shape,
            )
        )

        expected_print_mask = np.zeros(
            observed.shape,
            dtype=bool,
        )

        selected = (
            final == "PRINT"
        )

        expected_print_mask[
            y[selected],
            x[selected],
        ] = True

        if not np.array_equal(
            print_mask,
            expected_print_mask,
        ):
            missing_print = int(
                np.sum(
                    expected_print_mask
                    & ~print_mask
                )
            )

            unexpected_print = int(
                np.sum(
                    print_mask
                    & ~expected_print_mask
                )
            )

            raise RuntimeError(
                "Runtime PRINT mask differs from final "
                "PRINT decisions: "
                f"missing={missing_print}, "
                f"unexpected={unexpected_print}"
            )

        self._assert_output_contract(
            observed=observed,
            baseline=baseline,
            mask=mask,
            eligible=eligible,
            print_mask=print_mask,
            corrected=corrected,
        )

        return {
            "eligible": eligible,
            "probability": probability_map,
            "provisional": provisional_map,
            "final": final_map,
            "y": y,
            "x": x,
            "print_mask": print_mask,
            "corrected": corrected,
        }

    # ------------------------------------------------------------------
    # Decision validation
    # ------------------------------------------------------------------

    def _normalise_decision_result(
        self,
        decision_result: Any,
        probability_count: int,
        image_shape: tuple[int, int],
        y: np.ndarray,
        x: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        if isinstance(
            decision_result,
            dict,
        ):
            if (
                "provisional"
                not in decision_result
                or "final"
                not in decision_result
            ):
                raise RuntimeError(
                    "A5-v1 decisions() dictionary must "
                    "contain 'provisional' and 'final'"
                )

            provisional = (
                decision_result[
                    "provisional"
                ]
            )

            final = (
                decision_result[
                    "final"
                ]
            )

        elif (
            isinstance(
                decision_result,
                tuple,
            )
            and len(
                decision_result
            ) == 2
        ):
            provisional, final = (
                decision_result
            )

        else:
            raise RuntimeError(
                "A5-v1 decisions() must return either "
                "(provisional, final) or a dictionary"
            )

        provisional = np.asarray(
            provisional
        )

        final = np.asarray(
            final
        )

        if provisional.shape == image_shape:
            provisional = provisional[
                y,
                x,
            ]

        if final.shape == image_shape:
            final = final[
                y,
                x,
            ]

        if provisional.shape != (
            probability_count,
        ):
            raise RuntimeError(
                "A5-v1 provisional-decision "
                "shape mismatch: "
                f"{provisional.shape}"
            )

        if final.shape != (
            probability_count,
        ):
            raise RuntimeError(
                "A5-v1 final-decision shape mismatch: "
                f"{final.shape}"
            )

        provisional = (
            self._normalise_classes(
                provisional,
                "provisional",
            )
        )

        final = self._normalise_classes(
            final,
            "final",
        )

        # The page-global run filter may demote PRINT to
        # UNCERTAIN, but it must never manufacture PRINT.

        if np.any(
            (
                final == "PRINT"
            )
            & (
                provisional != "PRINT"
            )
        ):
            raise RuntimeError(
                "A5-v1 run filtering created a new "
                "PRINT decision"
            )

        # ANNOTATION is a final high-confidence class and must
        # not be altered by the PRINT run-length filter.

        if np.any(
            (
                provisional
                == "ANNOTATION"
            )
            & (
                final
                != "ANNOTATION"
            )
        ):
            raise RuntimeError(
                "A5-v1 run filtering altered an "
                "ANNOTATION decision"
            )

        return (
            provisional,
            final,
        )

    def _normalise_classes(
        self,
        values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        values = np.asarray(
            values
        )

        truth_codes = self.policy.get(
            "truth_codes",
            {},
        )

        if (
            truth_codes
            and np.issubdtype(
                values.dtype,
                np.number,
            )
        ):
            mapping = {
                int(key): str(
                    value
                ).upper()
                for key, value
                in truth_codes.items()
            }

            if not np.isfinite(
                values
            ).all():
                raise RuntimeError(
                    f"{name} decisions contain "
                    "non-finite numeric codes"
                )

            integer_values = (
                values.astype(
                    np.int64
                )
            )

            unknown = sorted(
                set(
                    integer_values.tolist()
                )
                - set(mapping)
            )

            if unknown:
                raise RuntimeError(
                    f"{name} decisions contain unknown "
                    f"numeric codes: {unknown}"
                )

            normalised = np.asarray(
                [
                    mapping[int(value)]
                    for value
                    in integer_values
                ],
                dtype=object,
            )

        else:
            normalised = np.char.upper(
                values.astype(
                    str
                )
            ).astype(
                object
            )

        unexpected = sorted(
            set(
                normalised.tolist()
            )
            - ALLOWED_DECISIONS
        )

        if unexpected:
            raise RuntimeError(
                f"{name} decisions contain unexpected "
                f"A5-v1 classes: {unexpected}"
            )

        return normalised

    # ------------------------------------------------------------------
    # Whole-page integrity checks
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_output_contract(
        observed: np.ndarray,
        baseline: np.ndarray,
        mask: np.ndarray,
        eligible: np.ndarray,
        print_mask: np.ndarray,
        corrected: np.ndarray,
    ) -> None:
        if np.any(
            print_mask
            & ~eligible
        ):
            raise RuntimeError(
                "A5-v1 PRINT decision escapes "
                "the eligible population"
            )

        if np.any(
            print_mask
            & ~mask
        ):
            raise RuntimeError(
                "A5-v1 PRINT decision escapes "
                "the authoritative mask"
            )

        if np.any(
            corrected[~mask]
            != observed[~mask]
        ):
            raise RuntimeError(
                "A5-v1 outside-authoritative-mask drift"
            )

        if np.any(
            corrected[~print_mask]
            != baseline[~print_mask]
        ):
            raise RuntimeError(
                "A5-v1 non-PRINT fallback regression"
            )

        if np.any(
            corrected[print_mask]
            != observed[print_mask]
        ):
            raise RuntimeError(
                "A5-v1 PRINT output does not equal "
                "the exact observed byte"
            )

        changed = (
            corrected != baseline
        )

        expected_changed = (
            print_mask
            & (
                observed != baseline
            )
        )

        if not np.array_equal(
            changed,
            expected_changed,
        ):
            unexpected = int(
                np.sum(
                    changed
                    & ~expected_changed
                )
            )

            missing = int(
                np.sum(
                    expected_changed
                    & ~changed
                )
            )

            raise RuntimeError(
                "A5-v1 page delta does not equal "
                "effective PRINT restoration: "
                f"unexpected={unexpected}, "
                f"missing={missing}"
            )

        if np.any(
            changed
            & (
                corrected != observed
            )
        ):
            raise RuntimeError(
                "A5-v1 changed byte does not equal observed"
            )

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_uint8_image(
        value: np.ndarray,
        name: str,
        expected_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        value = np.asarray(
            value
        )

        if value.ndim != 2:
            raise RuntimeError(
                f"{name} must be two-dimensional; "
                f"found shape {value.shape}"
            )

        if (
            expected_shape is not None
            and value.shape
            != expected_shape
        ):
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"expected {expected_shape}, "
                f"found {value.shape}"
            )

        if not np.issubdtype(
            value.dtype,
            np.number,
        ):
            raise RuntimeError(
                f"{name} must contain numeric values"
            )

        if not np.isfinite(
            value
        ).all():
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

        if (
            value.size
            and (
                value.min() < 0
                or value.max() > 255
            )
        ):
            raise RuntimeError(
                f"{name} contains values outside 0..255"
            )

        return np.ascontiguousarray(
            value.astype(
                np.uint8
            )
        )

    @staticmethod
    def _require_bool_mask(
        value: np.ndarray,
        name: str,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        value = np.asarray(
            value
        )

        if value.ndim != 2:
            raise RuntimeError(
                f"{name} must be two-dimensional; "
                f"found shape {value.shape}"
            )

        if value.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"expected {expected_shape}, "
                f"found {value.shape}"
            )

        if (
            np.issubdtype(
                value.dtype,
                np.number,
            )
            and not np.isfinite(
                value
            ).all()
        ):
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

        return np.ascontiguousarray(
            value.astype(
                bool
            )
        )