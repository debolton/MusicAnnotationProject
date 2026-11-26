from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np

from .canonical_v78 import build_v78


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

    Output contract:
        PRINT      -> copy the exact observed byte
        ANNOTATION -> retain the exact frozen V7.8 baseline byte
        UNCERTAIN  -> retain the exact frozen V7.8 baseline byte

    Integrity contract:
        outside authoritative mask -> exact observed byte
        outside final PRINT mask    -> exact frozen V7.8 baseline byte
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

        runtime_path = self.bundle_dir / "inference_runtime.py"

        if not runtime_path.is_file():
            raise RuntimeError(
                "Missing frozen A5-v1 inference runtime: "
                f"{runtime_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "a5_v1_frozen_runtime",
            runtime_path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                "Unable to create module specification for "
                f"{runtime_path}"
            )

        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        self.rt = runtime

        self._assert_runtime_api()

        loaded = self.rt.load_bundle(
            self.bundle_dir,
            device=device,
        )

        if not isinstance(loaded, tuple) or len(loaded) != 5:
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
            or self.policy.get("candidate_lock_id")
            or self.policy["policy_id"]
        )

        self.model_sha256 = self.policy.get("model_sha256")
        self.calibrator_sha256 = self.policy.get(
            "calibrator_sha256"
        )

        self.startup_replay = None

        if run_startup_replay:
            self.startup_replay = self._run_startup_replay()

    # ------------------------------------------------------------------
    # Frozen bundle validation
    # ------------------------------------------------------------------

    def _assert_runtime_api(self) -> None:
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
            if not callable(getattr(self.rt, name, None))
        ]

        if missing:
            raise RuntimeError(
                "Frozen A5-v1 runtime API is incomplete: "
                f"{missing}"
            )

    def _assert_bundle_contract(self) -> None:
        if not isinstance(self.policy, dict):
            raise RuntimeError(
                "Frozen A5-v1 policy must be a dictionary"
            )

        if not isinstance(self.schema, dict):
            raise RuntimeError(
                "Frozen A5-v1 schema must be a dictionary"
            )

        channels = self.schema.get("channels")

        if channels != EXPECTED_CHANNELS:
            raise RuntimeError(
                "A5-v1 channel order differs from the frozen "
                f"11-channel contract: {channels!r}"
            )

        patch = self.schema.get(
            "patch",
            self.schema.get("patch_size"),
        )

        if int(patch if patch is not None else -1) != (
            EXPECTED_PATCH_SIZE
        ):
            raise RuntimeError(
                "A5-v1 patch size differs from the frozen "
                f"{EXPECTED_PATCH_SIZE}x{EXPECTED_PATCH_SIZE} "
                f"contract: {patch!r}"
            )

        missing = sorted(
            set(EXPECTED_POLICY) - set(self.policy)
        )

        if missing:
            raise RuntimeError(
                "Frozen A5-v1 policy is missing fields: "
                f"{missing}"
            )

        for key, expected in EXPECTED_POLICY.items():
            actual = self.policy[key]

            if isinstance(expected, float):
                try:
                    actual_number = float(actual)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"A5-v1 policy field {key!r} is not "
                        f"numeric: {actual!r}"
                    ) from exc

                if actual_number != expected:
                    raise RuntimeError(
                        f"A5-v1 policy mismatch for {key}: "
                        f"{actual_number!r} != {expected!r}"
                    )
            elif actual != expected:
                raise RuntimeError(
                    f"A5-v1 policy mismatch for {key}: "
                    f"{actual!r} != {expected!r}"
                )

        effective = max(
            float(self.policy["print_threshold"]),
            (
                1.0
                + float(self.policy["print_margin"])
            )
            / 2.0,
        )

        if effective != float(
            self.policy["effective_print_threshold"]
        ):
            raise RuntimeError(
                "A5-v1 effective PRINT threshold is "
                "internally inconsistent: "
                f"calculated {effective!r}, recorded "
                f"{self.policy['effective_print_threshold']!r}"
            )

        self.a5_policy_id = str(
            self.policy["policy_id"]
        )

    # ------------------------------------------------------------------
    # Startup replay
    # ------------------------------------------------------------------

    def _run_startup_replay(self):
        """
        Run the existing frozen A5-v1 replay gate.

        The application replay is expected to verify the frozen
        PRINT-to-observed and non-PRINT-to-V7.8 contract.
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
                from .hf043_replay import run_ardross_replay
            except ImportError as exc:
                raise RuntimeError(
                    "No A5-v1 startup replay is available. "
                    "Provide runtime.run_replay() or "
                    "app.hf043_replay.run_ardross_replay()."
                ) from exc

            replay = run_ardross_replay(self)

        if replay is False:
            raise RuntimeError(
                "Frozen A5-v1 startup replay failed"
            )

        if isinstance(replay, dict):
            if replay.get("passed") is False:
                raise RuntimeError(
                    "Frozen A5-v1 startup replay failed: "
                    f"{replay}"
                )

            replay_policy_id = replay.get("policy_id")

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
                value = replay.get(key)

                if value is not None and int(value) != 0:
                    raise RuntimeError(
                        "Frozen A5-v1 startup replay failed "
                        f"{key}: {value}"
                    )

        return replay

    # ------------------------------------------------------------------
    # Public page API
    # ------------------------------------------------------------------

    def run(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        page: str | None = None,
        batch: int = 512,
    ) -> dict:
        del page

        base = build_v78(
            observed,
            reviewed_seed,
        )

        if "geometry_t80" not in base:
            raise RuntimeError(
                "build_v78() did not return the frozen "
                "geometry_t80 baseline required by A5-v1"
            )

        if "channels" not in base:
            raise RuntimeError(
                "build_v78() did not return A5-v1 channels"
            )

        if "routes" not in base:
            raise RuntimeError(
                "build_v78() did not return V7.8 routes"
            )

        result = self.process_arrays(
            arrays=base["channels"],
            baseline=base["geometry_t80"],
            authoritative_mask=base["mask"],
            batch=batch,
        )

        return {
            **base,

            # Frozen A5-v1 whole-page result.
            "a5": result["corrected"],
            "a5_v1": result["corrected"],

            # Calibrated PRINT probability.
            "p_print": result["probability"],

            # Three-class decisions.
            "provisional": result["provisional"],
            "final": result["final"],

            # Frozen candidate and effective intervention maps.
            "eligible_a5_v1": result["eligible"],
            "print_mask": result["print_mask"],

            # Candidate coordinates.
            "a5_ys": result["y"],
            "a5_xs": result["x"],

            # Frozen identities.
            "a5_policy_id": self.a5_policy_id,
            "bundle_id": self.bundle_id,
            "model_sha256": self.model_sha256,
            "calibrator_sha256": self.calibrator_sha256,
        }

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
        if not isinstance(arrays, dict):
            raise RuntimeError(
                "A5-v1 arrays must be a dictionary"
            )

        if not isinstance(batch, int) or batch < 1:
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
            "frozen V7.8 baseline",
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
                    "Authoritative mask and channel mask differ"
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

        routed = preserve | remove | ambiguous

        if np.any(preserve & remove):
            raise RuntimeError(
                "PRESERVE and REMOVE routes overlap"
            )

        if np.any(preserve & ambiguous):
            raise RuntimeError(
                "PRESERVE and AMBIGUOUS routes overlap"
            )

        if np.any(remove & ambiguous):
            raise RuntimeError(
                "REMOVE and AMBIGUOUS routes overlap"
            )

        if not np.array_equal(
            routed,
            mask,
        ):
            missing_route = int(
                np.sum(mask & ~routed)
            )
            escaped_route = int(
                np.sum(routed & ~mask)
            )

            raise RuntimeError(
                "V7.8 routes do not exactly partition the "
                "authoritative mask: "
                f"missing={missing_route}, "
                f"outside={escaped_route}"
            )

        if np.any(
            baseline[~mask]
            != observed[~mask]
        ):
            raise RuntimeError(
                "Frozen V7.8 baseline has outside-mask drift"
            )

        # Frozen A5-v1 candidate rule:
        #
        #   inside prepared authoritative mask
        #   AND observed byte below 200
        #   AND not already PRESERVE-routed by V7.8
        eligible = (
            mask
            & (observed < 200)
            & ~preserve
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
            & (observed >= 200)
        ):
            raise RuntimeError(
                "A5-v1 eligibility includes non-dark pixels"
            )

        y, x = np.where(eligible)
        y = y.astype(np.int64)
        x = x.astype(np.int64)

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

        probability = self.rt.score_coordinates(
            self.model,
            self.calibrator,
            padded,
            self.schema,
            y,
            x,
            self.device,
            batch=batch,
        )

        probability = np.asarray(
            probability,
            dtype=np.float64,
        )

        if probability.shape != (len(y),):
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
            np.any(probability < 0.0)
            or np.any(probability > 1.0)
        ):
            raise RuntimeError(
                "A5-v1 PRINT probability lies outside 0..1"
            )

        decision_result = self.rt.decisions(
            probability,
            self.policy,
            observed.shape,
            y,
            x,
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

        probability_map[y, x] = (
            probability.astype(np.float32)
        )
        provisional_map[y, x] = provisional
        final_map[y, x] = final

        corrected, print_mask = self.rt.assemble(
            observed,
            baseline,
            mask,
            y,
            x,
            final,
        )

        corrected = self._require_uint8_image(
            corrected,
            "A5-v1 corrected page",
            observed.shape,
        )

        print_mask = self._require_bool_mask(
            print_mask,
            "A5-v1 PRINT mask",
            observed.shape,
        )

        expected_print_mask = np.zeros(
            observed.shape,
            dtype=bool,
        )

        selected = (
            final
            == "PRINT"
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
                "Runtime PRINT mask differs from final PRINT "
                "decisions: "
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
        decision_result,
        probability_count: int,
        image_shape: tuple[int, int],
        y: np.ndarray,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(decision_result, dict):
            if (
                "provisional" not in decision_result
                or "final" not in decision_result
            ):
                raise RuntimeError(
                    "A5-v1 decisions() dictionary must contain "
                    "'provisional' and 'final'"
                )

            provisional = decision_result["provisional"]
            final = decision_result["final"]

        elif (
            isinstance(decision_result, tuple)
            and len(decision_result) == 2
        ):
            provisional, final = decision_result

        else:
            raise RuntimeError(
                "A5-v1 decisions() must return either "
                "(provisional, final) or a dictionary"
            )

        provisional = np.asarray(provisional)
        final = np.asarray(final)

        if provisional.shape == image_shape:
            provisional = provisional[y, x]

        if final.shape == image_shape:
            final = final[y, x]

        if provisional.shape != (
            probability_count,
        ):
            raise RuntimeError(
                "A5-v1 provisional-decision shape mismatch: "
                f"{provisional.shape}"
            )

        if final.shape != (
            probability_count,
        ):
            raise RuntimeError(
                "A5-v1 final-decision shape mismatch: "
                f"{final.shape}"
            )

        provisional = self._normalise_classes(
            provisional,
            "provisional",
        )

        final = self._normalise_classes(
            final,
            "final",
        )

        if np.any(
            (final == "PRINT")
            & (provisional != "PRINT")
        ):
            raise RuntimeError(
                "A5-v1 run filtering created a new PRINT "
                "decision"
            )

        if np.any(
            (provisional == "ANNOTATION")
            & (final != "ANNOTATION")
        ):
            raise RuntimeError(
                "A5-v1 run filtering altered an ANNOTATION "
                "decision"
            )

        return provisional, final

    def _normalise_classes(
        self,
        values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        values = np.asarray(values)

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
                int(key): str(value).upper()
                for key, value
                in truth_codes.items()
            }

            integer_values = values.astype(
                np.int64
            )

            unknown = sorted(
                set(integer_values.tolist())
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
                    for value in integer_values
                ],
                dtype=object,
            )
        else:
            normalised = np.char.upper(
                values.astype(str)
            ).astype(object)

        unexpected = sorted(
            set(normalised.tolist())
            - ALLOWED_DECISIONS
        )

        if unexpected:
            raise RuntimeError(
                f"{name} decisions contain unexpected "
                f"A5-v1 classes: {unexpected}"
            )

        return normalised

    # ------------------------------------------------------------------
    # Frozen whole-page integrity checks
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
                "A5-v1 PRINT decision escapes the eligible "
                "population"
            )

        if np.any(
            print_mask
            & ~mask
        ):
            raise RuntimeError(
                "A5-v1 PRINT decision escapes the "
                "authoritative mask"
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
                "A5-v1 non-PRINT frozen V7.8 regression"
            )

        if np.any(
            corrected[print_mask]
            != observed[print_mask]
        ):
            raise RuntimeError(
                "A5-v1 PRINT output does not equal the exact "
                "observed byte"
            )

        changed = (
            corrected
            != baseline
        )

        expected_changed = (
            print_mask
            & (
                observed
                != baseline
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
                "A5-v1 page delta does not equal effective "
                "PRINT restoration: "
                f"unexpected={unexpected}, missing={missing}"
            )

        if np.any(
            changed
            & (
                corrected
                != observed
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
        value = np.asarray(value)

        if value.ndim != 2:
            raise RuntimeError(
                f"{name} must be two-dimensional; "
                f"found shape {value.shape}"
            )

        if (
            expected_shape is not None
            and value.shape != expected_shape
        ):
            raise RuntimeError(
                f"{name} shape mismatch: expected "
                f"{expected_shape}, found {value.shape}"
            )

        if not np.issubdtype(
            value.dtype,
            np.number,
        ):
            raise RuntimeError(
                f"{name} must contain numeric values"
            )

        if not np.isfinite(value).all():
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

        if value.size and (
            value.min() < 0
            or value.max() > 255
        ):
            raise RuntimeError(
                f"{name} contains values outside 0..255"
            )

        return np.ascontiguousarray(
            value.astype(np.uint8)
        )

    @staticmethod
    def _require_bool_mask(
        value: np.ndarray,
        name: str,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        value = np.asarray(value)

        if value.ndim != 2:
            raise RuntimeError(
                f"{name} must be two-dimensional; "
                f"found shape {value.shape}"
            )

        if value.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: expected "
                f"{expected_shape}, found {value.shape}"
            )

        if (
            np.issubdtype(value.dtype, np.number)
            and not np.isfinite(value).all()
        ):
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

        return np.ascontiguousarray(
            value.astype(bool)
        )