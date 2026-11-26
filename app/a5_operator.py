from pathlib import Path
import importlib.util
import json

import numpy as np

from .canonical_v78 import build_v78


ROOT = Path(__file__).resolve().parents[1]

SAFE_CLEANUP_BUNDLE = (
    ROOT
    / "frozen"
    / "a5_v1"
)

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

A1_POLICY_ID = (
    "A1-v2-observed-remove-paper-ambiguous-observed"
)

DEFAULT_A5_POLICY_ID = (
    "A5-safe-cleanup-ambiguous-only"
)


class A5Operator:
    """
    Operational A5 safe-cleanup operator.

    A1 contract:
        PRESERVE  -> exact observed byte
        REMOVE    -> paper estimate
        AMBIGUOUS -> exact observed byte
        outside prepared mask -> exact observed byte

    A5 contract:
        eligible population -> AMBIGUOUS only
        CLEAN               -> paper estimate
        non-CLEAN           -> exact A1 byte
        outside AMBIGUOUS   -> exact A1 byte
        outside prepared mask -> exact observed byte

    This is not the Notebook 91 observed-PRINT operator.
    """

    a1_policy_id = A1_POLICY_ID
    a5_policy_id = DEFAULT_A5_POLICY_ID

    def __init__(
        self,
        device: str | None = None,
        bundle_dir: Path | str = SAFE_CLEANUP_BUNDLE,
        run_startup_replay: bool = True,
    ):
        self.bundle_dir = Path(bundle_dir)

        if not self.bundle_dir.exists():
            raise RuntimeError(
                "Missing operational safe-cleanup bundle: "
                f"{self.bundle_dir}"
            )

        runtime_path = (
            self.bundle_dir
            / "inference_runtime.py"
        )

        if not runtime_path.exists():
            raise RuntimeError(
                "Missing operational safe-cleanup runtime: "
                f"{runtime_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "a5_safe_cleanup_frozen_runtime",
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

        if not isinstance(loaded, tuple):
            raise RuntimeError(
                "load_bundle() must return a tuple"
            )

        if len(loaded) != 5:
            raise RuntimeError(
                "load_bundle() must return exactly: "
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

        self.bundle_id = (
            self.policy.get("bundle_id")
            or self.policy.get("candidate_lock_id")
            or self.policy.get("policy_id")
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
    # Bundle validation
    # ------------------------------------------------------------------

    def _assert_runtime_api(self) -> None:
        required = [
            "load_bundle",
            "make_padded",
            "score_coordinates",
            "decisions",
        ]

        missing = [
            name
            for name in required
            if not callable(
                getattr(self.rt, name, None)
            )
        ]

        if missing:
            raise RuntimeError(
                "Safe-cleanup frozen runtime API is incomplete: "
                f"{missing}"
            )

    def _assert_bundle_contract(self) -> None:
        if not isinstance(self.policy, dict):
            raise RuntimeError(
                "Safe-cleanup policy must be a dictionary"
            )

        if not isinstance(self.schema, dict):
            raise RuntimeError(
                "Safe-cleanup schema must be a dictionary"
            )

        channels = self.schema.get("channels")

        if channels != EXPECTED_CHANNELS:
            raise RuntimeError(
                "Safe-cleanup channel order differs from the "
                "expected 11-channel contract: "
                f"{channels}"
            )

        patch = self.schema.get(
            "patch",
            self.schema.get("patch_size"),
        )

        if int(patch if patch is not None else -1) != (
            EXPECTED_PATCH_SIZE
        ):
            raise RuntimeError(
                "Safe-cleanup patch size differs from the "
                f"expected {EXPECTED_PATCH_SIZE}x"
                f"{EXPECTED_PATCH_SIZE} contract: {patch}"
            )

        probability_semantics = str(
            self.policy.get(
                "probability_semantics",
                "",
            )
        ).upper()

        positive_class = str(
            self.policy.get(
                "positive_class",
                "",
            )
        ).upper()

        if (
            probability_semantics != "CLEAN"
            and positive_class != "CLEAN"
        ):
            raise RuntimeError(
                "Operational A5 bundle does not declare CLEAN "
                "probability semantics. The Notebook 91 "
                "observed-PRINT bundle must not be loaded here."
            )

        eligible_route = str(
            self.policy.get(
                "eligible_route",
                "",
            )
        ).upper()

        if eligible_route != "AMBIGUOUS":
            raise RuntimeError(
                "Operational A5 policy must declare "
                "eligible_route='AMBIGUOUS'; found "
                f"{eligible_route!r}"
            )

        clean_action = str(
            self.policy.get(
                "clean_action",
                "",
            )
        ).upper()

        if clean_action not in {
            "PAPER",
            "COPY_PAPER",
            "COPY_PAPER_ESTIMATE",
        }:
            raise RuntimeError(
                "Operational A5 policy must declare that CLEAN "
                "copies the paper estimate; found "
                f"{clean_action!r}"
            )

        default_action = str(
            self.policy.get(
                "default_action",
                self.policy.get(
                    "non_clean_action",
                    "",
                ),
            )
        ).upper()

        if default_action not in {
            "A1",
            "KEEP_A1",
            "COPY_A1",
        }:
            raise RuntimeError(
                "Operational A5 default must retain exact A1; "
                f"found {default_action!r}"
            )

        policy_id = self.policy.get("policy_id")

        if not policy_id:
            raise RuntimeError(
                "Safe-cleanup policy has no policy_id"
            )

        self.a5_policy_id = str(policy_id)

        truth_codes = self.policy.get(
            "truth_codes",
            {},
        )

        if truth_codes:
            normalised_codes = {
                str(key): str(value).upper()
                for key, value in truth_codes.items()
            }

            allowed_classes = {
                "CLEAN",
                "KEEP",
                "UNCERTAIN",
            }

            unexpected = (
                set(normalised_codes.values())
                - allowed_classes
            )

            if unexpected:
                raise RuntimeError(
                    "Unexpected safe-cleanup class names: "
                    f"{sorted(unexpected)}"
                )

            if "CLEAN" not in normalised_codes.values():
                raise RuntimeError(
                    "Safe-cleanup class mapping contains no "
                    "CLEAN class"
                )

        if callable(
            getattr(self.rt, "assemble", None)
        ):
            assemble_semantics = str(
                self.policy.get(
                    "runtime_assemble_semantics",
                    "",
                )
            ).upper()

            if assemble_semantics in {
                "PRINT_TO_OBSERVED",
                "COPY_OBSERVED_AT_PRINT",
                "OBSERVED_PRINT",
            }:
                raise RuntimeError(
                    "Safe-cleanup bundle exposes historical "
                    "observed-PRINT assembly semantics"
                )

    # ------------------------------------------------------------------
    # Startup replay
    # ------------------------------------------------------------------

    def _run_startup_replay(self) -> dict:
        """
        Run the operational replay supplied for the safe-cleanup bundle.

        Preferred API:
            runtime.run_replay(
                bundle_dir,
                model,
                calibrator,
                policy,
                schema,
                device,
            )

        A separate application replay module may be used as a fallback.
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
                from .a5_safe_cleanup_replay import (
                    run_safe_cleanup_replay,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "No operational safe-cleanup startup replay "
                    "is available. Provide runtime.run_replay() "
                    "or app.a5_safe_cleanup_replay."
                ) from exc

            replay = run_safe_cleanup_replay(self)

        if not isinstance(replay, dict):
            raise RuntimeError(
                "Safe-cleanup startup replay must return a "
                "dictionary"
            )

        if replay.get("passed") is not True:
            raise RuntimeError(
                "Safe-cleanup startup replay failed: "
                f"{json.dumps(replay, default=str)}"
            )

        replay_policy_id = replay.get("policy_id")

        if (
            replay_policy_id is not None
            and replay_policy_id
            != self.a5_policy_id
        ):
            raise RuntimeError(
                "Safe-cleanup replay policy mismatch: "
                f"{replay_policy_id} != "
                f"{self.a5_policy_id}"
            )

        required_zero_checks = [
            "outside_mask_drift",
            "non_ambiguous_regression",
            "non_clean_a1_regression",
            "clean_not_equal_paper",
        ]

        for name in required_zero_checks:
            value = replay.get(name)

            if value is not None and int(value) != 0:
                raise RuntimeError(
                    "Safe-cleanup startup replay failed "
                    f"{name}: {value}"
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

        result = self.process_arrays(
            arrays=base["channels"],
            a1=base["a1"],
            paper=base["routes"]["paper"],
            ambiguous=base["routes"]["ambiguous"],
            authoritative_mask=base["mask"],
            eligible=base["eligible"],
            batch=batch,
        )

        return {
            **base,
            "a5": result["corrected"],
            "p_clean": result["probability"],
            "provisional": result["provisional"],
            "final": result["final"],
            "clean_mask": result["clean_mask"],
            "eligible": result["eligible"],
            "ys": result["y"],
            "xs": result["x"],
            "a1_policy_id": self.a1_policy_id,
            "a5_policy_id": self.a5_policy_id,
            "bundle_id": self.bundle_id,
            "model_sha256": self.model_sha256,
            "calibrator_sha256": (
                self.calibrator_sha256
            ),
        }

    # ------------------------------------------------------------------
    # Array-level API
    # ------------------------------------------------------------------

    def process_arrays(
        self,
        arrays: dict,
        a1: np.ndarray,
        paper: np.ndarray,
        ambiguous: np.ndarray,
        authoritative_mask: np.ndarray | None = None,
        eligible: np.ndarray | None = None,
        batch: int = 512,
    ) -> dict:
        if not isinstance(arrays, dict):
            raise RuntimeError(
                "Safe-cleanup arrays must be a dictionary"
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
                "Safe-cleanup arrays omit channels: "
                f"{missing}"
            )

        observed = self._require_uint8_image(
            arrays["obs"],
            "observed",
        )
        a1 = self._require_uint8_image(
            a1,
            "a1",
            observed.shape,
        )
        paper = self._require_uint8_image(
            paper,
            "paper",
            observed.shape,
        )

        schema_mask = self._require_bool_mask(
            arrays["mask"],
            "schema mask",
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

        schema_ambiguous = self._require_bool_mask(
            arrays["ambiguous"],
            "schema ambiguous",
            observed.shape,
        )

        ambiguous = self._require_bool_mask(
            ambiguous,
            "ambiguous",
            observed.shape,
        )

        if not np.array_equal(
            ambiguous,
            schema_ambiguous,
        ):
            raise RuntimeError(
                "Supplied AMBIGUOUS route differs from the "
                "channel AMBIGUOUS route"
            )

        if eligible is None:
            eligible_mask = ambiguous.copy()
        else:
            eligible_mask = self._require_bool_mask(
                eligible,
                "eligible",
                observed.shape,
            )

        if np.any(eligible_mask & ~ambiguous):
            raise RuntimeError(
                "A5 eligibility includes non-AMBIGUOUS pixels"
            )

        if np.any(eligible_mask & ~mask):
            raise RuntimeError(
                "A5 eligibility escapes the prepared mask"
            )

        if np.any(ambiguous & ~mask):
            raise RuntimeError(
                "AMBIGUOUS route escapes the prepared mask"
            )

        if np.any(a1[ambiguous] != observed[ambiguous]):
            raise RuntimeError(
                "Operational A1 does not retain observed bytes "
                "at AMBIGUOUS coordinates"
            )

        if np.any(a1[~mask] != observed[~mask]):
            raise RuntimeError(
                "Operational A1 has outside-mask drift"
            )

        y, x = np.where(eligible_mask)
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
            clean_mask = np.zeros(
                observed.shape,
                dtype=bool,
            )

            corrected = self._assemble_safe_cleanup(
                observed=observed,
                a1=a1,
                paper=paper,
                mask=mask,
                ambiguous=ambiguous,
                eligible=eligible_mask,
                clean_mask=clean_mask,
            )

            return {
                "eligible": eligible_mask,
                "probability": probability_map,
                "provisional": provisional_map,
                "final": final_map,
                "y": y,
                "x": x,
                "clean_mask": clean_mask,
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
                "Safe-cleanup probability shape mismatch: "
                f"expected {(len(y),)}, "
                f"found {probability.shape}"
            )

        if not np.isfinite(probability).all():
            raise RuntimeError(
                "Safe-cleanup probability contains "
                "non-finite values"
            )

        if (
            np.any(probability < 0.0)
            or np.any(probability > 1.0)
        ):
            raise RuntimeError(
                "Safe-cleanup probability lies outside 0..1"
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
                decision_result,
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

        clean_at_coordinates = (
            np.char.upper(
                final.astype(str)
            )
            == "CLEAN"
        )

        clean_mask = np.zeros(
            observed.shape,
            dtype=bool,
        )
        clean_mask[
            y[clean_at_coordinates],
            x[clean_at_coordinates],
        ] = True

        if np.any(clean_mask & ~eligible_mask):
            raise RuntimeError(
                "CLEAN decision exists outside A5 eligibility"
            )

        corrected = self._assemble_safe_cleanup(
            observed=observed,
            a1=a1,
            paper=paper,
            mask=mask,
            ambiguous=ambiguous,
            eligible=eligible_mask,
            clean_mask=clean_mask,
        )

        return {
            "eligible": eligible_mask,
            "probability": probability_map,
            "provisional": provisional_map,
            "final": final_map,
            "y": y,
            "x": x,
            "clean_mask": clean_mask,
            "corrected": corrected,
        }

    # ------------------------------------------------------------------
    # Decision normalisation
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
                    "decisions() dictionary must contain "
                    "'provisional' and 'final'"
                )

            provisional = decision_result[
                "provisional"
            ]
            final = decision_result["final"]

        elif (
            isinstance(decision_result, tuple)
            and len(decision_result) == 2
        ):
            provisional, final = decision_result

        else:
            raise RuntimeError(
                "decisions() must return either "
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
                "Provisional-decision shape mismatch: "
                f"{provisional.shape}"
            )

        if final.shape != (
            probability_count,
        ):
            raise RuntimeError(
                "Final-decision shape mismatch: "
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

        if truth_codes and np.issubdtype(
            values.dtype,
            np.number,
        ):
            mapping = {
                int(key): str(value).upper()
                for key, value in truth_codes.items()
            }

            unknown = sorted(
                set(values.astype(int).tolist())
                - set(mapping)
            )

            if unknown:
                raise RuntimeError(
                    f"{name} decisions contain unknown codes: "
                    f"{unknown}"
                )

            normalised = np.asarray(
                [
                    mapping[int(value)]
                    for value in values
                ],
                dtype=object,
            )
        else:
            normalised = np.char.upper(
                values.astype(str)
            ).astype(object)

        allowed = {
            "CLEAN",
            "KEEP",
            "UNCERTAIN",
        }

        unexpected = sorted(
            set(normalised.tolist())
            - allowed
        )

        if unexpected:
            raise RuntimeError(
                f"{name} decisions contain unexpected "
                f"classes: {unexpected}"
            )

        return normalised

    # ------------------------------------------------------------------
    # Operational A5 page assembly
    # ------------------------------------------------------------------

    def _assemble_safe_cleanup(
        self,
        observed: np.ndarray,
        a1: np.ndarray,
        paper: np.ndarray,
        mask: np.ndarray,
        ambiguous: np.ndarray,
        eligible: np.ndarray,
        clean_mask: np.ndarray,
    ) -> np.ndarray:
        if np.any(clean_mask & ~eligible):
            raise RuntimeError(
                "A5 CLEAN escapes the eligible population"
            )

        if np.any(clean_mask & ~ambiguous):
            raise RuntimeError(
                "A5 CLEAN includes non-AMBIGUOUS pixels"
            )

        if np.any(clean_mask & ~mask):
            raise RuntimeError(
                "A5 CLEAN escapes the prepared mask"
            )

        corrected = a1.copy()
        corrected[clean_mask] = paper[clean_mask]

        if np.any(
            corrected[clean_mask]
            != paper[clean_mask]
        ):
            raise RuntimeError(
                "A5 CLEAN output does not equal paper"
            )

        non_clean = ~clean_mask

        if np.any(
            corrected[non_clean]
            != a1[non_clean]
        ):
            raise RuntimeError(
                "A5 non-CLEAN output differs from A1"
            )

        if np.any(
            corrected[~ambiguous]
            != a1[~ambiguous]
        ):
            raise RuntimeError(
                "A5 changed a non-AMBIGUOUS pixel"
            )

        if np.any(
            corrected[~mask]
            != observed[~mask]
        ):
            raise RuntimeError(
                "A5 changed a pixel outside the "
                "prepared mask"
            )

        changed = corrected != a1

        expected_changed = (
            clean_mask
            & (paper != a1)
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
                "A5 delta does not equal effective CLEAN-to-"
                "paper changes: "
                f"unexpected={unexpected}, missing={missing}"
            )

        return corrected

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
                f"found {value.shape}"
            )

        if expected_shape is not None:
            if value.shape != expected_shape:
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

        if value.min() < 0 or value.max() > 255:
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
                f"found {value.shape}"
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