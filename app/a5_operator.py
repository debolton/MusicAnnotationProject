from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
A5_V88_BUNDLE = ROOT / "frozen" / "a5_v88"

EXPECTED_CHANNELS = [
    "observed",
    "local_normalised",
    "reviewed_mask",
    "mask_depth",
    "ambiguous",
    "p_annotation",
    "p_print_value",
    "p_print_available",
    "edge",
    "horizontal_support",
    "vertical_support",
    "compact_support",
]

EXPECTED_PATCH_SIZE = 33
EXPECTED_POLICY_ID = "A5-v88-contextual-safe-cleanup"

ALLOWED_DECISIONS = {
    "CLEAN",
    "KEEP",
    "UNCERTAIN",
}


class A5Operator:
    """
    Notebook 88 contextual safe-cleanup operator.

    Eligible population:
        inside the authoritative reviewed mask
        AND V7.8 AMBIGUOUS
        AND not V7.8 PRESERVE-routed

    Assembly:
        CLEAN      -> exact frozen paper estimate byte
        KEEP       -> exact Notebook 87 A1 byte
        UNCERTAIN  -> exact Notebook 87 A1 byte
        outside authority -> exact observed byte
    """

    a5_policy_id = EXPECTED_POLICY_ID

    def __init__(
        self,
        device: str | None = None,
        bundle_dir: Path | str = A5_V88_BUNDLE,
        run_startup_replay: bool = True,
    ):
        self.bundle_dir = Path(bundle_dir)

        if not self.bundle_dir.is_dir():
            raise RuntimeError(
                "Missing A5-v88 bundle directory: "
                f"{self.bundle_dir}"
            )

        runtime_path = self.bundle_dir / "inference_runtime.py"
        if not runtime_path.is_file():
            raise RuntimeError(
                "Missing A5-v88 inference_runtime.py: "
                f"{runtime_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "a5_v88_frozen_runtime",
            runtime_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                "Unable to create module specification "
                f"for {runtime_path}"
            )

        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        self.rt = runtime

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
                "A5-v88 runtime API is incomplete: "
                f"{missing}"
            )

        loaded = self.rt.load_bundle(
            self.bundle_dir,
            device=device,
        )

        if not isinstance(loaded, tuple) or len(loaded) != 5:
            raise RuntimeError(
                "A5-v88 load_bundle() must return exactly "
                "(model, calibrator, policy, schema, device)"
            )

        (
            self.model,
            self.calibrator,
            self.policy,
            self.schema,
            self.device,
        ) = loaded

        self._validate_bundle_contract()

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
            replay = getattr(self.rt, "run_replay", None)
            if callable(replay):
                result = replay(
                    self.bundle_dir,
                    self.model,
                    self.calibrator,
                    self.policy,
                    self.schema,
                    self.device,
                )
                if result is False:
                    raise RuntimeError(
                        "Frozen A5-v88 startup replay failed"
                    )
                if isinstance(result, dict) and result.get(
                    "passed"
                ) is False:
                    raise RuntimeError(
                        "Frozen A5-v88 startup replay failed"
                    )
                self.startup_replay = result
            else:
                raise RuntimeError(
                    "A5-v88 startup replay is mandatory"
                )

    def _validate_bundle_contract(self) -> None:
        if not isinstance(self.policy, dict):
            raise RuntimeError(
                "A5-v88 policy must be a dictionary"
            )
        if not isinstance(self.schema, dict):
            raise RuntimeError(
                "A5-v88 schema must be a dictionary"
            )

        if self.schema.get("channels") != EXPECTED_CHANNELS:
            raise RuntimeError(
                "A5-v88 channel order differs from the "
                "Notebook 88 12-channel contract: "
                f"{self.schema.get('channels')!r}"
            )

        patch = self.schema.get(
            "patch",
            self.schema.get("patch_size"),
        )
        if int(patch if patch is not None else -1) != (
            EXPECTED_PATCH_SIZE
        ):
            raise RuntimeError(
                "A5-v88 patch size differs from "
                f"{EXPECTED_PATCH_SIZE}x{EXPECTED_PATCH_SIZE}"
            )

        required_policy = {
            "policy_id",
            "clean_threshold",
            "minimum_clean_run",
            "connectivity",
            "clean_action",
            "non_clean_action",
            "outside_authoritative_mask",
        }
        missing = sorted(required_policy - set(self.policy))
        if missing:
            raise RuntimeError(
                "A5-v88 policy is missing fields: "
                f"{missing}"
            )

        if str(self.policy["policy_id"]) != (
            EXPECTED_POLICY_ID
        ):
            raise RuntimeError(
                "Unexpected A5-v88 policy ID"
            )

        threshold = float(
            self.policy["clean_threshold"]
        )
        if not 0.0 <= threshold <= 1.0:
            raise RuntimeError(
                "A5-v88 clean_threshold must be 0..1"
            )
        if int(self.policy["minimum_clean_run"]) < 1:
            raise RuntimeError(
                "A5-v88 minimum_clean_run must be >= 1"
            )
        if int(self.policy["connectivity"]) not in (4, 8):
            raise RuntimeError(
                "A5-v88 connectivity must be 4 or 8"
            )
        if self.policy["clean_action"] != "COPY_PAPER_BYTE":
            raise RuntimeError(
                "A5-v88 clean_action must be COPY_PAPER_BYTE"
            )
        if self.policy["non_clean_action"] != "KEEP_A1_BYTE":
            raise RuntimeError(
                "A5-v88 non_clean_action must be KEEP_A1_BYTE"
            )
        if self.policy[
            "outside_authoritative_mask"
        ] != "IMMUTABLE_OBSERVED":
            raise RuntimeError(
                "A5-v88 outside-authoritative-mask contract "
                "must be IMMUTABLE_OBSERVED"
            )

        self.a5_policy_id = str(
            self.policy["policy_id"]
        )

    def run_from_a1(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        a1_result: dict,
        page: str | None = None,
        batch: int = 512,
    ) -> dict:
        del page

        observed = self._require_uint8(
            observed,
            "observed page",
        )
        seed = self._require_uint8(
            reviewed_seed,
            "reviewed authoritative seed",
            observed.shape,
        )

        if not isinstance(a1_result, dict):
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
        missing = sorted(required - set(a1_result))
        if missing:
            raise RuntimeError(
                "A1 result omitted fields required by "
                f"A5-v88: {missing}"
            )

        a1_observed = self._require_uint8(
            a1_result["observed"],
            "A1 observed page",
            observed.shape,
        )
        if not np.array_equal(a1_observed, observed):
            raise RuntimeError(
                "A1 observed page differs from the page "
                "supplied to A5-v88"
            )

        a1_seed = self._require_uint8(
            a1_result["reviewed_seed"],
            "A1 reviewed seed",
            observed.shape,
        )
        if not np.array_equal(a1_seed > 0, seed > 0):
            raise RuntimeError(
                "A1 reviewed seed differs from the seed "
                "supplied to A5-v88"
            )

        authority = self._require_bool(
            a1_result["mask"],
            "authoritative mask",
            observed.shape,
        )
        telea = self._require_uint8(
            a1_result["telea"],
            "Telea diagnostic",
            observed.shape,
        )
        paper = self._require_uint8(
            a1_result["paper"],
            "paper estimate",
            observed.shape,
        )
        local_normalised = self._require_uint8(
            a1_result["normalised"],
            "background-normalised page",
            observed.shape,
        )
        preserve = self._require_bool(
            a1_result["preserve"],
            "V7.8 PRESERVE route",
            observed.shape,
        )
        remove = self._require_bool(
            a1_result["remove"],
            "V7.8 REMOVE route",
            observed.shape,
        )
        ambiguous = self._require_bool(
            a1_result["ambiguous"],
            "V7.8 AMBIGUOUS route",
            observed.shape,
        )
        horizontal = self._require_bool(
            a1_result["horizontal"],
            "V7.8 horizontal support",
            observed.shape,
        )
        vertical = self._require_bool(
            a1_result["vertical"],
            "V7.8 vertical support",
            observed.shape,
        )
        compact = self._require_bool(
            a1_result["compact"],
            "V7.8 compact support",
            observed.shape,
        )
        geometry_t80 = self._require_uint8(
            a1_result["geometry_t80"],
            "V7.8 geometry_t80",
            observed.shape,
        )
        v89 = self._require_uint8(
            a1_result["v8_9_corrected"],
            "V8.9 corrected page",
            observed.shape,
        )
        a1 = self._require_uint8(
            a1_result["a1"],
            "Notebook 87 A1",
            observed.shape,
        )

        routes = a1_result["routes"]
        if not isinstance(routes, dict):
            raise RuntimeError(
                "a1_result['routes'] must be a dictionary"
            )
        for name in ("preserve", "remove", "ambiguous"):
            if name not in routes:
                raise RuntimeError(
                    f"a1_result['routes'] omits {name}"
                )

        routed = preserve | remove | ambiguous
        if not np.array_equal(routed, authority):
            raise RuntimeError(
                "V7.8 routes do not exactly partition "
                "the authoritative mask"
            )
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

        if np.any(a1[~authority] != observed[~authority]):
            raise RuntimeError(
                "Notebook 87 A1 changed pixels outside "
                "the authoritative mask"
            )

        edge = (
            cv2.Canny(observed, 70, 170) > 0
        )
        mask_depth = cv2.distanceTransform(
            authority.astype(np.uint8),
            cv2.DIST_L2,
            5,
        ).astype(np.float32)

        p_annotation = np.asarray(
            a1_result.get("p_annotation"),
            dtype=np.float32,
        ) if "p_annotation" in a1_result else np.zeros(
            observed.shape,
            dtype=np.float32,
        )

        p_print_value = np.asarray(
            a1_result.get("p_print_value"),
            dtype=np.float32,
        ) if "p_print_value" in a1_result else np.zeros(
            observed.shape,
            dtype=np.float32,
        )

        p_print_available = np.asarray(
            a1_result.get(
                "p_print_available",
                np.zeros(observed.shape, dtype=np.uint8),
            )
        ).astype(bool)

        if p_annotation.shape != observed.shape:
            raise RuntimeError(
                "p_annotation shape mismatch"
            )
        if p_print_value.shape != observed.shape:
            raise RuntimeError(
                "p_print_value shape mismatch"
            )
        if p_print_available.shape != observed.shape:
            raise RuntimeError(
                "p_print_available shape mismatch"
            )

        channels = {
            "observed": observed,
            "local_normalised": local_normalised,
            "reviewed_mask": authority,
            "mask_depth": mask_depth,
            "ambiguous": ambiguous,
            "p_annotation": p_annotation,
            "p_print_value": p_print_value,
            "p_print_available": p_print_available.astype(
                np.uint8
            ),
            "edge": edge,
            "horizontal_support": horizontal,
            "vertical_support": vertical,
            "compact_support": compact,
        }

        if list(channels) != EXPECTED_CHANNELS:
            raise RuntimeError(
                "Constructed A5-v88 channel order differs "
                "from the Notebook 88 contract"
            )

        result = self.process_arrays(
            arrays=channels,
            baseline=a1,
            paper=paper,
            authoritative_mask=authority,
            ambiguous_mask=ambiguous,
            preserve_mask=preserve,
            batch=batch,
        )

        a5 = self._require_uint8(
            result["corrected"],
            "A5-v88 corrected page",
            observed.shape,
        )
        clean_mask = self._require_bool(
            result["clean_mask"],
            "A5-v88 CLEAN mask",
            observed.shape,
        )

        if np.any(a5[clean_mask] != paper[clean_mask]):
            raise RuntimeError(
                "A5-v88 CLEAN output does not equal "
                "the exact paper byte"
            )
        if np.any(a5[~clean_mask] != a1[~clean_mask]):
            raise RuntimeError(
                "A5-v88 non-CLEAN output differs "
                "from Notebook 87 A1"
            )
        if np.any(a5[~authority] != observed[~authority]):
            raise RuntimeError(
                "A5-v88 changed pixels outside authority"
            )

        return {
            **a1_result,
            "mask": authority,
            "telea": telea,
            "paper": paper,
            "normalised": local_normalised,
            "horizontal": horizontal,
            "vertical": vertical,
            "compact": compact,
            "preserve": preserve,
            "remove": remove,
            "ambiguous": ambiguous,
            "channels": channels,
            "depth": mask_depth,
            "edge": edge,
            "routes": {
                **routes,
                "preserve": preserve,
                "remove": remove,
                "ambiguous": ambiguous,
            },
            "geometry_t80": geometry_t80,
            "v8_9_corrected": v89,
            "a1": a1,
            "a5": a5,
            "eligible_a5_v88": result["eligible"],
            "p_clean": result["probability"],
            "raw_p_clean": result["raw_probability"],
            "provisional": result["provisional"],
            "final": result["final"],
            "clean_mask": clean_mask,
            "a5_clean_mask": clean_mask,
            "a5_ys": result["y"],
            "a5_xs": result["x"],
            "a5_delta_from_a1": a5 != a1,
            "a5_delta_from_v7_8": a5 != geometry_t80,
            "a5_policy_id": self.a5_policy_id,
            "bundle_id": self.bundle_id,
            "model_sha256": self.model_sha256,
            "calibrator_sha256": self.calibrator_sha256,
            "a5_action": "COPY_PAPER_BYTE",
            "a5_fallback": "Notebook87 A1",
        }

    def run(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        page: str | None = None,
        batch: int = 512,
    ) -> dict:
        del observed
        del reviewed_seed
        del page
        del batch
        raise RuntimeError(
            "A5-v88.run() is disabled. "
            "A5-v88 requires a precomputed Notebook 87 "
            "A1 result. Call run_from_a1()."
        )

    def process_arrays(
        self,
        arrays: dict,
        baseline: np.ndarray,
        paper: np.ndarray,
        authoritative_mask: np.ndarray,
        ambiguous_mask: np.ndarray,
        preserve_mask: np.ndarray,
        batch: int = 512,
    ) -> dict:
        if not isinstance(arrays, dict):
            raise RuntimeError(
                "A5-v88 arrays must be a dictionary"
            )
        if not isinstance(batch, int) or batch < 1:
            raise ValueError(
                "batch must be a positive integer"
            )

        observed = self._require_uint8(
            arrays["observed"],
            "observed",
        )
        baseline = self._require_uint8(
            baseline,
            "A5-v88 fallback baseline",
            observed.shape,
        )
        paper = self._require_uint8(
            paper,
            "A5-v88 paper estimate",
            observed.shape,
        )
        mask = self._require_bool(
            authoritative_mask,
            "authoritative mask",
            observed.shape,
        )
        ambiguous = self._require_bool(
            ambiguous_mask,
            "AMBIGUOUS route",
            observed.shape,
        )
        preserve = self._require_bool(
            preserve_mask,
            "PRESERVE route",
            observed.shape,
        )

        required_channels = list(
            self.schema["channels"]
        )
        missing = [
            name
            for name in required_channels
            if name not in arrays
        ]
        if missing:
            raise RuntimeError(
                "A5-v88 arrays omit channels: "
                f"{missing}"
            )

        for name in required_channels:
            array = np.asarray(arrays[name])
            if array.shape != observed.shape:
                raise RuntimeError(
                    f"A5-v88 channel {name} shape mismatch"
                )
            if (
                np.issubdtype(array.dtype, np.number)
                and not np.isfinite(array).all()
            ):
                raise RuntimeError(
                    f"A5-v88 channel {name} contains "
                    "non-finite values"
                )

        schema_mask = self._require_bool(
            arrays["reviewed_mask"],
            "schema reviewed_mask",
            observed.shape,
        )
        if not np.array_equal(schema_mask, mask):
            raise RuntimeError(
                "Channel reviewed_mask differs from "
                "the authoritative mask"
            )

        schema_ambiguous = self._require_bool(
            arrays["ambiguous"],
            "schema ambiguous",
            observed.shape,
        )
        if not np.array_equal(schema_ambiguous, ambiguous):
            raise RuntimeError(
                "Channel ambiguous differs from "
                "the AMBIGUOUS route"
            )

        if np.any(baseline[~mask] != observed[~mask]):
            raise RuntimeError(
                "A5-v88 fallback baseline has "
                "outside-mask drift"
            )
        if np.any(paper[~mask] != observed[~mask]):
            raise RuntimeError(
                "A5-v88 paper estimate has "
                "outside-mask drift"
            )

        eligible = (
            mask
            & ambiguous
            & ~preserve
        )

        y, x = np.where(eligible)
        y = y.astype(np.int64)
        x = x.astype(np.int64)

        probability_map = np.full(
            observed.shape,
            np.nan,
            dtype=np.float32,
        )
        raw_probability_map = np.full(
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
            corrected = baseline.copy()

            self._validate_assembly(
                observed=observed,
                baseline=baseline,
                paper=paper,
                authority=mask,
                eligible=eligible,
                clean_mask=clean_mask,
                corrected=corrected,
            )

            return {
                "eligible": eligible,
                "probability": probability_map,
                "raw_probability": raw_probability_map,
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

        score_result = self.rt.score_coordinates(
            self.model,
            self.calibrator,
            padded,
            self.schema,
            y,
            x,
            self.device,
            batch=batch,
            return_raw=True,
        )

        if not isinstance(score_result, tuple):
            raise RuntimeError(
                "A5-v88 score_coordinates(return_raw=True) "
                "must return (calibrated, raw)"
            )
        if len(score_result) != 2:
            raise RuntimeError(
                "A5-v88 score_coordinates(return_raw=True) "
                "must return (calibrated, raw)"
            )

        probability = np.asarray(
            score_result[0],
            dtype=np.float64,
        )
        raw_probability = np.asarray(
            score_result[1],
            dtype=np.float64,
        )

        if probability.shape != (len(y),):
            raise RuntimeError(
                "A5-v88 CLEAN probability shape mismatch"
            )
        if raw_probability.shape != (len(y),):
            raise RuntimeError(
                "A5-v88 raw CLEAN probability shape mismatch"
            )
        if not np.isfinite(probability).all():
            raise RuntimeError(
                "A5-v88 CLEAN probability contains "
                "non-finite values"
            )
        if not np.isfinite(raw_probability).all():
            raise RuntimeError(
                "A5-v88 raw CLEAN probability contains "
                "non-finite values"
            )
        if (
            np.any(probability < 0.0)
            or np.any(probability > 1.0)
            or np.any(raw_probability < 0.0)
            or np.any(raw_probability > 1.0)
        ):
            raise RuntimeError(
                "A5-v88 probability lies outside 0..1"
            )

        decisions = self.rt.decisions(
            probability,
            self.policy,
            observed.shape,
            y,
            x,
        )

        if isinstance(decisions, dict):
            provisional = np.asarray(
                decisions["provisional"]
            )
            final = np.asarray(
                decisions["final"]
            )
        elif isinstance(decisions, tuple):
            provisional = np.asarray(decisions[0])
            final = np.asarray(decisions[1])
        else:
            raise RuntimeError(
                "A5-v88 decisions() must return "
                "(provisional, final)"
            )

        if provisional.shape != (len(y),):
            raise RuntimeError(
                "A5-v88 provisional shape mismatch"
            )
        if final.shape != (len(y),):
            raise RuntimeError(
                "A5-v88 final shape mismatch"
            )

        provisional = np.asarray(
            [
                str(value).upper()
                for value in provisional
            ],
            dtype=object,
        )
        final = np.asarray(
            [
                str(value).upper()
                for value in final
            ],
            dtype=object,
        )

        unexpected = (
            set(provisional.tolist())
            | set(final.tolist())
        ) - ALLOWED_DECISIONS
        if unexpected:
            raise RuntimeError(
                "A5-v88 unexpected classes: "
                f"{sorted(unexpected)}"
            )

        probability_map[y, x] = probability.astype(
            np.float32
        )
        raw_probability_map[y, x] = (
            raw_probability.astype(np.float32)
        )
        provisional_map[y, x] = provisional
        final_map[y, x] = final

        corrected, clean_mask = self.rt.assemble(
            observed,
            baseline,
            paper,
            mask,
            y,
            x,
            final,
        )

        corrected = self._require_uint8(
            corrected,
            "A5-v88 corrected page",
            observed.shape,
        )
        clean_mask = self._require_bool(
            clean_mask,
            "A5-v88 CLEAN mask",
            observed.shape,
        )

        selected = final == "CLEAN"
        expected_clean_mask = np.zeros(
            observed.shape,
            dtype=bool,
        )
        expected_clean_mask[y[selected], x[selected]] = True

        if not np.array_equal(
            clean_mask,
            expected_clean_mask,
        ):
            raise RuntimeError(
                "A5-v88 runtime CLEAN mask differs "
                "from final CLEAN decisions"
            )

        self._validate_assembly(
            observed=observed,
            baseline=baseline,
            paper=paper,
            authority=mask,
            eligible=eligible,
            clean_mask=clean_mask,
            corrected=corrected,
        )

        return {
            "eligible": eligible,
            "probability": probability_map,
            "raw_probability": raw_probability_map,
            "provisional": provisional_map,
            "final": final_map,
            "y": y,
            "x": x,
            "clean_mask": clean_mask,
            "corrected": corrected,
        }

    @staticmethod
    def _validate_assembly(
        observed: np.ndarray,
        baseline: np.ndarray,
        paper: np.ndarray,
        authority: np.ndarray,
        eligible: np.ndarray,
        clean_mask: np.ndarray,
        corrected: np.ndarray,
    ) -> None:
        if np.any(clean_mask & ~eligible):
            raise RuntimeError(
                "A5-v88 CLEAN escapes eligible population"
            )
        if np.any(clean_mask & ~authority):
            raise RuntimeError(
                "A5-v88 CLEAN escapes authoritative mask"
            )
        if np.any(corrected[~authority] != observed[~authority]):
            raise RuntimeError(
                "A5-v88 outside-authority drift"
            )
        if np.any(corrected[clean_mask] != paper[clean_mask]):
            raise RuntimeError(
                "A5-v88 CLEAN byte does not equal paper byte"
            )
        if np.any(corrected[~clean_mask] != baseline[~clean_mask]):
            raise RuntimeError(
                "A5-v88 non-CLEAN byte differs from A1"
            )

        changed = corrected != baseline
        expected_changed = (
            clean_mask
            & (paper != baseline)
        )
        if not np.array_equal(changed, expected_changed):
            raise RuntimeError(
                "A5-v88 page delta is not exactly "
                "effective CLEAN restoration"
            )

    @staticmethod
    def _require_uint8(
        value: np.ndarray,
        name: str,
        expected_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim != 2:
            raise RuntimeError(
                f"{name} must be two-dimensional"
            )
        if (
            expected_shape is not None
            and value.shape != expected_shape
        ):
            raise RuntimeError(
                f"{name} shape mismatch"
            )
        if not np.issubdtype(value.dtype, np.number):
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
    def _require_bool(
        value: np.ndarray,
        name: str,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim != 2:
            raise RuntimeError(
                f"{name} must be two-dimensional"
            )
        if value.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch"
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