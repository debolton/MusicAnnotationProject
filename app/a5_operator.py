"""A5-v1 observed-print operator.

This module is a clean implementation of the frozen A5-v1 runtime defined by
Notebook 91. It is inference-only: it never loads clean truth, trains, refits,
or recalibrates the model.

Frozen bundle convention:
    ROOT/frozen/a5_v1/
        model.pt
        model_config.json
        isotonic_calibrator.joblib
        channel_schema.json
        policy.json

The operator starts from the A1 corrected page and copies exact observed bytes
only at final PRINT coordinates. ANNOTATION and UNCERTAIN retain A1 bytes.
Pixels outside the authoritative mask must remain byte-for-byte equal to the
observed page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import joblib
import numpy as np
import torch
from torch import nn


DEFAULT_CHANNELS = (
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
)


class A5OperatorError(RuntimeError):
    """Fail-closed A5 operator error."""


class TopologyCNN(nn.Module):
    """The frozen 33x33 compact CNN architecture used by A5-v1."""

    def __init__(self, num_channels: int = 11) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 24, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1))


@dataclass(frozen=True)
class A5Result:
    image: np.ndarray
    probability: np.ndarray
    provisional_decision: np.ndarray
    final_decision: np.ndarray
    eligible_y: np.ndarray
    eligible_x: np.ndarray
    print_mask: np.ndarray


@dataclass(frozen=True)
class A5Bundle:
    root: Path
    model: TopologyCNN
    calibrator: Any
    policy: Mapping[str, Any]
    schema: Mapping[str, Any]
    model_config: Mapping[str, Any]
    device: torch.device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise A5OperatorError(f"Missing frozen bundle file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A5OperatorError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A5OperatorError(f"Expected a JSON object in {path}")
    return value


def resolve_bundle_dir(root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve ROOT/frozen/a5_v1, with A5_ROOT as an optional environment override."""
    base = Path(root or os.environ.get("A5_ROOT", ".")).expanduser().resolve()
    candidate = base / "frozen" / "a5_v1"
    if candidate.is_dir():
        return candidate
    # Also permit callers to pass the bundle directory itself.
    if base.is_dir() and (base / "policy.json").is_file():
        return base
    raise A5OperatorError(
        f"Frozen A5-v1 bundle not found. Expected {candidate} "
        "or pass the bundle directory directly."
    )


def load_bundle(
    root: str | os.PathLike[str] | None = None,
    device: str | torch.device | None = None,
) -> A5Bundle:
    bundle_dir = resolve_bundle_dir(root)
    policy = _read_json(bundle_dir / "policy.json")
    schema = _read_json(bundle_dir / "channel_schema.json")
    model_config = _read_json(bundle_dir / "model_config.json")

    required = (
        bundle_dir / "model.pt",
        bundle_dir / "isotonic_calibrator.joblib",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise A5OperatorError(f"Incomplete A5-v1 bundle; missing: {missing}")

    channels = schema.get("channels")
    patch = schema.get("patch")
    if channels != list(DEFAULT_CHANNELS):
        raise A5OperatorError(
            f"Unexpected A5 channel schema: {channels!r}; expected {list(DEFAULT_CHANNELS)!r}"
        )
    if int(patch) != 33:
        raise A5OperatorError(f"Unexpected A5 patch size: {patch!r}; expected 33")

    expected_policy = "A5-v1-print-p0980-margin0960-run2"
    if policy.get("policy_id") != expected_policy:
        raise A5OperatorError(
            f"Unexpected policy_id {policy.get('policy_id')!r}; expected {expected_policy!r}"
        )
    if policy.get("clean_image_permitted_at_runtime") is not False:
        raise A5OperatorError("Frozen policy does not explicitly prohibit clean images at runtime")

    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = TopologyCNN(len(channels)).to(selected_device)
    try:
        state = torch.load(
            bundle_dir / "model.pt",
            map_location=selected_device,
            weights_only=True,
        )
    except TypeError:  # Compatibility with older PyTorch.
        state = torch.load(bundle_dir / "model.pt", map_location=selected_device)
    model.load_state_dict(state, strict=True)
    model.eval()

    try:
        calibrator = joblib.load(bundle_dir / "isotonic_calibrator.joblib")
    except Exception as exc:
        raise A5OperatorError(f"Cannot load frozen calibrator: {exc}") from exc

    return A5Bundle(
        root=bundle_dir,
        model=model,
        calibrator=calibrator,
        policy=policy,
        schema=schema,
        model_config=model_config,
        device=selected_device,
    )


def _as_2d(name: str, value: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise A5OperatorError(f"{name}: expected a 2D array, got {array.shape}")
    if shape is not None and array.shape != shape:
        raise A5OperatorError(f"{name}: shape {array.shape} does not match {shape}")
    if not np.isfinite(array.astype(np.float32, copy=False)).all():
        raise A5OperatorError(f"{name}: contains non-finite values")
    return array


def prepare_channels(
    channels: Mapping[str, np.ndarray], schema: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    names = schema.get("channels")
    patch = int(schema.get("patch", 0))
    if not isinstance(names, list) or not names or patch <= 0 or patch % 2 != 1:
        raise A5OperatorError("Invalid frozen channel schema")

    missing = [name for name in names if name not in channels]
    extras = sorted(set(channels) - set(names))
    if missing:
        raise A5OperatorError(f"Missing A5 channels: {missing}")
    if extras:
        raise A5OperatorError(f"Unexpected A5 channels: {extras}")

    first = _as_2d(names[0], channels[names[0]])
    shape = first.shape
    half = patch // 2
    padded: dict[str, np.ndarray] = {}
    for name in names:
        array = _as_2d(name, channels[name], shape).astype(np.float32)
        if name in ("obs", "norm"):
            array /= 255.0
        padded[name] = np.pad(array, half, mode="reflect")
    return padded, shape


def eligible_coordinates(
    observed: np.ndarray,
    authoritative_mask: np.ndarray,
    preserve: np.ndarray,
    dark_threshold: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    observed = _as_2d("observed", observed)
    shape = observed.shape
    mask = _as_2d("authoritative_mask", authoritative_mask, shape) > 0
    preserve_mask = _as_2d("preserve", preserve, shape) > 0
    eligible = mask & (observed < int(dark_threshold)) & ~preserve_mask
    return tuple(a.astype(np.int32, copy=False) for a in np.where(eligible))


def score_coordinates_with_raw(
    bundle: A5Bundle,
    padded: Mapping[str, np.ndarray],
    ys: Sequence[int] | np.ndarray,
    xs: Sequence[int] | np.ndarray,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen raw and calibrated PRINT probabilities."""
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)
    if ys.shape != xs.shape or ys.ndim != 1:
        raise A5OperatorError("ys and xs must be equally sized one-dimensional arrays")
    if batch_size < 1:
        raise A5OperatorError("batch_size must be positive")
    if len(ys) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy()

    patch = int(bundle.schema["patch"])
    names = bundle.schema["channels"]
    raw_parts: list[np.ndarray] = []
    calibrated_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(ys), batch_size):
            stop = min(len(ys), start + batch_size)
            tensors = np.stack(
                [
                    np.stack(
                        [padded[name][y:y + patch, x:x + patch] for name in names]
                    )
                    for y, x in zip(ys[start:stop], xs[start:stop])
                ]
            ).astype(np.float32, copy=False)
            xb = torch.from_numpy(tensors).to(bundle.device)
            raw = (
                torch.softmax(bundle.model(xb), dim=1)[:, 1]
                .cpu().numpy().astype(np.float64, copy=False)
            )
            probability = np.asarray(bundle.calibrator.predict(raw), dtype=np.float64)
            if not np.isfinite(raw).all():
                raise A5OperatorError("Frozen model produced non-finite raw probabilities")
            if not np.isfinite(probability).all():
                raise A5OperatorError("Frozen calibrator produced non-finite probabilities")
            raw_parts.append(raw)
            calibrated_parts.append(probability)
    return np.concatenate(raw_parts), np.concatenate(calibrated_parts)


def score_coordinates(
    bundle: A5Bundle,
    padded: Mapping[str, np.ndarray],
    ys: Sequence[int] | np.ndarray,
    xs: Sequence[int] | np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Compatibility wrapper returning calibrated PRINT probabilities only."""
    _, calibrated = score_coordinates_with_raw(
        bundle, padded, ys, xs, batch_size=batch_size
    )
    return calibrated

def make_decisions(
    probability: np.ndarray,
    policy: Mapping[str, Any],
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)
    if not (len(p) == len(ys) == len(xs)):
        raise A5OperatorError("Probability and coordinate lengths do not match")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise A5OperatorError("A5 probabilities must be finite and within [0, 1]")

    q = 1.0 - p
    provisional = np.full(len(p), "UNCERTAIN", dtype="<U10")
    provisional[q >= float(policy["annotation_threshold"])] = "ANNOTATION"
    print_rule = (p >= float(policy["print_threshold"])) & (
        (p - q) >= float(policy["print_margin"])
    )
    provisional[print_rule] = "PRINT"
    final = provisional.copy()

    print_map = np.zeros(shape, dtype=np.uint8)
    print_indices = np.flatnonzero(provisional == "PRINT")
    print_map[ys[print_indices], xs[print_indices]] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        print_map, int(policy.get("connectivity", 8))
    )
    minimum_run = int(policy["minimum_print_run"])
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_run:
            continue
        component_y, component_x = np.where(labels == component_id)
        keys = set(zip(component_y.tolist(), component_x.tolist()))
        drop = [
            index
            for index in print_indices
            if (int(ys[index]), int(xs[index])) in keys
        ]
        final[drop] = "UNCERTAIN"
    return provisional, final


def assemble(
    observed: np.ndarray,
    a1: np.ndarray,
    authoritative_mask: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    final_decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = _as_2d("observed", observed).astype(np.uint8, copy=False)
    a1 = _as_2d("a1", a1, observed.shape).astype(np.uint8, copy=False)
    mask = _as_2d("authoritative_mask", authoritative_mask, observed.shape) > 0
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)
    decisions = np.asarray(final_decisions)

    output = a1.copy()
    selected = np.flatnonzero(decisions == "PRINT")
    py, px = ys[selected], xs[selected]
    output[py, px] = observed[py, px]
    print_mask = np.zeros(mask.shape, dtype=bool)
    print_mask[py, px] = True

    if np.any(output[~mask] != observed[~mask]):
        raise A5OperatorError("Outside-mask drift")
    if np.any(output[~print_mask] != a1[~print_mask]):
        raise A5OperatorError("Non-PRINT A1 regression")
    if np.any((output != a1) & (output != observed)):
        raise A5OperatorError("Changed byte is not equal to the observed byte")
    return output, print_mask

def _decision_mismatch_count(
    actual: np.ndarray,
    expected: np.ndarray,
) -> int:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)

    if actual_array.shape != expected_array.shape:
        return -1

    return int(
        np.count_nonzero(actual_array != expected_array)
    )

class A5Operator:
    def __init__(
        self,
        bundle_dir=None,
        *,
        device=None,
        batch_size=512,
        run_startup_replay=True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.run_startup_replay = bool(run_startup_replay)

        if self.batch_size <= 0:
            raise A5OperatorError(
                "batch_size must be greater than zero"
            )

        # load_bundle() already:
        # - resolves ROOT/frozen/a5_v1 or a direct bundle directory;
        # - validates policy and channel schema;
        # - loads the model and calibrator;
        # - selects the Torch device.
        self.bundle = load_bundle(
            root=bundle_dir,
            device=device,
        )

        # Use the resolved bundle directory from the loaded bundle.
        self.bundle_dir = self.bundle.root
        self.device = self.bundle.device
        self.a5_policy_id = str(self.bundle.policy.get("policy_id", ""))
        self.bundle_id = str(
            self.bundle.policy.get("bundle_id")
            or self.bundle.model_config.get("bundle_id")
            or self.a5_policy_id
        )
        self.model_sha256 = _sha256_file(self.bundle_dir / "model.pt")
        self.calibrator_sha256 = _sha256_file(
            self.bundle_dir / "isotonic_calibrator.joblib"
        )

        self.startup_replay: dict[str, Any] | None = None

        if self.run_startup_replay:
            self.startup_replay = self._run_startup_replay()

    def run(
        self,
        channels: Mapping[str, np.ndarray],
        a1: np.ndarray,
        ys: np.ndarray | None = None,
        xs: np.ndarray | None = None,
    ) -> A5Result:
        padded, shape = prepare_channels(
            channels,
            self.bundle.schema,
        )

        observed = _as_2d(
            "obs",
            channels["obs"],
            shape,
        ).astype(np.uint8, copy=False)

        mask = (
            _as_2d(
                "mask",
                channels["mask"],
                shape,
            )
            > 0
        )

        a1_array = _as_2d(
            "a1",
            a1,
            shape,
        ).astype(np.uint8, copy=False)

        if ys is None and xs is None:
            ys, xs = eligible_coordinates(
                observed,
                mask,
                channels["preserve"],
            )
        elif ys is None or xs is None:
            raise A5OperatorError(
                "Provide both ys and xs, or neither"
            )
        else:
            ys = np.asarray(ys, dtype=np.int32)
            xs = np.asarray(xs, dtype=np.int32)

        ys = np.asarray(ys, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)

        if ys.ndim != 1 or xs.ndim != 1:
            raise A5OperatorError(
                "Eligible coordinates must be "
                "one-dimensional arrays"
            )

        if ys.shape != xs.shape:
            raise A5OperatorError(
                "Eligible y and x coordinate arrays "
                "must have the same shape"
            )

        if len(ys):
            if (
                np.any(ys < 0)
                or np.any(xs < 0)
                or np.any(ys >= shape[0])
                or np.any(xs >= shape[1])
            ):
                raise A5OperatorError(
                    "Eligible coordinates fall outside the page"
                )

            if np.any(~mask[ys, xs]):
                raise A5OperatorError(
                    "Eligible coordinates include pixels outside "
                    "the authoritative mask"
                )

        probability = score_coordinates(
            self.bundle,
            padded,
            ys,
            xs,
            batch_size=self.batch_size,
        )

        provisional, final = make_decisions(
            probability,
            self.bundle.policy,
            shape,
            ys,
            xs,
        )

        image, print_mask = assemble(
            observed,
            a1_array,
            mask,
            ys,
            xs,
            final,
        )

        return A5Result(
            image=image,
            probability=probability,
            provisional_decision=provisional,
            final_decision=final,
            eligible_y=ys,
            eligible_x=xs,
            print_mask=print_mask,
        )

    @staticmethod
    def _a1_value(a1_result: Any, *names: str, required: bool = True) -> Any:
        for name in names:
            if isinstance(a1_result, Mapping) and name in a1_result:
                return a1_result[name]
            if hasattr(a1_result, name):
                return getattr(a1_result, name)
        if required:
            raise A5OperatorError(
                f"A1 result omitted required field; accepted names={names}"
            )
        return None

    def run_from_a1(
        self,
        observed: np.ndarray,
        reviewed_seed: np.ndarray,
        a1_result: Any,
        page: str | None = None,
    ) -> dict[str, Any]:
        """Run A5-v1 from an exact Notebook 87 A1 result.

        This is the application-facing adapter used by ``main.py``. It builds
        the frozen 11 A5 channels from A1/V7.8 evidence, runs A5, expands the
        coordinate results to full-page audit arrays, and preserves the full
        upstream lineage in the returned dictionary.
        """
        observed_u8 = _as_2d("observed", observed).astype(np.uint8, copy=False)
        shape = observed_u8.shape
        seed_u8 = _as_2d("reviewed_seed", reviewed_seed, shape).astype(
            np.uint8, copy=False
        )
        result_observed = self._a1_value(a1_result, "observed", "obs", required=False)
        if result_observed is not None and not np.array_equal(
            _as_2d("a1_result observed", result_observed, shape).astype(np.uint8),
            observed_u8,
        ):
            raise A5OperatorError("A1 result observed page differs from supplied observed page")

        mask = _as_2d(
            "authoritative_mask",
            self._a1_value(a1_result, "authoritative_mask", "mask"),
            shape,
        ) > 0
        a1 = _as_2d("a1", self._a1_value(a1_result, "a1", "corrected"), shape).astype(
            np.uint8, copy=False
        )
        normalised = _as_2d(
            "normalised", self._a1_value(a1_result, "normalised", "norm"), shape
        ).astype(np.uint8, copy=False)
        preserve = _as_2d("preserve", self._a1_value(a1_result, "preserve"), shape) > 0
        remove = _as_2d("remove", self._a1_value(a1_result, "remove"), shape) > 0
        ambiguous = _as_2d("ambiguous", self._a1_value(a1_result, "ambiguous"), shape) > 0
        horizontal = _as_2d("horizontal", self._a1_value(a1_result, "horizontal"), shape) > 0
        vertical = _as_2d("vertical", self._a1_value(a1_result, "vertical"), shape) > 0
        compact = _as_2d("compact", self._a1_value(a1_result, "compact"), shape) > 0
        if not np.array_equal(preserve | remove | ambiguous, mask):
            raise A5OperatorError("A1 V7.8 routes do not partition the authoritative mask")
        if np.any(a1[~mask] != observed_u8[~mask]):
            raise A5OperatorError("A1 contains outside-mask drift")

        channels = {
            "obs": observed_u8,
            "norm": normalised,
            "mask": mask.astype(np.uint8),
            "depth": cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5),
            "horizontal": horizontal.astype(np.uint8),
            "vertical": vertical.astype(np.uint8),
            "compact": compact.astype(np.uint8),
            "preserve": preserve.astype(np.uint8),
            "remove": remove.astype(np.uint8),
            "ambiguous": ambiguous.astype(np.uint8),
            "edge": (cv2.Canny(observed_u8, 70, 170) > 0).astype(np.uint8),
        }
        padded, _ = prepare_channels(channels, self.bundle.schema)
        ys, xs = eligible_coordinates(observed_u8, mask, preserve)
        raw, probability = score_coordinates_with_raw(
            self.bundle, padded, ys, xs, batch_size=self.batch_size
        )
        provisional_vector, final_vector = make_decisions(
            probability, self.bundle.policy, shape, ys, xs
        )
        image, print_mask = assemble(
            observed_u8, a1, mask, ys, xs, final_vector
        )

        raw_map = np.full(shape, np.nan, dtype=np.float32)
        probability_map = np.full(shape, np.nan, dtype=np.float32)
        provisional_map = np.full(shape, "INELIGIBLE", dtype="<U10")
        final_map = np.full(shape, "INELIGIBLE", dtype="<U10")
        raw_map[ys, xs] = raw.astype(np.float32)
        probability_map[ys, xs] = probability.astype(np.float32)
        provisional_map[ys, xs] = provisional_vector
        final_map[ys, xs] = final_vector
        eligible_map = np.zeros(shape, dtype=bool)
        eligible_map[ys, xs] = True

        routes = self._a1_value(a1_result, "routes", required=False)
        if not isinstance(routes, Mapping):
            routes = {}
        routes = {
            **dict(routes),
            "preserve": preserve,
            "remove": remove,
            "ambiguous": ambiguous,
            "horizontal": horizontal,
            "vertical": vertical,
            "compact": compact,
            "normalised": normalised,
        }
        output = dict(a1_result) if isinstance(a1_result, Mapping) else {}
        output.update({
            "page": page,
            "observed": observed_u8,
            "reviewed_seed": seed_u8,
            "mask": mask,
            "authoritative_mask": mask,
            "telea": self._a1_value(a1_result, "telea"),
            "paper": self._a1_value(a1_result, "paper"),
            "routes": routes,
            "geometry_t80": self._a1_value(
                a1_result, "geometry_t80", "baseline_v78"
            ),
            "v8_9_corrected": self._a1_value(a1_result, "v8_9_corrected", "v8_9"),
            "a1": a1,
            "a5": image,
            "channels": channels,
            "eligible_a5_v1": eligible_map,
            "raw_p_print": raw_map,
            "p_print": probability_map,
            "provisional": provisional_map,
            "final": final_map,
            "print_mask": print_mask,
            "a5_policy_id": self.a5_policy_id,
            "bundle_id": self.bundle_id,
            "model_sha256": self.model_sha256,
            "calibrator_sha256": self.calibrator_sha256,
        })
        for name in (
            "a1_policy_id", "v89_operator_sha256",
            "p_print_bundle_root", "remove_bundle_root", "a1_assembly"
        ):
            value = self._a1_value(a1_result, name, required=False)
            if value is not None:
                output[name] = value
        return output

    def _resolve_replay_dir(self) -> Path:
        """
        Locate the frozen replay directory.

        Supported layouts:

            frozen/a5_v1/
                policy.json
                ...
                replay/

        or:

            freeze_run/
                artefacts/
                    policy.json
                    ...
                replay/
        """
        candidates = [
            self.bundle_dir / "replay",
            self.bundle_dir.parent / "replay",
        ]

        # Additional compatibility where the bundle itself is called
        # artefacts and replay is its sibling.
        if self.bundle_dir.name == "artefacts":
            candidates.insert(
                0,
                self.bundle_dir.parent / "replay",
            )

        checked: list[str] = []

        for candidate in candidates:
            candidate = candidate.resolve()
            candidate_text = str(candidate)

            if candidate_text in checked:
                continue

            checked.append(candidate_text)

            if candidate.is_dir():
                return candidate

        raise A5OperatorError(
            "A5 startup replay directory was not found. "
            f"Checked: {checked}"
        )

    def _run_startup_replay(self) -> dict[str, Any]:
        """
        Run the frozen A5-v1 replay cases.

        The replay proves exact equality for:

        - calibrated probabilities;
        - provisional decisions;
        - final decisions after page-global run filtering;
        - final A5 page bytes.

        Any failure prevents the operator from starting.
        """
        replay_dir = self._resolve_replay_dir()
        contract_path = replay_dir / "replay_test.json"

        if not contract_path.is_file():
            raise A5OperatorError(
                "Missing A5 replay contract: "
                f"{contract_path}"
            )

        try:
            contract = json.loads(
                contract_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise A5OperatorError(
                "Cannot read A5 replay contract "
                f"{contract_path}: {exc}"
            ) from exc

        if not isinstance(contract, dict):
            raise A5OperatorError(
                "A5 replay contract must contain a JSON object"
            )

        policy_id = str(
            self.bundle.policy.get("policy_id", "")
        )

        replay_policy_id = str(
            contract.get("policy_id", "")
        )

        if not policy_id:
            raise A5OperatorError(
                "Loaded A5 policy has no policy_id"
            )

        if replay_policy_id != policy_id:
            raise A5OperatorError(
                "A5 replay policy mismatch: "
                f"bundle={policy_id!r}, "
                f"replay={replay_policy_id!r}"
            )

        if contract.get("passed") is not True:
            raise A5OperatorError(
                "Frozen A5 replay contract is not marked passed"
            )

        pages = contract.get("pages")

        if not isinstance(pages, list) or not pages:
            raise A5OperatorError(
                "A5 replay contract contains no replay pages"
            )

        checks: list[dict[str, Any]] = []

        for page_entry in pages:
            # Permit either:
            #
            #     "Page001"
            #
            # or:
            #
            #     {"page": "Page001"}
            if isinstance(page_entry, str):
                page = page_entry
            elif isinstance(page_entry, Mapping):
                page = str(
                    page_entry.get(
                        "page",
                        page_entry.get("name", ""),
                    )
                )
            else:
                raise A5OperatorError(
                    "Invalid A5 replay page entry: "
                    f"{page_entry!r}"
                )

            if not page:
                raise A5OperatorError(
                    "A5 replay page entry contains no page name"
                )

            page_dir = replay_dir / page

            inputs_path = (
                page_dir / "observable_inputs.npz"
            )
            expected_probability_path = (
                page_dir / "expected_probability.npy"
             )
            expected_provisional_path = (
                page_dir / "expected_provisional.npy"
            )
            expected_final_path = (
                page_dir / "expected_final.npy"
            )
            expected_a5_path = (
                page_dir / "expected_a5.png"
            )

            required_paths = [
                inputs_path,
                expected_probability_path,
                expected_provisional_path,
                expected_final_path,
                expected_a5_path,
            ]

            missing_paths = [
                str(path)
                for path in required_paths
                if not path.is_file()
            ]

            if missing_paths:
                raise A5OperatorError(
                    f"A5 replay page {page!r} is incomplete; "
                    f"missing: {missing_paths}"
                )

            try:
                channels, a1, ys, xs = _load_npz_inputs(
                    inputs_path
                )
            except Exception as exc:
                if isinstance(exc, A5OperatorError):
                    raise
                raise A5OperatorError(
                    f"Cannot load replay inputs for {page!r}: "
                    f"{exc}"
                ) from exc

            if ys is None or xs is None:
                raise A5OperatorError(
                    f"A5 replay page {page!r} does not contain "
                    "the frozen y and x coordinate arrays"
                )

            try:
                result = self.run(
                    channels=channels,
                    a1=a1,
                    ys=ys,
                    xs=xs,
                )
            except Exception as exc:
                if isinstance(exc, A5OperatorError):
                    raise
                raise A5OperatorError(
                    f"A5 replay inference failed for "
                    f"{page!r}: {exc}"
                ) from exc

            try:
                expected_probability = np.load(
                    expected_probability_path,
                    allow_pickle=False,
                )

                expected_provisional = np.load(
                    expected_provisional_path,
                    allow_pickle=True,
                )

                expected_final = np.load(
                    expected_final_path,
                    allow_pickle=True,
                )

                expected_a5 = cv2.imread(
                    str(expected_a5_path),
                    cv2.IMREAD_GRAYSCALE,
                )

            except Exception as exc:
                raise A5OperatorError(
                    f"Cannot load expected replay output for "
                    f"{page!r}: {exc}"
                ) from exc

            if expected_a5 is None:
                raise A5OperatorError(
                    "Cannot read expected replay page: "
                    f"{expected_a5_path}"
                )

            # Normalise decision arrays to the same fixed-width
            # Unicode representation used by make_decisions().
            expected_provisional = np.asarray(
                expected_provisional,
                dtype="<U10",
            )
            expected_final = np.asarray(
                expected_final,
                dtype="<U10",
            )

            actual_probability = np.asarray(
                result.probability,
                dtype=np.float64,
            )
            expected_probability = np.asarray(
                expected_probability,
                dtype=np.float64,
            )

            check = {
                "page": page,
                "eligible_pixels": int(
                    len(result.eligible_y)
                ),
                "probability_exact": bool(
                    np.array_equal(
                        actual_probability,
                        expected_probability,
                    )
                ),
                "provisional_exact": bool(
                    np.array_equal(
                        result.provisional_decision,
                        expected_provisional,
                    )
                ),
                "final_exact": bool(
                    np.array_equal(
                        result.final_decision,
                        expected_final,
                    )
                ),
                "page_exact": bool(
                    np.array_equal(
                        result.image,
                        expected_a5,
                    )
                ),
                "print_pixels": int(
                    result.print_mask.sum()
                ),
            }

            failed_checks = [
                name
                for name in (
                    "probability_exact",
                    "provisional_exact",
                    "final_exact",
                    "page_exact",
                )
                if not check[name]
            ]

            if failed_checks:
                details: list[str] = []

                if not check["probability_exact"]:
                    if (
                        actual_probability.shape
                        == expected_probability.shape
                        and actual_probability.size
                    ):
                        maximum_difference = float(
                            np.max(
                                np.abs(
                                    actual_probability
                                    - expected_probability
                                )
                            )
                        )
                    else:
                        maximum_difference = float("nan")

                    details.append(
                        "probability shapes "
                        f"{actual_probability.shape} versus "
                        f"{expected_probability.shape}; "
                        f"maximum absolute difference "
                        f"{maximum_difference}"
                    )

                if not check["provisional_exact"]:
                    details.append(
                        "provisional decision mismatches="
                        f"{_decision_mismatch_count(
                            result.provisional_decision, 
                            expected_provisional
                        )}"
                    )

                if not check["final_exact"]:
                    details.append(
                        "final decision mismatches="
                        f"{_decision_mismatch_count(
                            result.final_decision, expected_final
                        )}"
                    )

                if not check["page_exact"]:
                    if result.image.shape == expected_a5.shape:
                        page_differences = int(
                            np.count_nonzero(
                                result.image != expected_a5
                            )
                        )
                    else:
                        page_differences = -1

                    details.append(
                        "page shapes "
                        f"{result.image.shape} versus "
                        f"{expected_a5.shape}; "
                        f"different pixels={page_differences}"
                    )

                raise A5OperatorError(
                    f"A5 startup replay failed for {page!r}: "
                    f"{failed_checks}. "
                    + "; ".join(details)
                )

            checks.append(check)

        return {
            "policy_id": policy_id,
            "passed": True,
            "page_count": len(checks),
            "replay_directory": str(replay_dir),
            "checks": checks,
        }


def _load_npz_inputs(
    path: Path,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    if not path.is_file():
        raise A5OperatorError(
            f"Input NPZ not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        files = set(archive.files)

        required = (
            set(DEFAULT_CHANNELS)
            | {"a1"}
        )

        missing = required - files

        if missing:
            raise A5OperatorError(
                "Input NPZ schema mismatch. "
                f"Path: {path}; "
                f"missing: {sorted(missing)}; "
                f"available: {sorted(files)}"
            )

        channels = {
            name: archive[name]
            for name in DEFAULT_CHANNELS
        }

        a1 = archive["a1"]

        ys = (
            archive["y"]
            if "y" in files
            else None
        )

        xs = (
            archive["x"]
            if "x" in files
            else None
        )

    return channels, a1, ys, xs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen A5-v1 operator")
    parser.add_argument("--root", default=os.environ.get("A5_ROOT", "."), help="Repository root containing frozen/a5_v1")
    parser.add_argument("--input", required=True, type=Path, help="NPZ containing the 11 channels and a1; optional y/x")
    parser.add_argument("--output", required=True, type=Path, help="Output corrected PNG")
    parser.add_argument("--audit", type=Path, help="Optional output NPZ for probabilities, decisions and PRINT mask")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    channels, a1, ys, xs = _load_npz_inputs(args.input)
    operator = A5Operator(
        args.root,
        device=args.device,
        batch_size=args.batch_size,
        run_startup_replay=True,
    )
    result = operator.run(channels, a1, ys, xs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result.image):
        raise A5OperatorError(f"Could not write {args.output}")
    if args.audit is not None:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.audit,
            probability=result.probability,
            provisional_decision=result.provisional_decision,
            final_decision=result.final_decision,
            y=result.eligible_y,
            x=result.eligible_x,
            print_mask=result.print_mask.astype(np.uint8),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
