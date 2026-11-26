from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
from torch import nn


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
    "policy_id": "A5-v1-print-p0980-margin0960-run2",
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


class TopologyCNN(nn.Module):
    def __init__(self, nc: int = 11):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                nc,
                24,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2),

            nn.Conv2d(
                24,
                48,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),

            nn.Conv2d(
                48,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Linear(
            64,
            2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(
            self.net(x).flatten(1)
        )


def _require_file(
    bundle: Path,
    filename: str,
) -> Path:
    path = bundle / filename

    if not path.is_file():
        raise RuntimeError(
            f"Missing frozen A5-v1 bundle file: {path}"
        )

    return path


def _load_json(
    path: Path,
    name: str,
) -> dict:
    try:
        value = json.loads(
            path.read_text()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read {name} from {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            f"{name} must contain a JSON object"
        )

    return value


def _assert_float_equal(
    name: str,
    actual: Any,
    expected: float,
) -> None:
    try:
        actual_float = float(actual)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Frozen policy field {name!r} is not numeric: "
            f"{actual!r}"
        ) from exc

    if actual_float != float(expected):
        raise RuntimeError(
            f"Frozen policy mismatch for {name}: "
            f"{actual_float!r} != {expected!r}"
        )


def _assert_policy_contract(
    policy: dict,
) -> None:
    missing = sorted(
        set(EXPECTED_POLICY)
        - set(policy)
    )

    if missing:
        raise RuntimeError(
            "Frozen A5-v1 policy is missing fields: "
            f"{missing}"
        )

    for key, expected in EXPECTED_POLICY.items():
        actual = policy[key]

        if isinstance(expected, float):
            _assert_float_equal(
                key,
                actual,
                expected,
            )
            continue

        if actual != expected:
            raise RuntimeError(
                f"Frozen policy mismatch for {key}: "
                f"{actual!r} != {expected!r}"
            )

    print_threshold = float(
        policy["print_threshold"]
    )
    print_margin = float(
        policy["print_margin"]
    )
    effective_threshold = max(
        print_threshold,
        (1.0 + print_margin) / 2.0,
    )

    if effective_threshold != float(
        policy["effective_print_threshold"]
    ):
        raise RuntimeError(
            "Frozen effective PRINT threshold is inconsistent: "
            f"calculated {effective_threshold!r}, "
            f"recorded "
            f"{policy['effective_print_threshold']!r}"
        )

    if not 0.0 <= float(
        policy["annotation_threshold"]
    ) <= 1.0:
        raise RuntimeError(
            "annotation_threshold must be within 0..1"
        )

    if not 0.0 <= print_threshold <= 1.0:
        raise RuntimeError(
            "print_threshold must be within 0..1"
        )

    if not 0.0 <= print_margin <= 1.0:
        raise RuntimeError(
            "print_margin must be within 0..1"
        )

    if int(
        policy["minimum_print_run"]
    ) < 1:
        raise RuntimeError(
            "minimum_print_run must be at least 1"
        )

    if int(
        policy["connectivity"]
    ) not in {
        4,
        8,
    }:
        raise RuntimeError(
            "connectivity must be 4 or 8"
        )


def _assert_schema_contract(
    schema: dict,
    config: dict,
) -> None:
    channels = schema.get("channels")

    if channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "A5-v1 channel order differs from the frozen "
            "11-channel contract: "
            f"{channels!r}"
        )

    patch = schema.get(
        "patch",
        schema.get("patch_size"),
    )

    if patch is None:
        raise RuntimeError(
            "A5-v1 channel schema has no patch size"
        )

    if int(patch) != EXPECTED_PATCH_SIZE:
        raise RuntimeError(
            "A5-v1 patch size differs from the frozen "
            f"{EXPECTED_PATCH_SIZE}x{EXPECTED_PATCH_SIZE} "
            f"contract: {patch!r}"
        )

    if int(patch) % 2 != 1:
        raise RuntimeError(
            "A5-v1 patch size must be odd"
        )

    padding = schema.get(
        "padding"
    )

    if (
        padding is not None
        and str(padding).lower() != "reflect"
    ):
        raise RuntimeError(
            "A5-v1 padding differs from the frozen reflect "
            f"contract: {padding!r}"
        )

    configured_channels = config.get(
        "channels",
        config.get(
            "input_channels",
            len(channels),
        ),
    )

    if int(configured_channels) != len(channels):
        raise RuntimeError(
            "model_config channel count disagrees with "
            "channel_schema"
        )

    configured_patch = config.get(
        "patch",
        config.get(
            "patch_size",
            patch,
        ),
    )

    if int(configured_patch) != int(patch):
        raise RuntimeError(
            "model_config patch size disagrees with "
            "channel_schema"
        )


def _extract_state_dict(
    checkpoint: Any,
) -> dict:
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "model.pt does not contain a state dictionary"
        )

    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError(
            "Extracted model state is not a dictionary"
        )

    if state and all(
        str(key).startswith("module.")
        for key in state
    ):
        state = {
            str(key)[7:]: value
            for key, value in state.items()
        }

    return state


def load_bundle(
    bundle_dir: Path | str,
    device: str | None = None,
):
    bundle = Path(bundle_dir)

    if not bundle.is_dir():
        raise RuntimeError(
            f"Frozen A5-v1 bundle does not exist: {bundle}"
        )

    policy_path = _require_file(
        bundle,
        "policy.json",
    )
    schema_path = _require_file(
        bundle,
        "channel_schema.json",
    )
    config_path = _require_file(
        bundle,
        "model_config.json",
    )
    model_path = _require_file(
        bundle,
        "model.pt",
    )
    calibrator_path = _require_file(
        bundle,
        "isotonic_calibrator.joblib",
    )

    policy = _load_json(
        policy_path,
        "frozen policy",
    )
    schema = _load_json(
        schema_path,
        "channel schema",
    )
    config = _load_json(
        config_path,
        "model configuration",
    )

    _assert_policy_contract(policy)
    _assert_schema_contract(
        schema,
        config,
    )

    selected_device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if (
        str(selected_device).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    model = TopologyCNN(
        nc=len(schema["channels"])
    ).to(selected_device)

    try:
        checkpoint = torch.load(
            model_path,
            map_location=selected_device,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load frozen A5-v1 model: {exc}"
        ) from exc

    state = _extract_state_dict(
        checkpoint
    )

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Frozen A5-v1 checkpoint does not match "
            f"TopologyCNN exactly: {exc}"
        ) from exc

    model.eval()

    try:
        calibrator = joblib.load(
            calibrator_path
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load frozen A5-v1 calibrator: {exc}"
        ) from exc

    if not callable(
        getattr(
            calibrator,
            "predict",
            None,
        )
    ):
        raise RuntimeError(
            "Frozen A5-v1 calibrator has no predict() method"
        )

    return (
        model,
        calibrator,
        policy,
        schema,
        selected_device,
    )


def make_padded(
    channels: dict,
    schema: dict,
) -> dict:
    if not isinstance(channels, dict):
        raise RuntimeError(
            "A5-v1 channels must be a dictionary"
        )

    channel_order = schema.get(
        "channels"
    )

    if channel_order != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Runtime channel order differs from the frozen "
            "A5-v1 schema"
        )

    patch = int(
        schema.get(
            "patch",
            schema.get("patch_size", -1),
        )
    )

    if patch != EXPECTED_PATCH_SIZE:
        raise RuntimeError(
            f"Unexpected A5-v1 patch size: {patch}"
        )

    half = patch // 2
    output = {}
    shapes = set()

    for name in channel_order:
        if name not in channels:
            raise RuntimeError(
                f"Missing A5-v1 channel: {name}"
            )

        array = np.asarray(
            channels[name]
        )

        if array.ndim != 2:
            raise RuntimeError(
                f"{name}: expected a two-dimensional array; "
                f"found {array.shape}"
            )

        if (
            np.issubdtype(array.dtype, np.number)
            and not np.isfinite(array).all()
        ):
            raise RuntimeError(
                f"{name}: non-finite values"
            )

        shapes.add(array.shape)

        value = array.astype(
            np.float32,
            copy=True,
        )

        if name in {
            "obs",
            "norm",
        }:
            if value.size and (
                value.min() < 0.0
                or value.max() > 255.0
            ):
                raise RuntimeError(
                    f"{name}: expected values within 0..255 "
                    "before frozen /255 normalisation"
                )

            value /= 255.0

        output[name] = np.pad(
            value,
            half,
            mode="reflect",
        )

    if len(shapes) != 1:
        raise RuntimeError(
            f"A5-v1 channel shape mismatch: {sorted(shapes)}"
        )

    return output


def score_coordinates(
    model: nn.Module,
    calibrator: Any,
    padded: dict,
    schema: dict,
    ys: np.ndarray,
    xs: np.ndarray,
    device: str,
    batch: int = 512,
) -> np.ndarray:
    if not isinstance(batch, int) or batch < 1:
        raise ValueError(
            "batch must be a positive integer"
        )

    ys = np.asarray(
        ys,
        dtype=np.int64,
    )
    xs = np.asarray(
        xs,
        dtype=np.int64,
    )

    if ys.ndim != 1 or xs.ndim != 1:
        raise RuntimeError(
            "ys and xs must be one-dimensional"
        )

    if ys.shape != xs.shape:
        raise RuntimeError(
            "Y/X coordinate length mismatch"
        )

    if len(ys) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    channels = schema["channels"]
    patch = int(
        schema.get(
            "patch",
            schema.get("patch_size", -1),
        )
    )
    half = patch // 2

    missing = [
        name
        for name in channels
        if name not in padded
    ]

    if missing:
        raise RuntimeError(
            f"Padded channel set is incomplete: {missing}"
        )

    padded_shapes = {
        np.asarray(padded[name]).shape
        for name in channels
    }

    if len(padded_shapes) != 1:
        raise RuntimeError(
            "Padded A5-v1 channel shapes differ: "
            f"{sorted(padded_shapes)}"
        )

    padded_height, padded_width = next(
        iter(padded_shapes)
    )

    image_height = (
        padded_height
        - 2 * half
    )
    image_width = (
        padded_width
        - 2 * half
    )

    if (
        np.any(ys < 0)
        or np.any(xs < 0)
        or np.any(ys >= image_height)
        or np.any(xs >= image_width)
    ):
        raise RuntimeError(
            "A5-v1 scoring coordinate lies outside the image"
        )

    raw_parts = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(ys),
            batch,
        ):
            stop = min(
                len(ys),
                start + batch,
            )

            tensors = np.empty(
                (
                    stop - start,
                    len(channels),
                    patch,
                    patch,
                ),
                dtype=np.float32,
            )

            for row_index, (
                y,
                x,
            ) in enumerate(
                zip(
                    ys[start:stop],
                    xs[start:stop],
                )
            ):
                y = int(y)
                x = int(x)

                for channel_index, name in enumerate(
                    channels
                ):
                    candidate = padded[name][
                        y:y + patch,
                        x:x + patch,
                    ]

                    if candidate.shape != (
                        patch,
                        patch,
                    ):
                        raise RuntimeError(
                            f"Patch extraction failed for {name} "
                            f"at ({y}, {x}): {candidate.shape}"
                        )

                    tensors[
                        row_index,
                        channel_index,
                    ] = candidate

            tensor_batch = torch.from_numpy(
                tensors
            ).to(device)

            logits = model(
                tensor_batch
            )

            if (
                logits.ndim != 2
                or logits.shape[1] != 2
            ):
                raise RuntimeError(
                    "A5-v1 model must produce two logits per "
                    f"candidate; found {tuple(logits.shape)}"
                )

            raw_print = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            raw_parts.append(
                raw_print
            )

    raw_probability = np.concatenate(
        raw_parts
    )

    if raw_probability.shape != (
        len(ys),
    ):
        raise RuntimeError(
            "Raw PRINT probability length mismatch"
        )

    if not np.isfinite(
        raw_probability
    ).all():
        raise RuntimeError(
            "Raw PRINT probability contains non-finite values"
        )

    try:
        calibrated = calibrator.predict(
            raw_probability
        )
    except Exception as exc:
        raise RuntimeError(
            f"A5-v1 PRINT calibration failed: {exc}"
        ) from exc

    calibrated = np.asarray(
        calibrated,
        dtype=np.float64,
    )

    if calibrated.shape != (
        len(ys),
    ):
        raise RuntimeError(
            "Calibrated PRINT probability length mismatch"
        )

    if not np.isfinite(
        calibrated
    ).all():
        raise RuntimeError(
            "Calibrated PRINT probability contains "
            "non-finite values"
        )

    tolerance = 1e-12

    if (
        np.any(calibrated < -tolerance)
        or np.any(calibrated > 1.0 + tolerance)
    ):
        raise RuntimeError(
            "Calibrated PRINT probability lies outside 0..1"
        )

    return np.clip(
        calibrated,
        0.0,
        1.0,
    )


def decisions(
    probability: np.ndarray,
    policy: dict,
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
):
    _assert_policy_contract(
        policy
    )

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )
    ys = np.asarray(
        ys,
        dtype=np.int64,
    )
    xs = np.asarray(
        xs,
        dtype=np.int64,
    )

    if probability.ndim != 1:
        raise RuntimeError(
            "PRINT probability must be one-dimensional"
        )

    if not (
        probability.shape
        == ys.shape
        == xs.shape
    ):
        raise RuntimeError(
            "Probability/Y/X coordinate length mismatch"
        )

    if len(shape) != 2:
        raise RuntimeError(
            "Page shape must contain two dimensions"
        )

    height = int(shape[0])
    width = int(shape[1])

    if height < 1 or width < 1:
        raise RuntimeError(
            f"Invalid page shape: {shape}"
        )

    if (
        np.any(ys < 0)
        or np.any(xs < 0)
        or np.any(ys >= height)
        or np.any(xs >= width)
    ):
        raise RuntimeError(
            "A5-v1 decision coordinate lies outside the page"
        )

    if len(
        set(
            zip(
                ys.tolist(),
                xs.tolist(),
            )
        )
    ) != len(ys):
        raise RuntimeError(
            "Duplicate A5-v1 decision coordinates supplied"
        )

    if not np.isfinite(
        probability
    ).all():
        raise RuntimeError(
            "PRINT probability contains non-finite values"
        )

    if (
        np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise RuntimeError(
            "PRINT probability lies outside 0..1"
        )

    p_print = probability
    p_annotation = (
        1.0
        - p_print
    )

    provisional = np.full(
        len(p_print),
        "UNCERTAIN",
        dtype=object,
    )

    annotation = (
        p_annotation
        >= float(
            policy["annotation_threshold"]
        )
    )

    print_selected = (
        (
            p_print
            >= float(
                policy["print_threshold"]
            )
        )
        & (
            (
                p_print
                - p_annotation
            )
            >= float(
                policy["print_margin"]
            )
        )
    )

    provisional[
        annotation
    ] = "ANNOTATION"

    provisional[
        print_selected
    ] = "PRINT"

    if not set(
        provisional.tolist()
    ).issubset(
        ALLOWED_DECISIONS
    ):
        raise RuntimeError(
            "Unexpected provisional A5-v1 decision"
        )

    final = provisional.copy()

    minimum_run = int(
        policy["minimum_print_run"]
    )

    if minimum_run > 1:
        provisional_print_indices = np.where(
            provisional == "PRINT"
        )[0]

        print_map = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        print_map[
            ys[provisional_print_indices],
            xs[provisional_print_indices],
        ] = 1

        component_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                print_map,
                connectivity=int(
                    policy["connectivity"]
                ),
            )
        )

        if len(provisional_print_indices):
            coordinate_labels = labels[
                ys[provisional_print_indices],
                xs[provisional_print_indices],
            ]

            for component_id in range(
                1,
                component_count,
            ):
                component_area = int(
                    stats[
                       component_id,
                        cv2.CC_STAT_AREA,
                    ]
                )

                if component_area >= minimum_run:
                    continue

                demote = (
                    coordinate_labels
                    == component_id
                )

                final[
                    provisional_print_indices[
                        demote
                    ]
                ] = "UNCERTAIN"

    if not set(
        final.tolist()
    ).issubset(
        ALLOWED_DECISIONS
    ):
        raise RuntimeError(
            "Unexpected final A5-v1 decision"
        )

    if np.any(
        (final == "PRINT")
        & (provisional != "PRINT")
    ):
        raise RuntimeError(
            "Run filtering created a new PRINT decision"
        )

    if np.any(
        (provisional == "ANNOTATION")
        & (final != "ANNOTATION")
    ):
        raise RuntimeError(
            "Run filtering altered an ANNOTATION decision"
        )

    return provisional, final


def assemble(
    observed: np.ndarray,
    a1: np.ndarray,
    authoritative_mask: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    final_decisions: np.ndarray,
):
    observed = np.asarray(
        observed
    )
    a1 = np.asarray(
        a1
    )
    authoritative_mask = np.asarray(
        authoritative_mask
    )

    if observed.ndim != 2:
        raise RuntimeError(
            f"Observed page must be 2-D; found {observed.shape}"
        )

    if a1.ndim != 2:
        raise RuntimeError(
            f"A1 page must be 2-D; found {a1.shape}"
        )

    if authoritative_mask.ndim != 2:
        raise RuntimeError(
            "Authoritative mask must be 2-D; "
            f"found {authoritative_mask.shape}"
        )

    if not (
        observed.shape
        == a1.shape
        == authoritative_mask.shape
    ):
        raise RuntimeError(
            "Observed/A1/authoritative-mask shape mismatch"
        )

    if not np.issubdtype(
        observed.dtype,
        np.number,
    ):
        raise RuntimeError(
            "Observed page must contain numeric values"
        )

    if not np.issubdtype(
        a1.dtype,
        np.number,
    ):
        raise RuntimeError(
            "A1 page must contain numeric values"
        )

    if (
        not np.isfinite(observed).all()
        or not np.isfinite(a1).all()
    ):
        raise RuntimeError(
            "Observed or A1 page contains non-finite values"
        )

    if observed.size and (
        observed.min() < 0
        or observed.max() > 255
    ):
        raise RuntimeError(
            "Observed page contains values outside 0..255"
        )

    if a1.size and (
        a1.min() < 0
        or a1.max() > 255
    ):
        raise RuntimeError(
            "A1 page contains values outside 0..255"
        )

    observed = np.ascontiguousarray(
        observed.astype(np.uint8)
    )
    a1 = np.ascontiguousarray(
        a1.astype(np.uint8)
    )
    mask = np.ascontiguousarray(
        authoritative_mask.astype(bool)
    )

    ys = np.asarray(
        ys,
        dtype=np.int64,
    )
    xs = np.asarray(
        xs,
        dtype=np.int64,
    )
    final_decisions = np.asarray(
        final_decisions,
        dtype=object,
    )

    if not (
        ys.shape
        == xs.shape
        == final_decisions.shape
    ):
        raise RuntimeError(
            "Coordinate/final-decision length mismatch"
        )

    if ys.ndim != 1:
        raise RuntimeError(
            "Coordinates and decisions must be one-dimensional"
        )

    height, width = observed.shape

    if (
        np.any(ys < 0)
        or np.any(xs < 0)
        or np.any(ys >= height)
        or np.any(xs >= width)
    ):
        raise RuntimeError(
            "A5-v1 assembly coordinate lies outside the page"
        )

    if len(
        set(
            zip(
                ys.tolist(),
                xs.tolist(),
            )
        )
    ) != len(ys):
        raise RuntimeError(
            "Duplicate A5-v1 assembly coordinates supplied"
        )

    final_upper = np.char.upper(
        final_decisions.astype(str)
    ).astype(object)

    unexpected = (
        set(final_upper.tolist())
        - ALLOWED_DECISIONS
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected final A5-v1 decisions: "
            f"{sorted(unexpected)}"
        )

    if len(ys) and np.any(
        ~mask[ys, xs]
    ):
        raise RuntimeError(
            "A5-v1 candidate coordinate escapes the "
            "authoritative mask"
        )

    print_indices = np.where(
        final_upper == "PRINT"
    )[0]

    print_y = ys[
        print_indices
    ]
    print_x = xs[
        print_indices
    ]

    print_mask = np.zeros(
        observed.shape,
        dtype=bool,
    )

    print_mask[
        print_y,
        print_x,
    ] = True

    output = a1.copy()

    # Frozen A5-v1 action:
    # PRINT -> copy the exact observed byte.
    output[
        print_mask
    ] = observed[
        print_mask
    ]

    if np.any(
        print_mask
        & ~mask
    ):
        raise RuntimeError(
            "A5-v1 PRINT decision escapes the "
            "authoritative mask"
        )

    if np.any(
        output[~mask]
        != observed[~mask]
    ):
        raise RuntimeError(
            "A5-v1 outside-authoritative-mask drift"
        )

    if np.any(
        output[~print_mask]
        != a1[~print_mask]
    ):
        raise RuntimeError(
            "A5-v1 non-PRINT A1 regression"
        )

    if np.any(
        output[print_mask]
        != observed[print_mask]
    ):
        raise RuntimeError(
            "A5-v1 PRINT output does not equal the exact "
            "observed byte"
        )

    changed = (
        output
        != a1
    )

    expected_changed = (
        print_mask
        & (
            observed
            != a1
        )
    )

    if not np.array_equal(
        changed,
        expected_changed,
    ):
        unexpected_changes = int(
            np.sum(
                changed
                & ~expected_changed
            )
        )
        missing_changes = int(
            np.sum(
                expected_changed
                & ~changed
            )
        )

        raise RuntimeError(
            "A5-v1 page delta does not equal effective "
            "PRINT restoration: "
            f"unexpected={unexpected_changes}, "
            f"missing={missing_changes}"
        )

    if np.any(
        changed
        & (
            output
            != observed
        )
    ):
        raise RuntimeError(
            "A5-v1 changed byte does not equal observed"
        )

    return output, print_mask