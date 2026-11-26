"""
Frozen operational A5 safe-cleanup runtime.

EXPECTED BUNDLE LAYOUT
----------------------
a5_safe_cleanup/
    inference_runtime.py
    model.pt
    isotonic_calibrator.joblib
    policy.json
    channel_schema.json
    model_config.json

EXPECTED SEMANTICS
------------------
Input population:
    AMBIGUOUS pixels only.

Model positive class:
    CLEAN.

Operational action:
    CLEAN     -> copy paper estimate.
    KEEP      -> retain exact A1.
    UNCERTAIN -> retain exact A1.

This runtime deliberately rejects an observed-PRINT bundle. Do not place this
file inside frozen/a5_v1 or edit the Notebook 91 frozen bundle in place.
"""

from __future__ import annotations

import hashlib
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

ALLOWED_DECISIONS = {
    "CLEAN",
    "KEEP",
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


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_file(
    bundle: Path,
    filename: str,
) -> Path:
    path = bundle / filename

    if not path.is_file():
        raise RuntimeError(
            f"Missing safe-cleanup bundle file: {path}"
        )

    return path


def load_json(
    path: Path,
    name: str,
) -> dict:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read {name}: {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            f"{name} must contain a JSON object"
        )

    return value


def require_uint8_image(
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

    if expected_shape is not None and value.shape != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: "
            f"expected {expected_shape}, found {value.shape}"
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

    if value.size:
        if value.min() < 0 or value.max() > 255:
            raise RuntimeError(
                f"{name} contains values outside 0..255"
            )

    return np.ascontiguousarray(
        value.astype(np.uint8)
    )


def require_bool_mask(
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
            f"{name} shape mismatch: "
            f"expected {expected_shape}, found {value.shape}"
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


def verify_declared_hash(
    policy: dict,
    key: str,
    path: Path,
) -> None:
    expected = policy.get(key)

    if expected in {
        None,
        "",
    }:
        return

    actual = sha256_file(path)

    if actual.lower() != str(expected).lower():
        raise RuntimeError(
            f"Bundle hash mismatch for {path.name}: "
            f"{actual} != {expected}"
        )


def assert_policy_contract(
    policy: dict,
) -> None:
    policy_id = policy.get("policy_id")

    if not policy_id:
        raise RuntimeError(
            "Safe-cleanup policy has no policy_id"
        )

    probability_semantics = str(
        policy.get(
            "probability_semantics",
            policy.get("positive_class", ""),
        )
    ).upper()

    if probability_semantics != "CLEAN":
        raise RuntimeError(
            "This runtime requires a genuine CLEAN model. "
            f"Bundle declares {probability_semantics!r}. "
            "Do not load the Notebook 91 observed-PRINT bundle."
        )

    positive_class = str(
        policy.get(
            "positive_class",
            "CLEAN",
        )
    ).upper()

    if positive_class != "CLEAN":
        raise RuntimeError(
            "Safe-cleanup positive_class must be CLEAN; "
            f"found {positive_class!r}"
        )

    eligible_route = str(
        policy.get(
            "eligible_route",
            "",
        )
    ).upper()

    if eligible_route != "AMBIGUOUS":
        raise RuntimeError(
            "Safe-cleanup eligible_route must be AMBIGUOUS; "
            f"found {eligible_route!r}"
        )

    clean_action = str(
        policy.get(
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
            "Safe-cleanup clean_action must copy the paper "
            f"estimate; found {clean_action!r}"
        )

    default_action = str(
        policy.get(
            "default_action",
            policy.get(
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
            "Safe-cleanup default action must retain A1; "
            f"found {default_action!r}"
        )

    clean_threshold = policy.get(
        "clean_threshold"
    )

    if clean_threshold is None:
        raise RuntimeError(
            "Safe-cleanup policy has no clean_threshold"
        )

    clean_threshold = float(clean_threshold)

    if not 0.0 <= clean_threshold <= 1.0:
        raise RuntimeError(
            "clean_threshold must be within 0..1"
        )

    keep_threshold = policy.get(
        "keep_threshold"
    )

    if keep_threshold is not None:
        keep_threshold = float(keep_threshold)

        if not 0.0 <= keep_threshold <= 1.0:
            raise RuntimeError(
                "keep_threshold must be within 0..1"
            )

        if keep_threshold >= clean_threshold:
            raise RuntimeError(
                "keep_threshold must be lower than "
                "clean_threshold"
            )

    minimum_clean_run = int(
        policy.get(
            "minimum_clean_run",
            1,
        )
    )

    if minimum_clean_run < 1:
        raise RuntimeError(
            "minimum_clean_run must be at least 1"
        )

    connectivity = int(
        policy.get(
            "connectivity",
            8,
        )
    )

    if connectivity not in {
        4,
        8,
    }:
        raise RuntimeError(
            "connectivity must be 4 or 8"
        )

    truth_codes = policy.get(
        "truth_codes",
        {},
    )

    if truth_codes:
        if not isinstance(truth_codes, dict):
            raise RuntimeError(
                "truth_codes must be a dictionary"
            )

        classes = {
            str(value).upper()
            for value in truth_codes.values()
        }

        unexpected = (
            classes
            - ALLOWED_DECISIONS
        )

        if unexpected:
            raise RuntimeError(
                "Unexpected safe-cleanup classes: "
                f"{sorted(unexpected)}"
            )

        if "CLEAN" not in classes:
            raise RuntimeError(
                "truth_codes contains no CLEAN class"
            )


def assert_schema_contract(
    schema: dict,
    config: dict,
) -> None:
    channels = schema.get("channels")

    if channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Safe-cleanup channel order mismatch: "
            f"{channels}"
        )

    patch = schema.get(
        "patch",
        schema.get("patch_size"),
    )

    if int(patch if patch is not None else -1) != (
        EXPECTED_PATCH_SIZE
    ):
        raise RuntimeError(
            "Safe-cleanup patch size mismatch: "
            f"{patch}"
        )

    padding = str(
        schema.get(
            "padding",
            "reflect",
        )
    ).lower()

    if padding != "reflect":
        raise RuntimeError(
            "Safe-cleanup padding must be reflect; "
            f"found {padding!r}"
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


def extract_state_dict(
    loaded: Any,
) -> dict:
    if isinstance(loaded, dict):
        if "state_dict" in loaded:
            state = loaded["state_dict"]
        elif "model_state_dict" in loaded:
            state = loaded["model_state_dict"]
        else:
            state = loaded
    else:
        raise RuntimeError(
            "model.pt did not contain a state dictionary"
        )

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
            f"Safe-cleanup bundle does not exist: {bundle}"
        )

    policy_path = require_file(
        bundle,
        "policy.json",
    )
    schema_path = require_file(
        bundle,
        "channel_schema.json",
    )
    config_path = require_file(
        bundle,
        "model_config.json",
    )
    model_path = require_file(
        bundle,
        "model.pt",
    )
    calibrator_path = require_file(
        bundle,
        "isotonic_calibrator.joblib",
    )

    policy = load_json(
        policy_path,
        "policy",
    )
    schema = load_json(
        schema_path,
        "channel schema",
    )
    config = load_json(
        config_path,
        "model config",
    )

    assert_policy_contract(policy)
    assert_schema_contract(
        schema,
        config,
    )

    verify_declared_hash(
        policy,
        "model_sha256",
        model_path,
    )
    verify_declared_hash(
        policy,
        "calibrator_sha256",
        calibrator_path,
    )
    verify_declared_hash(
        policy,
        "schema_sha256",
        schema_path,
    )
    verify_declared_hash(
        policy,
        "model_config_sha256",
        config_path,
    )

    selected_device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if (
        selected_device.startswith("cuda")
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
            f"Unable to load safe-cleanup model: {exc}"
        ) from exc

    state = extract_state_dict(checkpoint)

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Safe-cleanup checkpoint does not match "
            f"TopologyCNN exactly: {exc}"
        ) from exc

    model.eval()

    try:
        calibrator = joblib.load(
            calibrator_path
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load safe-cleanup calibrator: {exc}"
        ) from exc

    if not callable(
        getattr(
            calibrator,
            "predict",
            None,
        )
    ):
        raise RuntimeError(
            "Safe-cleanup calibrator has no predict() method"
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
            "channels must be a dictionary"
        )

    expected_channels = schema.get(
        "channels"
    )

    if expected_channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Runtime channel schema is not the expected "
            "ordered contract"
        )

    patch = int(
        schema.get(
            "patch",
            schema.get("patch_size", -1),
        )
    )

    if patch != EXPECTED_PATCH_SIZE:
        raise RuntimeError(
            f"Unexpected patch size: {patch}"
        )

    if patch % 2 != 1:
        raise RuntimeError(
            "Patch size must be odd"
        )

    half = patch // 2
    output: dict[str, np.ndarray] = {}
    shapes = set()

    for name in expected_channels:
        if name not in channels:
            raise RuntimeError(
                f"Missing channel: {name}"
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
                value.min() < 0
                or value.max() > 255
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
            f"Channel shape mismatch: {sorted(shapes)}"
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

    patch = int(
        schema.get(
            "patch",
            schema.get("patch_size", -1),
        )
    )

    channels = schema["channels"]

    if len(ys) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    first_shape = padded[channels[0]].shape
    half = patch // 2
    image_height = first_shape[0] - 2 * half
    image_width = first_shape[1] - 2 * half

    if (
        np.any(ys < 0)
        or np.any(xs < 0)
        or np.any(ys >= image_height)
        or np.any(xs >= image_width)
    ):
        raise RuntimeError(
            "Scoring coordinate lies outside the image"
        )

    raw_parts: list[np.ndarray] = []

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

            for local_index, (
                y,
                x,
            ) in enumerate(
                zip(
                    ys[start:stop],
                    xs[start:stop],
                )
            ):
                for channel_index, name in enumerate(
                    channels
                ):
                    candidate = padded[name][
                        int(y):int(y) + patch,
                        int(x):int(x) + patch,
                    ]

                    if candidate.shape != (
                        patch,
                        patch,
                    ):
                        raise RuntimeError(
                            f"Patch extraction failed for {name} "
                            f"at ({int(y)}, {int(x)}): "
                            f"{candidate.shape}"
                        )

                    tensors[
                        local_index,
                        channel_index,
                    ] = candidate

            tensor_batch = torch.from_numpy(
                tensors
            ).to(device)

            logits = model(tensor_batch)

            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(
                    "Safe-cleanup model must produce two logits "
                    f"per candidate; found {tuple(logits.shape)}"
                )

            raw_clean = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            raw_parts.append(raw_clean)

    raw_probability = np.concatenate(
        raw_parts
    )

    if raw_probability.shape != (
        len(ys),
    ):
        raise RuntimeError(
            "Raw CLEAN probability length mismatch"
        )

    if not np.isfinite(
        raw_probability
    ).all():
        raise RuntimeError(
            "Raw CLEAN probability contains non-finite values"
        )

    try:
        calibrated = calibrator.predict(
            raw_probability
        )
    except Exception as exc:
        raise RuntimeError(
            f"CLEAN calibration failed: {exc}"
        ) from exc

    calibrated = np.asarray(
        calibrated,
        dtype=np.float64,
    )

    if calibrated.shape != (
        len(ys),
    ):
        raise RuntimeError(
            "Calibrated CLEAN probability length mismatch"
        )

    if not np.isfinite(
        calibrated
    ).all():
        raise RuntimeError(
            "Calibrated CLEAN probability contains "
            "non-finite values"
        )

    tolerance = 1e-12

    if (
        np.any(calibrated < -tolerance)
        or np.any(calibrated > 1.0 + tolerance)
    ):
        raise RuntimeError(
            "Calibrated CLEAN probability lies outside 0..1"
        )

    return np.clip(
        calibrated,
        0.0,
        1.0,
    )


def map_truth_codes(
    values: np.ndarray,
    policy: dict,
) -> np.ndarray:
    values = np.asarray(values)

    truth_codes = policy.get(
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

        unknown = (
            set(integer_values.tolist())
            - set(mapping)
        )

        if unknown:
            raise RuntimeError(
                "Decision output contains unknown numeric "
                f"codes: {sorted(unknown)}"
            )

        result = np.asarray(
            [
                mapping[int(value)]
                for value in integer_values
            ],
            dtype=object,
        )
    else:
        result = np.char.upper(
            values.astype(str)
        ).astype(object)

    unexpected = (
        set(result.tolist())
        - ALLOWED_DECISIONS
    )

    if unexpected:
        raise RuntimeError(
            "Decision output contains unexpected classes: "
            f"{sorted(unexpected)}"
        )

    return result


def decisions(
    probability: np.ndarray,
    policy: dict,
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
):
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
            "Probability must be one-dimensional"
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
            "Page shape must have two dimensions"
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
            "Decision coordinate lies outside the page"
        )

    if len(set(zip(
        ys.tolist(),
        xs.tolist(),
    ))) != len(ys):
        raise RuntimeError(
            "Duplicate decision coordinates supplied"
        )

    if not np.isfinite(
        probability
    ).all():
        raise RuntimeError(
            "CLEAN probability contains non-finite values"
        )

    if (
        np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise RuntimeError(
            "CLEAN probability lies outside 0..1"
        )

    clean_threshold = float(
        policy["clean_threshold"]
    )

    keep_threshold_value = policy.get(
        "keep_threshold"
    )

    provisional = np.full(
        len(probability),
        "UNCERTAIN",
        dtype=object,
    )

    if keep_threshold_value is not None:
        keep_threshold = float(
            keep_threshold_value
        )

        provisional[
            probability <= keep_threshold
        ] = "KEEP"

    provisional[
        probability >= clean_threshold
    ] = "CLEAN"

    provisional = map_truth_codes(
        provisional,
        policy,
    )

    final = provisional.copy()

    minimum_clean_run = int(
        policy.get(
            "minimum_clean_run",
            1,
        )
    )

    if minimum_clean_run > 1:
        clean_indices = np.where(
            provisional == "CLEAN"
        )[0]

        clean_map = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        clean_map[
            ys[clean_indices],
            xs[clean_indices],
        ] = 1

        connectivity = int(
            policy.get(
                "connectivity",
                8,
            )
        )

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                clean_map,
                connectivity=connectivity,
            )
        )

        for component_id in range(
            1,
            count,
        ):
            component_area = int(
                stats[
                    component_id,
                    cv2.CC_STAT_AREA,
                ]
            )

            if component_area >= minimum_clean_run:
                continue

            selected_labels = labels[
                ys[clean_indices],
                xs[clean_indices],
            ]

            demote = (
                selected_labels
                == component_id
            )

            final[
                clean_indices[demote]
            ] = "UNCERTAIN"

    final = map_truth_codes(
        final,
        policy,
    )

    return provisional, final


def decision_mask(
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
    final_decisions: np.ndarray,
    positive_class: str = "CLEAN",
) -> np.ndarray:
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
            "Coordinate/decision length mismatch"
        )

    selected = (
        np.char.upper(
            final_decisions.astype(str)
        )
        == positive_class.upper()
    )

    output = np.zeros(
        shape,
        dtype=bool,
    )

    output[
        ys[selected],
        xs[selected],
    ] = True

    return output


def assemble_operational_a5(
    observed: np.ndarray,
    a1: np.ndarray,
    paper: np.ndarray,
    authoritative_mask: np.ndarray,
    ambiguous: np.ndarray,
    eligible: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    final_decisions: np.ndarray,
):
    observed = require_uint8_image(
        observed,
        "observed",
    )
    shape = observed.shape

    a1 = require_uint8_image(
        a1,
        "a1",
        shape,
    )
    paper = require_uint8_image(
        paper,
        "paper",
        shape,
    )

    authority = require_bool_mask(
        authoritative_mask,
        "authoritative_mask",
        shape,
    )
    ambiguous = require_bool_mask(
        ambiguous,
        "ambiguous",
        shape,
    )
    eligible = require_bool_mask(
        eligible,
        "eligible",
        shape,
    )

    ys = np.asarray(
        ys,
        dtype=np.int64,
    )
    xs = np.asarray(
        xs,
        dtype=np.int64,
    )

    if ys.shape != xs.shape:
        raise RuntimeError(
            "Y/X coordinate length mismatch"
        )

    if np.any(
        ambiguous
        & ~authority
    ):
        raise RuntimeError(
            "AMBIGUOUS route escapes the authoritative mask"
        )

    if np.any(
        eligible
        & ~ambiguous
    ):
        raise RuntimeError(
            "A5 eligibility includes non-AMBIGUOUS pixels"
        )

    if np.any(
        eligible
        & ~authority
    ):
        raise RuntimeError(
            "A5 eligibility escapes the authoritative mask"
        )

    coordinate_mask = np.zeros(
        shape,
        dtype=bool,
    )
    coordinate_mask[
        ys,
        xs,
    ] = True

    if not np.array_equal(
        coordinate_mask,
        eligible,
    ):
        missing_coordinates = int(
            np.sum(
                eligible
                & ~coordinate_mask
            )
        )

        unexpected_coordinates = int(
            np.sum(
                coordinate_mask
                & ~eligible
            )
        )

        raise RuntimeError(
            "A5 coordinates do not exactly represent the "
            "eligible mask: "
            f"missing={missing_coordinates}, "
            f"unexpected={unexpected_coordinates}"
        )

    clean_mask = decision_mask(
        shape=shape,
        ys=ys,
        xs=xs,
        final_decisions=final_decisions,
        positive_class="CLEAN",
    )

    if np.any(
        clean_mask
        & ~eligible
    ):
        raise RuntimeError(
            "CLEAN decision escapes A5 eligibility"
        )

    if np.any(
        clean_mask
        & ~ambiguous
    ):
        raise RuntimeError(
            "CLEAN decision includes non-AMBIGUOUS pixels"
        )

    if np.any(
        clean_mask
        & ~authority
    ):
        raise RuntimeError(
            "CLEAN decision escapes the authoritative mask"
        )

    output = a1.copy()
    output[clean_mask] = paper[clean_mask]

    if np.any(
        output[clean_mask]
        != paper[clean_mask]
    ):
        raise RuntimeError(
            "A5 CLEAN output does not equal the paper estimate"
        )

    if np.any(
        output[~clean_mask]
        != a1[~clean_mask]
    ):
        raise RuntimeError(
            "A5 non-CLEAN output differs from A1"
        )

    if np.any(
        output[~ambiguous]
        != a1[~ambiguous]
    ):
        raise RuntimeError(
            "A5 changed a non-AMBIGUOUS pixel"
        )

    if np.any(
        output[~authority]
        != observed[~authority]
    ):
        raise RuntimeError(
            "A5 changed a pixel outside the authoritative mask"
        )

    changed = output != a1

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
            "A5 page delta does not equal effective "
            "CLEAN-to-paper changes: "
            f"unexpected={unexpected}, missing={missing}"
        )

    return output, clean_mask


def assemble(
    observed: np.ndarray,
    a1: np.ndarray,
    paper: np.ndarray,
    authoritative_mask: np.ndarray,
    ambiguous: np.ndarray,
    eligible: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    final_decisions: np.ndarray,
):
    """
    Compatibility entry point for an operator expecting runtime.assemble().

    This function intentionally uses CLEAN-to-paper semantics. It is not
    compatible with the historical observed-PRINT call signature.
    """

    return assemble_operational_a5(
        observed=observed,
        a1=a1,
        paper=paper,
        authoritative_mask=authoritative_mask,
        ambiguous=ambiguous,
        eligible=eligible,
        ys=ys,
        xs=xs,
        final_decisions=final_decisions,
    )