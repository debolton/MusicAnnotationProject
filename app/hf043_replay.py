from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ARDROSS = (
    ROOT
    / "frozen"
    / "a5_v1"
    / "replay"
    / "ardross"
)

CHANNELS = [
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

EXPECTED_POLICY_ID = "A5-v1-print-p0980-margin0960-run2"

VALID_CLASSES = {
    "PRINT",
    "ANNOTATION",
    "UNCERTAIN",
}

EXPECTED_REPLAY_ROWS = 5000


class ReplayFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def _read_vector(
    path: Path,
    kind: str = "float",
) -> np.ndarray:
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.reader(f))

    values = []

    for row in rows:
        if not row:
            continue

        value = row[-1].strip()

        try:
            if kind == "float":
                values.append(float(value))
            elif kind == "int":
                values.append(int(float(value)))
            else:
                raise ValueError(
                    f"Unknown vector kind: {kind}"
                )
        except ValueError:
            # Allows a single ordinary CSV header row.
            continue

    dtype = (
        np.float64
        if kind == "float"
        else np.int64
    )

    return np.asarray(
        values,
        dtype=dtype,
    )


def _read_string_vector(
    path: Path,
) -> np.ndarray:
    values = []

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        for row in csv.reader(f):
            if not row:
                continue

            value = row[-1].strip().upper()

            if value in VALID_CLASSES:
                values.append(value)

    return np.asarray(
        values,
        dtype=str,
    )


def _read_matrix(
    path: Path,
    dtype=np.float32,
) -> np.ndarray:
    try:
        array = np.loadtxt(
            path,
            delimiter=",",
            dtype=dtype,
        )
    except Exception as exc:
        raise ReplayFailure(
            f"Cannot read replay matrix {path}: {exc}"
        ) from exc

    if array.ndim != 2:
        raise ReplayFailure(
            f"{path.name}: expected 2-D matrix, "
            f"got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ReplayFailure(
            f"{path.name}: non-finite values"
        )

    return array


def _read_image(
    name: str,
    flags: int = cv2.IMREAD_GRAYSCALE,
) -> np.ndarray:
    path = ARDROSS / name

    array = cv2.imread(
        str(path),
        flags,
    )

    if array is None:
        raise ReplayFailure(
            f"Unreadable replay image: {path}"
        )

    return array


def load_replay():
    required = [
        "obs.png",
        "norm.png",
        "mask.png",
        "depth.csv",
        "horizontal.png",
        "vertical.png",
        "compact.png",
        "preserve.png",
        "remove.png",
        "ambiguous.png",
        "edge.png",
        "x.csv",
        "y.csv",
        "expected_probability.csv",
        "expected_provisional.csv",
        "expected_final.csv",
    ]

    missing = [
        name
        for name in required
        if not (ARDROSS / name).exists()
    ]

    if missing:
        raise ReplayFailure(
            "Ardross replay incomplete: "
            + ", ".join(missing)
        )

    arrays = {
        name: _read_image(
            f"{name}.png"
        )
        for name in [
            "obs",
            "norm",
            "mask",
            "horizontal",
            "vertical",
            "compact",
            "preserve",
            "remove",
            "ambiguous",
            "edge",
        ]
    }

    height, width = arrays["obs"].shape

    for name, array in arrays.items():
        if array.shape != (
            height,
            width,
        ):
            raise ReplayFailure(
                f"Ardross {name} shape "
                f"{array.shape} != "
                f"{(height, width)}"
            )

    depth = _read_matrix(
        ARDROSS / "depth.csv",
        np.float32,
    )

    if depth.shape != (
        height,
        width,
    ):
        raise ReplayFailure(
            "Ardross depth shape mismatch: "
            f"depth={depth.shape}, "
            f"obs={(height, width)}"
        )

    arrays["depth"] = depth

    # --------------------------------------------------------------
    # Frozen tensor-input contract
    #
    # obs and norm remain byte-scale here.
    # inference_runtime.make_padded() owns /255 exactly once.
    # --------------------------------------------------------------

    arrays["obs"] = arrays[
        "obs"
    ].astype(
        np.uint8,
        copy=False,
    )

    arrays["norm"] = arrays[
        "norm"
    ].astype(
        np.uint8,
        copy=False,
    )

    for name in [
        "mask",
        "horizontal",
        "vertical",
        "compact",
        "preserve",
        "remove",
        "ambiguous",
        "edge",
    ]:
        arrays[name] = (
            arrays[name]
            > 0
        )

    # --------------------------------------------------------------
    # Structural consistency of the frozen channels.
    # --------------------------------------------------------------

    mask = arrays["mask"]
    preserve = arrays["preserve"]
    remove = arrays["remove"]
    ambiguous = arrays["ambiguous"]

    if np.any(
        preserve
        & remove
    ):
        raise ReplayFailure(
            "Ardross PRESERVE and REMOVE routes overlap"
        )

    if np.any(
        preserve
        & ambiguous
    ):
        raise ReplayFailure(
            "Ardross PRESERVE and AMBIGUOUS routes overlap"
        )

    if np.any(
        remove
        & ambiguous
    ):
        raise ReplayFailure(
            "Ardross REMOVE and AMBIGUOUS routes overlap"
        )

    routed = (
        preserve
        | remove
        | ambiguous
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

        raise ReplayFailure(
            "Ardross routes do not exactly partition "
            "the authoritative mask: "
            f"missing={missing_route}, "
            f"outside={escaped_route}"
        )

    x = _read_vector(
        ARDROSS / "x.csv",
        "int",
    ).astype(
        np.int64,
        copy=False,
    )

    y = _read_vector(
        ARDROSS / "y.csv",
        "int",
    ).astype(
        np.int64,
        copy=False,
    )

    expected_probability = _read_vector(
        ARDROSS / "expected_probability.csv",
        "float",
    ).astype(
        np.float64,
        copy=False,
    )

    expected_provisional = (
        _read_string_vector(
            ARDROSS
            / "expected_provisional.csv"
        )
    )

    expected_final = (
        _read_string_vector(
            ARDROSS
            / "expected_final.csv"
        )
    )

    lengths = {
        "x": len(x),
        "y": len(y),
        "expected_probability": len(
            expected_probability
        ),
        "expected_provisional": len(
            expected_provisional
        ),
        "expected_final": len(
            expected_final
        ),
    }

    if len(
        set(
            lengths.values()
        )
    ) != 1:
        raise ReplayFailure(
            "Ardross replay vectors have inconsistent "
            f"row counts: {lengths}"
        )

    if len(x) != EXPECTED_REPLAY_ROWS:
        raise ReplayFailure(
            "Ardross replay must contain "
            f"{EXPECTED_REPLAY_ROWS} coordinates, "
            f"found {len(x)}"
        )

    if (
        (x < 0).any()
        or (x >= width).any()
        or (y < 0).any()
        or (y >= height).any()
    ):
        raise ReplayFailure(
            "Ardross replay coordinate out of bounds"
        )

    if len(
        set(
            zip(
                y.tolist(),
                x.tolist(),
            )
        )
    ) != len(x):
        raise ReplayFailure(
            "Ardross replay contains duplicate "
            "page coordinates"
        )

    # Every frozen replay coordinate must satisfy the registered
    # A5-v1 candidate rule:
    #
    #   authoritative mask
    #   AND observed < 200
    #   AND not already PRESERVE-routed.
    coordinate_mask = mask[
        y,
        x,
    ]

    coordinate_dark = (
        arrays["obs"][
            y,
            x,
        ]
        < 200
    )

    coordinate_preserve = preserve[
        y,
        x,
    ]

    if not coordinate_mask.all():
        raise ReplayFailure(
            "Ardross replay contains coordinates "
            "outside the authoritative mask"
        )

    if not coordinate_dark.all():
        raise ReplayFailure(
            "Ardross replay contains coordinates "
            "with observed value >= 200"
        )

    if coordinate_preserve.any():
        raise ReplayFailure(
            "Ardross replay contains coordinates "
            "already routed PRESERVE"
        )

    if not np.isfinite(
        expected_probability
    ).all():
        raise ReplayFailure(
            "Expected Ardross probability contains "
            "non-finite values"
        )

    if (
        (expected_probability < 0).any()
        or (expected_probability > 1).any()
    ):
        raise ReplayFailure(
            "Expected Ardross probability lies outside 0..1"
        )

    if not set(
        expected_provisional.tolist()
    ).issubset(
        VALID_CLASSES
    ):
        raise ReplayFailure(
            "Expected provisional replay contains "
            "an invalid class"
        )

    if not set(
        expected_final.tolist()
    ).issubset(
        VALID_CLASSES
    ):
        raise ReplayFailure(
            "Expected final replay contains "
            "an invalid class"
        )

    return (
        arrays,
        y,
        x,
        expected_probability,
        expected_provisional,
        expected_final,
    )


def run_ardross_replay(
    operator,
    probability_atol: float = 1e-7,
):
    # --------------------------------------------------------------
    # Operator/bundle identity
    # --------------------------------------------------------------

    if getattr(
        operator,
        "a5_policy_id",
        None,
    ) != EXPECTED_POLICY_ID:
        raise ReplayFailure(
            "Unexpected operator policy ID: "
            f"{getattr(operator, 'a5_policy_id', None)!r}"
        )

    policy = getattr(
        operator,
        "policy",
        None,
    )

    if not isinstance(
        policy,
        dict,
    ):
        raise ReplayFailure(
            "Operator has no frozen policy dictionary"
        )

    if policy.get(
        "policy_id"
    ) != EXPECTED_POLICY_ID:
        raise ReplayFailure(
            "Frozen policy ID mismatch: "
            f"{policy.get('policy_id')!r}"
        )

    schema = getattr(
        operator,
        "schema",
        None,
    )

    if not isinstance(
        schema,
        dict,
    ):
        raise ReplayFailure(
            "Operator has no frozen channel schema"
        )

    if schema.get(
        "channels"
    ) != CHANNELS:
        raise ReplayFailure(
            "Ardross replay channel order differs "
            "from the frozen A5-v1 schema"
        )

    patch = schema.get(
        "patch",
        schema.get(
            "patch_size",
        ),
    )

    if int(
        patch
        if patch is not None
        else -1
    ) != 33:
        raise ReplayFailure(
            "Ardross replay requires frozen "
            f"33x33 patches; found {patch!r}"
        )

    if probability_atol < 0:
        raise ValueError(
            "probability_atol must be non-negative"
        )

    # --------------------------------------------------------------
    # Load immutable replay evidence
    # --------------------------------------------------------------

    (
        arrays,
        y,
        x,
        expected_probability,
        expected_provisional,
        expected_final,
    ) = load_replay()

    observed = np.asarray(
        arrays["obs"]
    )

    normalised = np.asarray(
        arrays["norm"]
    )

    # These channels must still be byte scale.
    # make_padded() performs the frozen /255 conversion.
    if observed.max() <= 1:
        raise ReplayFailure(
            "Ardross obs appears pre-normalised; "
            "make_padded() must own /255 scaling."
        )

    if normalised.max() <= 1:
        raise ReplayFailure(
            "Ardross norm appears pre-normalised; "
            "make_padded() must own /255 scaling."
        )

    # --------------------------------------------------------------
    # Frozen tensor construction
    # --------------------------------------------------------------

    padded = operator.rt.make_padded(
        arrays,
        operator.schema,
    )

    missing_padded = [
        name
        for name in CHANNELS
        if name not in padded
    ]

    if missing_padded:
        raise ReplayFailure(
            "Frozen tensor builder omitted channels: "
            f"{missing_padded}"
        )

    # --------------------------------------------------------------
    # Frozen model + isotonic calibration
    # --------------------------------------------------------------

    got_probability = np.asarray(
        operator.rt.score_coordinates(
            operator.model,
            operator.calibrator,
            padded,
            operator.schema,
            y,
            x,
            operator.device,
            batch=512,
        ),
        dtype=np.float64,
    ).reshape(-1)

    if got_probability.shape != (
        EXPECTED_REPLAY_ROWS,
    ):
        raise ReplayFailure(
            "Ardross probability output shape mismatch: "
            f"{got_probability.shape}"
        )

    if not np.isfinite(
        got_probability
    ).all():
        raise ReplayFailure(
            "Ardross scored probability contains "
            "non-finite values"
        )

    if (
        (got_probability < 0).any()
        or (got_probability > 1).any()
    ):
        raise ReplayFailure(
            "Ardross scored probability lies outside 0..1"
        )

    # --------------------------------------------------------------
    # Frozen three-class rule + page-global run filtering
    # --------------------------------------------------------------

    decision_result = operator.rt.decisions(
        got_probability,
        operator.policy,
        observed.shape,
        y,
        x,
    )

    if (
        not isinstance(
            decision_result,
            tuple,
        )
        or len(
            decision_result
        ) != 2
    ):
        raise ReplayFailure(
            "A5-v1 decisions() must return "
            "(provisional, final)"
        )

    got_provisional = np.asarray(
        decision_result[0],
        dtype=str,
    ).reshape(-1)

    got_final = np.asarray(
        decision_result[1],
        dtype=str,
    ).reshape(-1)

    if not (
        len(got_probability)
        == len(got_provisional)
        == len(got_final)
        == len(y)
        == len(x)
        == EXPECTED_REPLAY_ROWS
    ):
        raise ReplayFailure(
            "Ardross replay output length mismatch: "
            f"probability={len(got_probability)}, "
            f"provisional={len(got_provisional)}, "
            f"final={len(got_final)}, "
            f"y={len(y)}, "
            f"x={len(x)}"
        )

    unexpected_provisional = (
        set(
            got_provisional.tolist()
        )
        - VALID_CLASSES
    )

    unexpected_final = (
        set(
            got_final.tolist()
        )
        - VALID_CLASSES
    )

    if unexpected_provisional:
        raise ReplayFailure(
            "Unexpected provisional A5-v1 classes: "
            f"{sorted(unexpected_provisional)}"
        )

    if unexpected_final:
        raise ReplayFailure(
            "Unexpected final A5-v1 classes: "
            f"{sorted(unexpected_final)}"
        )

    # Connected-run filtering may demote provisional PRINT
    # to UNCERTAIN. It must never create new PRINT decisions.
    if np.any(
        (got_final == "PRINT")
        & (got_provisional != "PRINT")
    ):
        raise ReplayFailure(
            "A5-v1 run filtering created a new PRINT decision"
        )

    # ANNOTATION is not subject to PRINT connected-run filtering.
    if np.any(
        (got_provisional == "ANNOTATION")
        & (got_final != "ANNOTATION")
    ):
        raise ReplayFailure(
            "A5-v1 run filtering altered an "
            "ANNOTATION decision"
        )

    # Existing UNCERTAIN decisions cannot become PRINT.
    if np.any(
        (got_provisional == "UNCERTAIN")
        & (got_final == "PRINT")
    ):
        raise ReplayFailure(
            "A5-v1 run filtering promoted UNCERTAIN to PRINT"
        )

    # --------------------------------------------------------------
    # Exact frozen replay comparisons
    # --------------------------------------------------------------

    probability_exact = bool(
        np.array_equal(
            got_probability,
            expected_probability,
        )
    )

    probability_within_atol = bool(
        np.allclose(
            got_probability,
            expected_probability,
            rtol=0.0,
            atol=probability_atol,
            equal_nan=False,
        )
    )

    probability_max_abs_error = float(
        np.max(
            np.abs(
                got_probability
                - expected_probability
            )
        )
    )

    provisional_exact = bool(
        np.array_equal(
            got_provisional,
            expected_provisional,
        )
    )

    provisional_mismatches = int(
        np.sum(
            got_provisional
            != expected_provisional
        )
    )

    final_exact = bool(
        np.array_equal(
            got_final,
            expected_final,
        )
    )

    final_mismatches = int(
        np.sum(
            got_final
            != expected_final
        )
    )

    provisional_print = int(
        np.sum(
            got_provisional
            == "PRINT"
        )
    )

    final_print = int(
        np.sum(
            got_final
            == "PRINT"
        )
    )

    run_demotions = int(
        np.sum(
            (got_provisional == "PRINT")
            & (got_final == "UNCERTAIN")
        )
    )

    annotation_count = int(
        np.sum(
            got_final
            == "ANNOTATION"
        )
    )

    uncertain_count = int(
        np.sum(
            got_final
            == "UNCERTAIN"
        )
    )

    result = {
        "name": "ardross_a5_startup_replay",
        "policy_id": operator.a5_policy_id,
        "rows": int(
            len(got_probability)
        ),

        "probability_exact": (
            probability_exact
        ),
        "probability_atol": float(
            probability_atol
        ),
        "probability_within_atol": (
            probability_within_atol
        ),
        "probability_max_abs_error": (
            probability_max_abs_error
        ),

        "provisional_exact": (
            provisional_exact
        ),
        "provisional_mismatches": (
            provisional_mismatches
        ),

        "final_exact": (
            final_exact
        ),
        "final_mismatches": (
            final_mismatches
        ),

        "provisional_print": (
            provisional_print
        ),
        "final_print": (
            final_print
        ),
        "run_filter_demotions": (
            run_demotions
        ),
        "final_annotation": (
            annotation_count
        ),
        "final_uncertain": (
            uncertain_count
        ),

        # Explicitly state what this replay proves.
        "tensor_replay_tested": True,
        "model_replay_tested": True,
        "calibration_replay_tested": True,
        "decision_replay_tested": True,

        # These require a separate whole-page replay fixture
        # containing observed + geometry_t80 + expected A5 bytes.
        "whole_page_assembly_tested": False,
        "outside_mask_drift": None,
        "non_print_baseline_regression": None,
        "print_not_equal_observed": None,
        "changed_byte_not_equal_observed": None,
        "unexpected_changed_byte": None,
    }

    result["passed"] = bool(
        result["probability_within_atol"]
        and result["provisional_exact"]
        and result["final_exact"]
    )

    if not result["passed"]:
        raise ReplayFailure(
            "Mandatory Ardross A5 replay failed: "
            + json.dumps(
                result,
                sort_keys=True,
            )
        )

    return result