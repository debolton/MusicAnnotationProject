from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
from typing import Any

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

EXPECTED_POLICY_ID = (
    "A5-v1-print-p0980-margin0960-run2"
)

EXPECTED_PATCH_SIZE = 33
EXPECTED_REPLAY_ROWS = 5000

VALID_CLASSES = {
    "PRINT",
    "ANNOTATION",
    "UNCERTAIN",
}


class ReplayFailure(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def sha256_file(
    path: Path,
) -> str:
    if not path.is_file():
        raise ReplayFailure(
            f"Cannot hash missing replay file: {path}"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1048576
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def array_sha256(
    array: np.ndarray,
) -> str:
    value = np.ascontiguousarray(
        np.asarray(
            array
        )
    )

    digest = hashlib.sha256()

    digest.update(
        str(
            value.dtype
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            value.shape
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        value.tobytes()
    )

    return digest.hexdigest()


def _require_files(
    names: list[str],
) -> None:
    missing = [
        name
        for name in names
        if not (
            ARDROSS
            / name
        ).is_file()
    ]

    if missing:
        raise ReplayFailure(
            "Ardross replay incomplete: "
            + ", ".join(
                missing
            )
        )


def _read_vector(
    path: Path,
    kind: str = "float",
) -> np.ndarray:
    if kind not in {
        "float",
        "int",
    }:
        raise ValueError(
            f"Unknown vector kind: {kind}"
        )

    if not path.is_file():
        raise ReplayFailure(
            f"Missing replay vector: {path}"
        )

    values = []

    try:
        with path.open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            rows = csv.reader(
                handle
            )

            for row in rows:
                if not row:
                    continue

                value = row[-1].strip()

                if not value:
                    continue

                try:
                    if kind == "float":
                        values.append(
                            float(
                                value
                            )
                        )

                    else:
                        numeric = float(
                            value
                        )

                        if not np.isfinite(
                            numeric
                        ):
                            raise ValueError(
                                "non-finite integer value"
                            )

                        integer = int(
                            numeric
                        )

                        if float(
                            integer
                        ) != numeric:
                            raise ValueError(
                                "non-integral coordinate value"
                            )

                        values.append(
                            integer
                        )

                except ValueError:
                    # Permit an ordinary CSV header row.
                    # A malformed data row will normally make the
                    # final frozen row-count check fail closed.
                    continue

    except Exception as exc:
        raise ReplayFailure(
            f"Cannot read replay vector {path}: {exc}"
        ) from exc

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
    if not path.is_file():
        raise ReplayFailure(
            f"Missing replay class vector: {path}"
        )

    values = []

    try:
        with path.open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            for row in csv.reader(
                handle
            ):
                if not row:
                    continue

                value = (
                    row[-1]
                    .strip()
                    .upper()
                )

                if value in VALID_CLASSES:
                    values.append(
                        value
                    )

    except Exception as exc:
        raise ReplayFailure(
            "Cannot read replay class vector "
            f"{path}: {exc}"
        ) from exc

    return np.asarray(
        values,
        dtype=str,
    )


def _read_matrix(
    path: Path,
    dtype: Any = np.float32,
) -> np.ndarray:
    if not path.is_file():
        raise ReplayFailure(
            f"Missing replay matrix: {path}"
        )

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

    if not np.isfinite(
        array
    ).all():
        raise ReplayFailure(
            f"{path.name}: non-finite values"
        )

    return array


def _read_image(
    name: str,
    flags: int = cv2.IMREAD_GRAYSCALE,
) -> np.ndarray:
    path = (
        ARDROSS
        / name
    )

    if not path.is_file():
        raise ReplayFailure(
            f"Missing replay image: {path}"
        )

    array = cv2.imread(
        str(
            path
        ),
        flags,
    )

    if array is None:
        raise ReplayFailure(
            f"Unreadable replay image: {path}"
        )

    return array


# ---------------------------------------------------------------------------
# Frozen replay fixture
# ---------------------------------------------------------------------------

def load_replay():
    """
    Load the frozen 5,000-coordinate Ardross inference replay.

    This fixture proves:

        - frozen 11-channel tensor construction;
        - frozen 33x33 patch preprocessing;
        - frozen model inference;
        - frozen isotonic calibration;
        - provisional class decisions;
        - connected-run filtering over the supplied
          5,000-coordinate replay population.

    This fixture does not currently contain:

        - a frozen Notebook 87 A1 page;
        - a frozen expected A5 output page;
        - a frozen expected A5 PRINT mask;
        - every eligible coordinate on the page.

    It therefore does not prove whole-page A1-to-A5 assembly.
    """

    required = [
        "README.txt",
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

    _require_files(
        required
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

    observed = arrays[
        "obs"
    ]

    if observed.ndim != 2:
        raise ReplayFailure(
            "Ardross observed image must be 2-D"
        )

    height, width = (
        observed.shape
    )

    if (
        height < 1
        or width < 1
    ):
        raise ReplayFailure(
            "Ardross observed image is empty"
        )

    for name, array in arrays.items():
        if array.ndim != 2:
            raise ReplayFailure(
                f"Ardross {name} must be 2-D; "
                f"found {array.shape}"
            )

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
        ARDROSS
        / "depth.csv",
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

    if np.any(
        depth < 0.0
    ):
        raise ReplayFailure(
            "Ardross depth contains negative values"
        )

    # Construct the dictionary in the exact frozen channel order.
    arrays = {
        "obs": image_arrays[
            "obs"
        ],

        "norm": image_arrays[
            "norm"
        ],

        "mask": image_arrays[
            "mask"
        ],

        "depth": np.ascontiguousarray(
            depth.astype(
                np.float32,
                copy=False,
            )
        ),

        "horizontal": image_arrays[
            "horizontal"
        ],

        "vertical": image_arrays[
            "vertical"
        ],

        "compact": image_arrays[
            "compact"
        ],

        "preserve": image_arrays[
            "preserve"
        ],

        "remove": image_arrays[
            "remove"
        ],

        "ambiguous": image_arrays[
            "ambiguous"
        ],

        "edge": image_arrays[
            "edge"
        ],
    }

    if list(
        arrays
    ) != CHANNELS:
        raise ReplayFailure(
            "Loaded Ardross channel order differs "
            "from the frozen A5-v1 order: "
            f"{list(arrays)!r}"
        )

    # ------------------------------------------------------------
    # Frozen tensor-input contract
    #
    # obs and norm remain byte scale here.
    # inference_runtime.make_padded() owns /255 exactly once.
    # ------------------------------------------------------------

    arrays[
        "obs"
    ] = np.ascontiguousarray(
        arrays[
            "obs"
        ].astype(
            np.uint8,
            copy=False,
        )
    )

    arrays[
        "norm"
    ] = np.ascontiguousarray(
        arrays[
            "norm"
        ].astype(
            np.uint8,
            copy=False,
        )
    )

    binary_channels = [
        "mask",
        "horizontal",
        "vertical",
        "compact",
        "preserve",
        "remove",
        "ambiguous",
        "edge",
    ]

    for name in binary_channels:
        arrays[
            name
        ] = np.ascontiguousarray(
            arrays[
                name
            ]
            > 0
        )

    if list(
        arrays.keys()
    ) != CHANNELS:
        raise ReplayFailure(
            "Loaded Ardross channel order differs "
            "from the frozen A5-v1 order: "
            f"{list(arrays.keys())!r}"
        )

    # ------------------------------------------------------------
    # Structural consistency of the frozen channels
    # ------------------------------------------------------------

    mask = arrays[
        "mask"
    ]

    preserve = arrays[
        "preserve"
    ]

    remove = arrays[
        "remove"
    ]

    ambiguous = arrays[
        "ambiguous"
    ]

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
            "Ardross PRESERVE and AMBIGUOUS "
            "routes overlap"
        )

    if np.any(
        remove
        & ambiguous
    ):
        raise ReplayFailure(
            "Ardross REMOVE and AMBIGUOUS "
            "routes overlap"
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

    for name in binary_channels:
        if arrays[
            name
        ].dtype != bool:
            raise ReplayFailure(
                f"Ardross {name} channel is not boolean"
            )

    # ------------------------------------------------------------
    # Frozen coordinate and expected-output vectors
    # ------------------------------------------------------------

    x = _read_vector(
        ARDROSS
        / "x.csv",
        "int",
    ).astype(
        np.int64,
        copy=False,
    )

    y = _read_vector(
        ARDROSS
        / "y.csv",
        "int",
    ).astype(
        np.int64,
        copy=False,
    )

    expected_probability = _read_vector(
        ARDROSS
        / "expected_probability.csv",
        "float",
    ).astype(
        np.float64,
        copy=False,
    )

    expected_provisional = _read_string_vector(
        ARDROSS
        / "expected_provisional.csv"
    )

    expected_final = _read_string_vector(
        ARDROSS
        / "expected_final.csv"
    )

    lengths = {
        "x": len(
            x
        ),
        "y": len(
            y
        ),
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

    if len(
        x
    ) != EXPECTED_REPLAY_ROWS:
        raise ReplayFailure(
            "Ardross replay must contain "
            f"{EXPECTED_REPLAY_ROWS} coordinates, "
            f"found {len(x)}"
        )

    if (
        np.any(
            x < 0
        )
        or np.any(
            x >= width
        )
        or np.any(
            y < 0
        )
        or np.any(
            y >= height
        )
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
    ) != len(
        x
    ):
        raise ReplayFailure(
            "Ardross replay contains duplicate "
            "page coordinates"
        )

    # ------------------------------------------------------------
    # Registered A5-v1 candidate rule
    #
    # authoritative mask
    # AND observed < 200
    # AND not already PRESERVE-routed
    # ------------------------------------------------------------

    coordinate_mask = mask[
        y,
        x,
    ]

    coordinate_dark = (
        arrays[
            "obs"
        ][
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

    # ------------------------------------------------------------
    # Expected-output integrity
    # ------------------------------------------------------------

    if not np.isfinite(
        expected_probability
    ).all():
        raise ReplayFailure(
            "Expected Ardross probability contains "
            "non-finite values"
        )

    if (
        np.any(
            expected_probability < 0.0
        )
        or np.any(
            expected_probability > 1.0
        )
    ):
        raise ReplayFailure(
            "Expected Ardross probability lies "
            "outside 0..1"
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

    # Final PRINT may only originate from provisional PRINT.

    if np.any(
        (
            expected_final
            == "PRINT"
        )
        & (
            expected_provisional
            != "PRINT"
        )
    ):
        raise ReplayFailure(
            "Expected replay run filtering creates "
            "a new PRINT decision"
        )

    # ANNOTATION is not affected by the PRINT run filter.

    if np.any(
        (
            expected_provisional
            == "ANNOTATION"
        )
        & (
            expected_final
            != "ANNOTATION"
        )
    ):
        raise ReplayFailure(
            "Expected replay run filtering alters "
            "an ANNOTATION decision"
        )

    expected_changed_decision = (
        expected_provisional
        != expected_final
    )

    expected_permitted_change = (
        (
            expected_provisional
            == "PRINT"
        )
        & (
            expected_final
            == "UNCERTAIN"
        )
    )

    if np.any(
        expected_changed_decision
        & ~expected_permitted_change
    ):
        raise ReplayFailure(
            "Expected replay contains a decision "
            "transition other than PRINT -> UNCERTAIN"
        )

    return (
        arrays,
        y,
        x,
        expected_probability,
        expected_provisional,
        expected_final,
    )


# ---------------------------------------------------------------------------
# Frozen inference replay
# ---------------------------------------------------------------------------

def run_ardross_replay(
    operator: Any,
    probability_atol: float = 1e-7,
) -> dict:
    """
    Run the frozen Ardross 5,000-coordinate A5 replay.

    This replay is intentionally explicit about its scope.

    It tests:
        - channel/tensor construction;
        - model inference;
        - isotonic calibration;
        - provisional decisions;
        - connected-run final decisions.

    It does not claim to test whole-page A1-to-A5 assembly because
    the current frozen fixture has no Notebook 87 A1 page or expected
    A5 page and is not the complete eligible page population.
    """

    if not isinstance(
        probability_atol,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        raise ValueError(
            "probability_atol must be numeric"
        )

    probability_atol = float(
        probability_atol
    )

    if (
        not np.isfinite(
            probability_atol
        )
        or probability_atol < 0.0
    ):
        raise ValueError(
            "probability_atol must be a finite "
            "non-negative number"
        )

    # ------------------------------------------------------------
    # Operator and frozen-bundle identity
    # ------------------------------------------------------------

    if getattr(
        operator,
        "a5_policy_id",
        None,
    ) != EXPECTED_POLICY_ID:
        raise ReplayFailure(
            "Unexpected operator policy ID: "
            f"{getattr(operator, 'a5_policy_id', None)!r}"
        )

    runtime = getattr(
        operator,
        "rt",
        None,
    )

    if runtime is None:
        raise ReplayFailure(
            "Operator has no inference runtime"
        )

    for function_name in [
        "make_padded",
        "score_coordinates",
        "decisions",
        "assemble",
    ]:
        if not callable(
            getattr(
                runtime,
                function_name,
                None,
            )
        ):
            raise ReplayFailure(
                "Operator inference runtime has no "
                f"callable {function_name}()"
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

    try:
        patch = int(
            patch
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ReplayFailure(
            "Frozen schema patch size is invalid: "
            f"{patch!r}"
        ) from exc

    if patch != EXPECTED_PATCH_SIZE:
        raise ReplayFailure(
            "Ardross replay requires frozen "
            f"{EXPECTED_PATCH_SIZE}x"
            f"{EXPECTED_PATCH_SIZE} patches; "
            f"found {patch!r}"
        )

    padding = schema.get(
        "padding"
    )

    if (
        padding is not None
        and str(
            padding
        ).lower()
        != "reflect"
    ):
        raise ReplayFailure(
            "Ardross replay requires reflect padding; "
            f"found {padding!r}"
        )

    model = getattr(
        operator,
        "model",
        None,
    )

    calibrator = getattr(
        operator,
        "calibrator",
        None,
    )

    device = getattr(
        operator,
        "device",
        None,
    )

    if model is None:
        raise ReplayFailure(
            "Operator has no frozen model"
        )

    if calibrator is None:
        raise ReplayFailure(
            "Operator has no frozen calibrator"
        )

    if not callable(
        getattr(
            calibrator,
            "predict",
            None,
        )
    ):
        raise ReplayFailure(
            "Operator calibrator has no predict() method"
        )

    if device is None:
        raise ReplayFailure(
            "Operator has no inference device"
        )

    # ------------------------------------------------------------
    # Load immutable replay evidence
    # ------------------------------------------------------------

    (
        arrays,
        y,
        x,
        expected_probability,
        expected_provisional,
        expected_final,
    ) = load_replay()

    observed = np.asarray(
        arrays[
            "obs"
        ]
    )

    normalised = np.asarray(
        arrays[
            "norm"
        ]
    )

    # obs and norm must be byte-scale fixture inputs.
    # make_padded() owns the frozen /255 normalisation.

    if (
        observed.size == 0
        or observed.max() <= 1
    ):
        raise ReplayFailure(
            "Ardross obs appears pre-normalised; "
            "make_padded() must own /255 scaling"
        )

    if (
        normalised.size == 0
        or normalised.max() <= 1
    ):
        raise ReplayFailure(
            "Ardross norm appears pre-normalised; "
            "make_padded() must own /255 scaling"
        )

    # ------------------------------------------------------------
    # Frozen tensor construction
    # ------------------------------------------------------------

    padded = runtime.make_padded(
        arrays,
        schema,
    )

    if not isinstance(
        padded,
        dict,
    ):
        raise ReplayFailure(
            "Frozen tensor builder did not return "
            "a dictionary"
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

    unexpected_padded = sorted(
        set(
            padded
        )
        - set(
            CHANNELS
        )
    )

    if unexpected_padded:
        raise ReplayFailure(
            "Frozen tensor builder returned unexpected "
            f"channels: {unexpected_padded}"
        )

    half = (
        EXPECTED_PATCH_SIZE
        // 2
    )

    expected_padded_shape = (
        observed.shape[
            0
        ]
        + 2 * half,
        observed.shape[
            1
        ]
        + 2 * half,
    )

    for name in CHANNELS:
        value = np.asarray(
            padded[
                name
            ]
        )

        if value.shape != expected_padded_shape:
            raise ReplayFailure(
                f"Padded {name} shape "
                f"{value.shape} != "
                f"{expected_padded_shape}"
            )

        if value.dtype != np.float32:
            raise ReplayFailure(
                f"Padded {name} dtype "
                f"{value.dtype} != float32"
            )

        if not np.isfinite(
            value
        ).all():
            raise ReplayFailure(
                f"Padded {name} contains "
                "non-finite values"
            )

    # Explicitly verify the two frozen /255 channels.

    expected_observed_padded = np.pad(
        arrays[
            "obs"
        ].astype(
            np.float32
        )
        / 255.0,
        half,
        mode="reflect",
    )

    expected_normalised_padded = np.pad(
        arrays[
            "norm"
        ].astype(
            np.float32
        )
        / 255.0,
        half,
        mode="reflect",
    )

    if not np.array_equal(
        padded[
            "obs"
        ],
        expected_observed_padded,
    ):
        raise ReplayFailure(
            "Frozen padded obs channel does not equal "
            "the exact reflect-padded /255 representation"
        )

    if not np.array_equal(
        padded[
            "norm"
        ],
        expected_normalised_padded,
    ):
        raise ReplayFailure(
            "Frozen padded norm channel does not equal "
            "the exact reflect-padded /255 representation"
        )

    # Remaining channels must not receive /255 scaling.

    for name in [
        "mask",
        "depth",
        "horizontal",
        "vertical",
        "compact",
        "preserve",
        "remove",
        "ambiguous",
        "edge",
    ]:
        expected = np.pad(
            arrays[
                name
            ].astype(
                np.float32
            ),
            half,
            mode="reflect",
        )

        if not np.array_equal(
            padded[
                name
            ],
            expected,
        ):
            raise ReplayFailure(
                f"Frozen padded {name} channel differs "
                "from the exact native-float32 "
                "reflect-padded representation"
            )

    # ------------------------------------------------------------
    # Frozen model and isotonic calibration
    # ------------------------------------------------------------

    got_probability = np.asarray(
        runtime.score_coordinates(
            model,
            calibrator,
            padded,
            schema,
            y,
            x,
            device,
            batch=512,
        ),
        dtype=np.float64,
    ).reshape(
        -1
    )

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
        np.any(
            got_probability < 0.0
        )
        or np.any(
            got_probability > 1.0
        )
    ):
        raise ReplayFailure(
            "Ardross scored probability lies "
            "outside 0..1"
        )

    # ------------------------------------------------------------
    # Frozen three-class rule and page-global run filtering
    #
    # This is global over the supplied 5,000 replay coordinates.
    # The fixture does not claim that those 5,000 rows constitute
    # the complete eligible page population.
    # ------------------------------------------------------------

    decision_result = runtime.decisions(
        got_probability,
        policy,
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

    got_provisional = np.char.upper(
        np.asarray(
            decision_result[
                0
            ],
            dtype=str,
        ).reshape(
            -1
        )
    )

    got_final = np.char.upper(
        np.asarray(
            decision_result[
                1
            ],
            dtype=str,
        ).reshape(
            -1
        )
    )

    if not (
        len(
            got_probability
        )
        == len(
            got_provisional
        )
        == len(
            got_final
        )
        == len(
            y
        )
        == len(
            x
        )
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
    # to UNCERTAIN. It must never create PRINT.

    if np.any(
        (
            got_final
            == "PRINT"
        )
        & (
            got_provisional
            != "PRINT"
        )
    ):
        raise ReplayFailure(
            "A5-v1 run filtering created a new "
            "PRINT decision"
        )

    # ANNOTATION is not subject to the PRINT run filter.

    if np.any(
        (
            got_provisional
            == "ANNOTATION"
        )
        & (
            got_final
            != "ANNOTATION"
        )
    ):
        raise ReplayFailure(
            "A5-v1 run filtering altered an "
            "ANNOTATION decision"
        )

    # Existing UNCERTAIN decisions cannot become PRINT.

    if np.any(
        (
            got_provisional
            == "UNCERTAIN"
        )
        & (
            got_final
            == "PRINT"
        )
    ):
        raise ReplayFailure(
            "A5-v1 run filtering promoted "
            "UNCERTAIN to PRINT"
        )

    # The only permitted decision transition is
    # PRINT -> UNCERTAIN.

    changed_decision = (
        got_provisional
        != got_final
    )

    permitted_change = (
        (
            got_provisional
            == "PRINT"
        )
        & (
            got_final
            == "UNCERTAIN"
        )
    )

    if np.any(
        changed_decision
        & ~permitted_change
    ):
        raise ReplayFailure(
            "A5-v1 run filtering made an "
            "unexpected decision transition"
        )

    # ------------------------------------------------------------
    # Exact frozen replay comparisons
    # ------------------------------------------------------------

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
            (
                got_provisional
                == "PRINT"
            )
            & (
                got_final
                == "UNCERTAIN"
            )
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

    if (
        final_print
        + annotation_count
        + uncertain_count
        != EXPECTED_REPLAY_ROWS
    ):
        raise ReplayFailure(
            "Final replay class counts do not sum "
            "to the frozen replay row count"
        )

    if (
        provisional_print
        - final_print
        != run_demotions
    ):
        raise ReplayFailure(
            "Run-demotion count is inconsistent with "
            "the provisional/final PRINT counts"
        )

    # ------------------------------------------------------------
    # Frozen fixture identities
    # ------------------------------------------------------------

    replay_file_hashes = {
        path.name: sha256_file(
            path
        )
        for path in sorted(
            ARDROSS.iterdir()
        )
        if path.is_file()
    }

    array_hashes = {
        name: array_sha256(
            arrays[
                name
            ]
        )
        for name in CHANNELS
    }

    array_hashes.update(
        {
            "y": array_sha256(
                y
            ),
            "x": array_sha256(
                x
            ),
            "expected_probability": array_sha256(
                expected_probability
            ),
            "got_probability": array_sha256(
                got_probability
            ),
            "expected_provisional": array_sha256(
                expected_provisional
            ),
            "got_provisional": array_sha256(
                got_provisional
            ),
            "expected_final": array_sha256(
                expected_final
            ),
            "got_final": array_sha256(
                got_final
            ),
        }
    )

    # ------------------------------------------------------------
    # Replay result
    #
    # IMPORTANT:
    #
    # whole_page_assembly_tested remains False until the replay
    # fixture contains:
    #
    #   1. the exact Notebook 87 A1 page;
    #   2. the expected A5 output page;
    #   3. the expected PRINT mask; and
    #   4. the complete eligible coordinate population needed
    #      for a true whole-page replay.
    #
    # Do not change this field merely to satisfy a startup gate.
    # ------------------------------------------------------------

    result = {
        "name": (
            "ardross_a5_startup_replay"
        ),

        "policy_id": getattr(
            operator,
            "a5_policy_id",
            None,
        ),

        "bundle_id": getattr(
            operator,
            "bundle_id",
            None,
        ),

        "model_sha256": getattr(
            operator,
            "model_sha256",
            None,
        ),

        "calibrator_sha256": getattr(
            operator,
            "calibrator_sha256",
            None,
        ),

        "rows": int(
            len(
                got_probability
            )
        ),

        "expected_rows": (
            EXPECTED_REPLAY_ROWS
        ),

        "channel_order": list(
            CHANNELS
        ),

        "patch_size": (
            EXPECTED_PATCH_SIZE
        ),

        "padding": (
            schema.get(
                "padding",
                "reflect",
            )
        ),

        "coordinate_population": (
            "FROZEN_5000_COORDINATE_REPLAY_SET"
        ),

        "complete_eligible_page_population": False,

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

        # --------------------------------------------------------
        # Positive inference-replay declarations
        # --------------------------------------------------------

        "tensor_replay_tested": True,

        "model_replay_tested": True,

        "calibration_replay_tested": True,

        "decision_replay_tested": True,

        "page_shape_used_for_decisions": True,

        "run_filter_applied_after_all_replay_scores": True,

        # --------------------------------------------------------
        # Honest whole-page assembly status
        # --------------------------------------------------------

        "whole_page_assembly_tested": False,

        "a1_fixture_available": False,

        "expected_a5_fixture_available": False,

        "expected_print_mask_fixture_available": False,

        # These are deliberately unavailable until a genuine
        # whole-page fixture exists.

        "print_equals_observed": None,

        "non_print_equals_a1": None,

        "outside_mask_equals_observed": None,

        "delta_equals_effective_print_restoration": None,

        "expected_a5_exact": None,

        "expected_print_mask_exact": None,

        # --------------------------------------------------------
        # Legacy assembly diagnostics retained for compatibility
        #
        # None means "not tested", not zero.
        # --------------------------------------------------------

        "outside_mask_drift": None,

        "outside_authoritative_mask_drift": None,

        "non_print_baseline_regression": None,

        "non_print_v78_regression": None,

        "print_not_equal_observed": None,

        "changed_byte_not_equal_observed": None,

        "unexpected_changed_byte": None,

        "whole_page_assembly_limitation": (
            "The frozen Ardross fixture contains the "
            "11 channel arrays, 5,000 candidate coordinates, "
            "expected calibrated probabilities and expected "
            "provisional/final decisions. It does not contain "
            "a frozen Notebook 87 A1 page, expected A5 output "
            "page, expected PRINT mask or the complete eligible "
            "coordinate population required for whole-page "
            "A1-to-A5 assembly replay."
        ),

        "replay_directory": str(
            ARDROSS
        ),

        "replay_file_hashes": (
            replay_file_hashes
        ),

        "array_hashes": (
            array_hashes
        ),
    }

    # ------------------------------------------------------------
    # Replay PASS means only that every test actually performed
    # by this fixture passed.
    #
    # whole_page_assembly_tested=False is an explicit limitation,
    # not a failed inference replay.
    # ------------------------------------------------------------

    result[
        "passed"
    ] = bool(
        result[
            "probability_within_atol"
        ]
        and result[
            "provisional_exact"
        ]
        and result[
            "final_exact"
        ]
        and result[
            "tensor_replay_tested"
        ]
        and result[
            "model_replay_tested"
        ]
        and result[
            "calibration_replay_tested"
        ]
        and result[
            "decision_replay_tested"
        ]
    )

    if not result[
        "passed"
    ]:
        raise ReplayFailure(
            "Mandatory Ardross A5 replay failed: "
            + json.dumps(
                result,
                sort_keys=True,
                default=str,
            )
        )

    return result