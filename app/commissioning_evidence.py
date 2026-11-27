from __future__ import annotations

import csv
import importlib.metadata as md
import json
import platform
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_VERSION = "4.5.1"
HF_VERSION = "HF-04.5.1"

A1_POLICY_ID = (
    "NOTEBOOK87_A1_FROM_FROZEN_V8_9"
)

A5_POLICY_ID = (
    "A5-v1-print-p0980-margin0960-run2"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _version(
    distribution: str,
) -> str | None:
    try:
        return md.version(
            distribution
        )

    except Exception:
        return None


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def _write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    if not rows:
        temporary.write_text(
            "",
            encoding="utf-8",
        )

        temporary.replace(
            path
        )

        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )

    temporary.replace(
        path
    )


def _require_dict(
    value: Any,
    name: str,
) -> dict:
    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"{name} must be a dictionary"
        )

    return value


def _require_true(
    value: dict,
    field: str,
    name: str,
) -> None:
    if value.get(
        field
    ) is not True:
        raise RuntimeError(
            f"{name} did not confirm {field}"
        )


def _require_boolean(
    value: dict,
    field: str,
    name: str,
) -> bool:
    field_value = value.get(
        field
    )

    if not isinstance(
        field_value,
        bool,
    ):
        raise RuntimeError(
            f"{name} must declare {field} "
            "as a boolean; "
            f"found {field_value!r}"
        )

    return field_value


def _validate_zero_diagnostics(
    result: dict,
    fields: list[str],
    name: str,
) -> None:
    for field in fields:
        value = result.get(
            field
        )

        # None means the diagnostic was not tested by the
        # current replay fixture.
        if value is None:
            continue

        if isinstance(
            value,
            bool,
        ):
            numeric_value = int(
                value
            )

        elif isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            numeric_value = int(
                value
            )

            if float(
                numeric_value
            ) != float(
                value
            ):
                raise RuntimeError(
                    f"{name} returned a non-integral "
                    f"diagnostic for {field}: "
                    f"{value!r}"
                )

        else:
            raise RuntimeError(
                f"{name} returned a non-numeric "
                f"diagnostic for {field}: "
                f"{value!r}"
            )

        if numeric_value != 0:
            raise RuntimeError(
                f"{name} failed {field}: "
                f"{numeric_value}"
            )


# ---------------------------------------------------------------------------
# Runtime identities
# ---------------------------------------------------------------------------

def runtime_identities() -> dict[str, Any]:
    out = {
        "app_version": APP_VERSION,
        "hf_version": HF_VERSION,

        "python": sys.version,

        "python_executable": (
            sys.executable
        ),

        "platform": (
            platform.platform()
        ),

        "numpy": _version(
            "numpy"
        ),

        "pandas": _version(
            "pandas"
        ),

        "scikit_learn": _version(
            "scikit-learn"
        ),

        "joblib": _version(
            "joblib"
        ),

        "opencv_python_headless": (
            _version(
                "opencv-python-headless"
            )
        ),

        "opencv_python": _version(
            "opencv-python"
        ),

        "pillow": _version(
            "Pillow"
        ),

        "torch": _version(
            "torch"
        ),

        "torchvision": _version(
            "torchvision"
        ),

        "ultralytics": _version(
            "ultralytics"
        ),

        "fastapi": _version(
            "fastapi"
        ),

        "uvicorn": _version(
            "uvicorn"
        ),
    }

    try:
        import cv2

        out[
            "opencv_runtime"
        ] = cv2.__version__

    except Exception:
        out[
            "opencv_runtime"
        ] = None

    try:
        import torch

        out[
            "torch_runtime"
        ] = torch.__version__

        out[
            "cuda_runtime"
        ] = torch.version.cuda

        out[
            "cuda_available"
        ] = bool(
            torch.cuda.is_available()
        )

        out[
            "gpu"
        ] = (
            torch.cuda.get_device_name(
                0
            )
            if torch.cuda.is_available()
            else None
        )

    except Exception:
        out[
            "torch_runtime"
        ] = None

        out[
            "cuda_runtime"
        ] = None

        out[
            "cuda_available"
        ] = False

        out[
            "gpu"
        ] = None

    return out


# ---------------------------------------------------------------------------
# Replay-result validation
# ---------------------------------------------------------------------------

def _validate_a1_replay(
    result: dict,
) -> None:
    _require_dict(
        result,
        "V8.9/A1 replay result",
    )

    _require_true(
        result,
        "passed",
        "V8.9/A1 replay",
    )

    policy_id = result.get(
        "a1_policy_id"
    )

    if policy_id != A1_POLICY_ID:
        raise RuntimeError(
            "V8.9/A1 replay policy mismatch: "
            f"{policy_id!r} "
            f"!= {A1_POLICY_ID!r}"
        )

    required_true = [
        "v89_operator_identity_verified",
        "v89_bundle_identities_verified",
        "v89_page_replay_tested",
        "a1_assembly_tested",
        "a1_ambiguous_equals_observed",
        "a1_non_ambiguous_equals_v89",
        "a1_outside_mask_equals_observed",
        "a1_delta_exact",
    ]

    for field in required_true:
        _require_true(
            result,
            field,
            "V8.9/A1 replay",
        )

    for field in (
        "v89_operator_sha256",
        "p_print_bundle_id",
        "remove_bundle_id",
    ):
        if not result.get(
            field
        ):
            raise RuntimeError(
                "V8.9/A1 replay omitted "
                f"identity {field}"
            )


def _validate_a5_replay(
    result: dict,
) -> None:
    """
    Validate the frozen A5 replay without claiming that the
    coordinate-only Ardross fixture proves whole-page assembly.

    The current fixture must prove:

        - tensor construction;
        - model inference;
        - isotonic calibration;
        - provisional and final decisions.

    Whole-page assembly checks become mandatory only when the replay
    explicitly declares whole_page_assembly_tested=True.
    """

    _require_dict(
        result,
        "A5 replay  result",
    )

    _require_true(
        result,
        "passed",
       "A5 replay",
    )

    policy_id = result.get(
        "policy_id"
    )

    if str(
        policy_id
    ) != A5_POLICY_ID:
        raise RuntimeError(
            "A5 replay policy mismatch: "
            f"{policy_id!r} "
            f"!= {A5_POLICY_ID!r}"
        )

    required_inference_checks = [
        "tensor_replay_tested",
        "model_replay_tested",
        "calibration_replay_tested",
        "decision_replay_tested",
    ]

    for field in required_inference_checks:
        _require_true(
            result,
            field,
            "A5 replay",
        )

    whole_page_tested = _require_boolean(
        result,
        "whole_page_assembly_tested",
        "A5 replay",
    )

    fixture_flags = [
        "a1_fixture_available",
        "expected_a5_fixture_available",
        "expected_print_mask_fixture_available",
        "complete_eligible_page_population",
    ]

    assembly_checks = [
        "print_equals_observed",
        "non_print_equals_a1",
        "outside_mask_equals_observed",
        (
            "delta_equals_"
            "effective_print_restoration"
        ),
        "expected_a5_exact",
        "expected_print_mask_exact",
    ]

    if whole_page_tested:
        for field in fixture_flags:
            _require_true(
                result,
                field,
                "A5 whole-page replay",
            )

        for field in assembly_checks:
            _require_true(
                result,
                field,
                "A5 whole-page replay",
            )

    else:
        # A coordinate-only replay must not report positive
        # whole-page assembly assertions.

        for field in fixture_flags:
            if result.get(
                field
            ) is not False:
                raise RuntimeError(
                    "A5 replay declared whole-page "
                    "assembly untested but did not "
                    f"declare {field}=False"
                )

        for field in assembly_checks:
            if result.get(
                field
            ) is not None:
                raise RuntimeError(
                    "A5 replay declared whole-page "
                    "assembly untested but returned "
                    f"{field}={result.get(field)!r}"
                )

        limitation = result.get(
            "whole_page_assembly_limitation"
        )

        if (
            not isinstance(
                limitation,
                str,
            )
            or not limitation.strip()
        ):
            raise RuntimeError(
                "Coordinate-only A5 replay must "
                "record its whole-page assembly "
                "limitation"
            )

    _validate_zero_diagnostics(
        result=result,
        fields=[
            "outside_mask_drift",
            "outside_authoritative_mask_drift",
            "non_print_baseline_regression",
            "non_print_v78_regression",
            "print_not_equal_observed",
            "changed_byte_not_equal_observed",
            "unexpected_changed_byte",
        ],
        name="A5 replay",
    )


def _validate_v78_replay(
    result: dict,
) -> None:
    _require_dict(
        result,
        "V7.8 equivalence result",
    )

    verdict = str(
        result.get(
            "verdict",
            "",
        )
    ).upper()

    passed = (
        result.get(
            "passed"
        ) is True
        or verdict == "PASS"
    )

    if not passed:
        raise RuntimeError(
            "V7.8 equivalence did not pass"
        )


# ---------------------------------------------------------------------------
# A1/V8.9 evidence
# ---------------------------------------------------------------------------

def _write_a1_evidence(
    startup_root: Path,
    result: dict,
) -> None:
    directory = (
        startup_root
        / "v89_a1_replay"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        directory
        / "replay_result.json",
        result,
    )

    array_hashes = result.get(
        "array_hashes"
    )

    if isinstance(
        array_hashes,
        dict,
    ):
        _write_json(
            directory
            / "array_hashes.json",
            array_hashes,
        )

    page_comparison = result.get(
        "page_comparison"
    )

    if isinstance(
        page_comparison,
        list,
    ):
        rows = [
            row
            for row in page_comparison
            if isinstance(
                row,
                dict,
            )
        ]

        _write_csv(
            directory
            / "page_comparison.csv",
            rows,
        )


# ---------------------------------------------------------------------------
# A5 evidence
# ---------------------------------------------------------------------------

def _write_a5_evidence(
    startup_root: Path,
    result: dict,
) -> None:
    directory = (
        startup_root
        / "ardross_a5_replay"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        directory
        / "replay_result.json",
        result,
    )

    replay_file_hashes = result.get(
        "replay_file_hashes"
    )

    if isinstance(
        replay_file_hashes,
        dict,
    ):
        _write_json(
            directory
            / "replay_file_hashes.json",
            replay_file_hashes,
        )

    array_hashes = result.get(
        "array_hashes"
    )

    if isinstance(
        array_hashes,
        dict,
    ):
        _write_json(
            directory
            / "array_hashes.json",
            array_hashes,
        )


# ---------------------------------------------------------------------------
# V7.8 evidence
# ---------------------------------------------------------------------------

def _write_v78_evidence(
    startup_root: Path,
    result: dict,
) -> None:
    directory = (
        startup_root
        / "v78_routing_equivalence"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        directory
        / "replay_result.json",
        result,
    )

    # main.py currently requires summary.json.
    _write_json(
        directory
        / "summary.json",
        result,
    )

    rows = result.get(
        "channel_comparison"
    )

    if rows is None:
        rows = result.get(
            "channels"
        )

    if isinstance(
        rows,
        list,
    ):
        rows = [
            row
            for row in rows
            if isinstance(
                row,
                dict,
            )
        ]

    else:
        rows = []

    # Preserve the required physical file even if the existing
    # V7.8 replay result has no row-level comparison structure.
    _write_csv(
        directory
        / "channel_comparison.csv",
        rows,
    )


# ---------------------------------------------------------------------------
# Complete HF-04.5.1 commissioning pack
# ---------------------------------------------------------------------------

def write_commissioning_evidence(
    startup_root: Path,
    a5_replay_result: dict,
    v78_summary: dict,
    a1_replay_result: dict | None = None,
    *,
    production_authorised: bool | None = None,
    independent_validation_completed: bool | None = None,
    policy_status: str | None = None,
) -> dict:
    """
    Persist the HF-04.5.1 commissioning evidence pack.

    Required technical lineage:

        frozen V7.8
            -> frozen V8.9
            -> Notebook 87 A1
            -> frozen A5-v1

    The commissioning verdict proves only the integrity and replay
    checks that were genuinely performed.

    The current Ardross A5 fixture proves the frozen inference path
    over its 5,000-coordinate replay population. It does not prove
    whole-page A1-to-A5 assembly unless the replay result explicitly
    declares whole_page_assembly_tested=True and supplies all required
    whole-page fixture and equality checks.

    Commissioning does not create production authorisation.

    Compatibility:
        a1_replay_result is optional only so an older caller fails
        with a clear commissioning error rather than a Python
        signature error.
    """

    startup_root = Path(
        startup_root
    )

    startup_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if a1_replay_result is None:
        raise RuntimeError(
            "HF-04.5.1 commissioning requires a "
            "V8.9/Notebook87 A1 replay result"
        )

    if (
        production_authorised is not None
        and not isinstance(
            production_authorised,
            bool,
        )
    ):
        raise RuntimeError(
            "production_authorised must be "
            "a boolean or None"
        )

    if (
        independent_validation_completed
        is not None
        and not isinstance(
            independent_validation_completed,
            bool,
        )
    ):
        raise RuntimeError(
            "independent_validation_completed must "
            "be a boolean or None"
        )

    if (
        policy_status is not None
        and not isinstance(
            policy_status,
            str,
        )
    ):
        raise RuntimeError(
            "policy_status must be a string or None"
        )

    _validate_a1_replay(
        a1_replay_result
    )

    _validate_a5_replay(
        a5_replay_result
    )

    _validate_v78_replay(
        v78_summary
    )

    a5_whole_page_tested = bool(
        a5_replay_result[
            "whole_page_assembly_tested"
        ]
    )

    # ------------------------------------------------------------
    # Persist each gate independently.
    # ------------------------------------------------------------

    _write_a1_evidence(
        startup_root,
        a1_replay_result,
    )

    _write_a5_evidence(
        startup_root,
        a5_replay_result,
    )

    _write_v78_evidence(
        startup_root,
        v78_summary,
    )

    # ------------------------------------------------------------
    # Runtime/software identity
    # ------------------------------------------------------------

    runtime = runtime_identities()

    runtime.update(
        {
            "a1_policy_id": (
                A1_POLICY_ID
            ),

            "v89_operator_sha256": (
                a1_replay_result.get(
                    "v89_operator_sha256"
                )
            ),

            "v89_p_print_bundle_id": (
                a1_replay_result.get(
                    "p_print_bundle_id"
                )
            ),

            "v89_remove_bundle_id": (
                a1_replay_result.get(
                    "remove_bundle_id"
                )
            ),

            "a5_policy_id": (
                A5_POLICY_ID
            ),

            "a5_bundle_id": (
                a5_replay_result.get(
                    "bundle_id"
                )
            ),

            "a5_model_sha256": (
                a5_replay_result.get(
                    "model_sha256"
                )
            ),

            "a5_calibrator_sha256": (
                a5_replay_result.get(
                    "calibrator_sha256"
                )
            ),

            "a5_tensor_replay_tested": bool(
                a5_replay_result.get(
                    "tensor_replay_tested",
                    False,
                )
            ),

            "a5_model_replay_tested": bool(
                a5_replay_result.get(
                    "model_replay_tested",
                    False,
                )
            ),

            "a5_calibration_replay_tested": bool(
                a5_replay_result.get(
                    "calibration_replay_tested",
                    False,
                )
            ),

            "a5_decision_replay_tested": bool(
                a5_replay_result.get(
                    "decision_replay_tested",
                    False,
                )
            ),

            "a5_whole_page_assembly_tested": (
                a5_whole_page_tested
            ),

            "a5_whole_page_assembly_limitation": (
                a5_replay_result.get(
                    "whole_page_assembly_limitation"
                )
            ),
        }
    )

    runtime_path = (
        startup_root
        / "runtime_software_identities.json"
    )

    _write_json(
        runtime_path,
        runtime,
    )

    # ------------------------------------------------------------
    # Commissioning interpretation
    # ------------------------------------------------------------

    if a5_whole_page_tested:
        a5_scope = (
            "Frozen A5 tensor construction, model inference, "
            "calibration, decisions and whole-page A1-to-A5 "
            "assembly were replay-tested."
        )

    else:
        a5_scope = (
            "Frozen A5 tensor construction, model inference, "
            "calibration and decisions were replay-tested over "
            "the frozen 5,000-coordinate Ardross population. "
            "Whole-page A1-to-A5 assembly was not tested by the "
            "available fixture."
        )

    # ------------------------------------------------------------
    # Commissioning verdict
    #
    # "passed" remains for compatibility with main.py. It means
    # that every mandatory check applicable to the available
    # declared fixtures passed. It does not mean that an untested
    # whole-page A5 replay occurred.
    # ------------------------------------------------------------

    verdict = {
        "version": HF_VERSION,

        "app_version": APP_VERSION,

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        # Compatibility headline used by /startup-evidence.
        "passed": True,

        # Explicit interpretation.
        "integrity_gate_passed": True,

        "runtime_replay_passed": True,

        "candidate_bundle_loadable": True,

        "production_authorised": (
            production_authorised
        ),

        "independent_validation_completed": (
            independent_validation_completed
        ),

        "policy_status": (
            policy_status
        ),

        "authorisation_interpretation": (
            "Commissioning PASS proves only that the "
            "declared frozen runtime lineage passed the "
            "integrity and replay checks applicable to "
            "the available fixtures. It does not itself "
            "authorise production deployment and does not "
            "convert an untested whole-page replay into a "
            "tested one."
        ),

        "a5_replay_scope": (
            a5_scope
        ),

        "a5_whole_page_assembly_tested": (
            a5_whole_page_tested
        ),

        "a5_whole_page_assembly_limitation": (
            None
            if a5_whole_page_tested
            else a5_replay_result.get(
                "whole_page_assembly_limitation"
            )
        ),

        "lineage": [
            "V7.8 geometry_t80",
            "frozen V8.9 corrected",
            "Notebook 87 A1",
            "A5-v1",
        ],

        "gates": {
            # ----------------------------------------------------
            # V7.8
            # ----------------------------------------------------

            "v78_equivalence": {
                "passed": True,

                "name": (
                    v78_summary.get(
                        "name",
                        (
                            "ardross_v78_"
                            "startup_equivalence"
                        ),
                    )
                ),

                "verdict": (
                    v78_summary.get(
                        "verdict",
                        "PASS",
                    )
                ),

                "channels": (
                    v78_summary.get(
                        "channels"
                    )
                ),

                "exact_channels": (
                    v78_summary.get(
                        "exact_channels"
                    )
                ),

                "mismatched_channels": (
                    v78_summary.get(
                        "mismatched_channels",
                        [],
                    )
                ),

                "evidence": [
                    (
                        "v78_routing_equivalence/"
                        "replay_result.json"
                    ),
                    (
                        "v78_routing_equivalence/"
                        "summary.json"
                    ),
                    (
                        "v78_routing_equivalence/"
                        "channel_comparison.csv"
                    ),
                ],
            },

            # ----------------------------------------------------
            # Frozen V8.9 -> Notebook 87 A1
            # ----------------------------------------------------

            "v89_notebook87_a1_replay": {
                "passed": True,

                "name": (
                    a1_replay_result.get(
                        "name",
                        "v89_notebook87_a1_replay",
                    )
                ),

                "a1_policy_id": (
                    a1_replay_result[
                        "a1_policy_id"
                    ]
                ),

                "v89_operator_sha256": (
                    a1_replay_result[
                        "v89_operator_sha256"
                    ]
                ),

                "p_print_bundle_id": (
                    a1_replay_result[
                        "p_print_bundle_id"
                    ]
                ),

                "remove_bundle_id": (
                    a1_replay_result[
                        "remove_bundle_id"
                    ]
                ),

                "v89_operator_identity_verified": (
                    a1_replay_result[
                        "v89_operator_identity_verified"
                    ]
                ),

                "v89_bundle_identities_verified": (
                    a1_replay_result[
                        "v89_bundle_identities_verified"
                    ]
                ),

                "v89_page_replay_tested": (
                    a1_replay_result[
                        "v89_page_replay_tested"
                    ]
                ),

                "a1_assembly_tested": (
                    a1_replay_result[
                        "a1_assembly_tested"
                    ]
                ),

                "a1_ambiguous_equals_observed": (
                    a1_replay_result[
                        "a1_ambiguous_equals_observed"
                    ]
                ),

                "a1_non_ambiguous_equals_v89": (
                    a1_replay_result[
                        "a1_non_ambiguous_equals_v89"
                    ]
                ),

                "a1_outside_mask_equals_observed": (
                    a1_replay_result[
                        "a1_outside_mask_equals_observed"
                    ]
                ),

                "a1_delta_exact": (
                    a1_replay_result[
                        "a1_delta_exact"
                    ]
                ),

                "evidence": (
                    "v89_a1_replay/"
                    "replay_result.json"
                ),
            },

            # ----------------------------------------------------
            # Frozen A5 inference replay
            #
            # Whole-page fields reflect the actual replay result.
            # They are never hard-coded to True.
            # ----------------------------------------------------

            "a5_frozen_replay": {
                "passed": True,

                "name": (
                    a5_replay_result.get(
                        "name",
                        "ardross_a5_startup_replay",
                    )
                ),

                "policy_id": (
                    a5_replay_result.get(
                        "policy_id"
                    )
                ),

                "bundle_id": (
                    a5_replay_result.get(
                        "bundle_id"
                    )
                ),

                "model_sha256": (
                    a5_replay_result.get(
                        "model_sha256"
                    )
                ),

                "calibrator_sha256": (
                    a5_replay_result.get(
                        "calibrator_sha256"
                    )
                ),

                "rows": (
                    a5_replay_result.get(
                        "rows"
                    )
                ),

                "expected_rows": (
                    a5_replay_result.get(
                        "expected_rows"
                    )
                ),

                "coordinate_population": (
                    a5_replay_result.get(
                        "coordinate_population"
                    )
                ),

                "complete_eligible_page_population": (
                    a5_replay_result.get(
                        "complete_eligible_page_population"
                    )
                ),

                "probability_atol": (
                    a5_replay_result.get(
                        "probability_atol"
                    )
                ),

                "probability_exact": (
                    a5_replay_result.get(
                        "probability_exact"
                    )
                ),

                "probability_within_atol": (
                    a5_replay_result.get(
                        "probability_within_atol"
                    )
                ),

                "probability_max_abs_error": (
                    a5_replay_result.get(
                        "probability_max_abs_error"
                    )
                ),

                "provisional_exact": (
                    a5_replay_result.get(
                        "provisional_exact"
                    )
                ),

                "provisional_mismatches": (
                    a5_replay_result.get(
                        "provisional_mismatches",
                        0,
                    )
                ),

                "final_exact": (
                    a5_replay_result.get(
                        "final_exact"
                    )
                ),

                "final_mismatches": (
                    a5_replay_result.get(
                        "final_mismatches",
                        0,
                    )
                ),

                "tensor_replay_tested": (
                    a5_replay_result.get(
                        "tensor_replay_tested"
                    )
                ),

                "model_replay_tested": (
                    a5_replay_result.get(
                        "model_replay_tested"
                    )
                ),

                "calibration_replay_tested": (
                    a5_replay_result.get(
                        "calibration_replay_tested"
                    )
                ),

                "decision_replay_tested": (
                    a5_replay_result.get(
                        "decision_replay_tested"
                    )
                ),

                "page_shape_used_for_decisions": (
                    a5_replay_result.get(
                        "page_shape_used_for_decisions"
                    )
                ),

                (
                    "run_filter_applied_after_"
                    "all_replay_scores"
                ): (
                    a5_replay_result.get(
                        (
                            "run_filter_applied_after_"
                            "all_replay_scores"
                        )
                    )
                ),

                "whole_page_assembly_tested": (
                    a5_whole_page_tested
                ),

                "a1_fixture_available": (
                    a5_replay_result.get(
                        "a1_fixture_available"
                    )
                ),

                "expected_a5_fixture_available": (
                    a5_replay_result.get(
                        "expected_a5_fixture_available"
                    )
                ),

                (
                    "expected_print_mask_"
                    "fixture_available"
                ): (
                    a5_replay_result.get(
                        (
                            "expected_print_mask_"
                            "fixture_available"
                        )
                    )
                ),

                "print_equals_observed": (
                    a5_replay_result.get(
                        "print_equals_observed"
                    )
                ),

                "non_print_equals_a1": (
                    a5_replay_result.get(
                        "non_print_equals_a1"
                    )
                ),

                "outside_mask_equals_observed": (
                    a5_replay_result.get(
                        "outside_mask_equals_observed"
                    )
                ),

                (
                    "delta_equals_"
                    "effective_print_restoration"
                ): (
                    a5_replay_result.get(
                        (
                            "delta_equals_"
                            "effective_print_restoration"
                        )
                    )
                ),

                "expected_a5_exact": (
                    a5_replay_result.get(
                        "expected_a5_exact"
                    )
                ),

                "expected_print_mask_exact": (
                    a5_replay_result.get(
                        "expected_print_mask_exact"
                    )
                ),

                "whole_page_assembly_limitation": (
                    a5_replay_result.get(
                        "whole_page_assembly_limitation"
                    )
                ),

                "evidence": (
                    "ardross_a5_replay/"
                    "replay_result.json"
                ),
            },
        },

        "runtime_evidence": (
            "runtime_software_identities.json"
        ),
    }

    verdict_path = (
        startup_root
        / "startup_verdict.json"
    )

    _write_json(
        verdict_path,
        verdict,
    )

    return verdict