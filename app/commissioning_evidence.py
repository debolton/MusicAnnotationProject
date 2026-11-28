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

A1_POLICY_ID = "NOTEBOOK87_A1_FROM_FROZEN_V8_9"
A5_POLICY_ID = "A5-v1-print-p0980-margin0960-run2"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _version(distribution: str) -> str | None:
    try:
        return md.version(distribution)
    except Exception:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return

    fieldnames = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def _require_dict(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a dictionary")
    return value


def _require_true(value: dict, field: str, name: str) -> None:
    if value.get(field) is not True:
        raise RuntimeError(f"{name} did not confirm {field}")


def _require_boolean(value: dict, field: str, name: str) -> bool:
    field_value = value.get(field)
    if not isinstance(field_value, bool):
        raise RuntimeError(
            f"{name} must declare {field} as a boolean; "
            f"found {field_value!r}"
        )
    return field_value


def _validate_nonnegative_integer(
    value: Any,
    field: str,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"{name} must declare {field} as an integer; found {value!r}"
        )
    if value < 0:
        raise RuntimeError(f"{name} declared negative {field}: {value}")
    return value


def _validate_zero_diagnostics(
    result: dict,
    fields: list[str],
    name: str,
) -> None:
    for field in fields:
        value = result.get(field)

        # None means the diagnostic was not tested by the fixture.
        if value is None:
            continue

        if isinstance(value, bool):
            numeric_value = int(value)
        elif isinstance(value, (int, float)):
            numeric_value = int(value)
            if float(numeric_value) != float(value):
                raise RuntimeError(
                    f"{name} returned a non-integral diagnostic for "
                    f"{field}: {value!r}"
                )
        else:
            raise RuntimeError(
                f"{name} returned a non-numeric diagnostic for "
                f"{field}: {value!r}"
            )

        if numeric_value != 0:
            raise RuntimeError(
                f"{name} failed {field}: {numeric_value}"
            )


def _page_checks(result: dict) -> list[dict]:
    checks = result.get("checks")
    if not isinstance(checks, list) or not checks:
        raise RuntimeError("A5 replay contains no page checks")
    if not all(isinstance(check, dict) for check in checks):
        raise RuntimeError("A5 replay page checks must be dictionaries")
    return checks


def _all_page_checks_true(checks: list[dict], field: str) -> bool:
    return bool(checks) and all(check.get(field) is True for check in checks)


# ---------------------------------------------------------------------------
# Runtime identities
# ---------------------------------------------------------------------------

def runtime_identities() -> dict[str, Any]:
    out = {
        "app_version": APP_VERSION,
        "hf_version": HF_VERSION,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scikit_learn": _version("scikit-learn"),
        "joblib": _version("joblib"),
        "opencv_python_headless": _version("opencv-python-headless"),
        "opencv_python": _version("opencv-python"),
        "pillow": _version("Pillow"),
        "torch": _version("torch"),
        "torchvision": _version("torchvision"),
        "ultralytics": _version("ultralytics"),
        "fastapi": _version("fastapi"),
        "uvicorn": _version("uvicorn"),
    }

    try:
        import cv2

        out["opencv_runtime"] = cv2.__version__
    except Exception:
        out["opencv_runtime"] = None

    try:
        import torch

        out["torch_runtime"] = torch.__version__
        out["cuda_runtime"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["gpu"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
    except Exception:
        out["torch_runtime"] = None
        out["cuda_runtime"] = None
        out["cuda_available"] = False
        out["gpu"] = None

    return out


# ---------------------------------------------------------------------------
# Replay-result validation
# ---------------------------------------------------------------------------

def _validate_a1_replay(result: dict) -> None:
    _require_dict(result, "V8.9/A1 replay result")
    _require_true(result, "passed", "V8.9/A1 replay")

    policy_id = result.get("a1_policy_id")
    if policy_id != A1_POLICY_ID:
        raise RuntimeError(
            "V8.9/A1 replay policy mismatch: "
            f"{policy_id!r} != {A1_POLICY_ID!r}"
        )

    for field in (
        "v89_operator_identity_verified",
        "v89_bundle_identities_verified",
    ):
        _require_true(result, field, "V8.9/A1 replay")

    for field in (
        "v89_operator_sha256",
        "p_print_bundle_id",
        "remove_bundle_id",
    ):
        if not result.get(field):
            raise RuntimeError(
                f"V8.9/A1 replay omitted identity {field}"
            )


def _validate_a5_replay(result: dict) -> None:
    """
    Validate the frozen A5 exact page replay.

    The replay proves exact calibrated probabilities, provisional decisions,
    final page-global decisions and assembled A5 page bytes for every page in
    the frozen replay contract.

    expected_final.npy is authoritative for the final PRINT population. A
    separately persisted expected PRINT-mask file is optional rather than
    mandatory.
    """
    _require_dict(result, "A5 replay result")
    _require_true(result, "passed", "A5 replay")

    policy_id = result.get("policy_id")
    if str(policy_id) != A5_POLICY_ID:
        raise RuntimeError(
            "A5 replay policy mismatch: "
            f"{policy_id!r} != {A5_POLICY_ID!r}"
        )

    for field in (
        "tensor_replay_tested",
        "model_replay_tested",
        "calibration_replay_tested",
        "decision_replay_tested",
    ):
        _require_true(result, field, "A5 replay")

    checks = _page_checks(result)
    page_count = result.get("page_count")
    if isinstance(page_count, bool) or not isinstance(page_count, int):
        raise RuntimeError("A5 replay page_count must be an integer")
    if page_count != len(checks):
        raise RuntimeError(
            "A5 replay page-count mismatch: "
            f"page_count={page_count}, checks={len(checks)}"
        )

    seen_pages: set[str] = set()
    total_eligible = 0
    total_print = 0

    for index, check in enumerate(checks):
        page = check.get("page")
        if not isinstance(page, str) or not page.strip():
            raise RuntimeError(
                f"A5 replay page check {index} has no valid page name"
            )
        page = page.strip()
        if page in seen_pages:
            raise RuntimeError(
                f"A5 replay contains duplicate page {page!r}"
            )
        seen_pages.add(page)

        eligible = _validate_nonnegative_integer(
            check.get("eligible_pixels"),
            "eligible_pixels",
            f"A5 replay page {page!r}",
        )
        print_pixels = _validate_nonnegative_integer(
            check.get("print_pixels"),
            "print_pixels",
            f"A5 replay page {page!r}",
        )
        if print_pixels > eligible:
            raise RuntimeError(
                f"A5 replay page {page!r}: print_pixels={print_pixels} "
                f"exceeds eligible_pixels={eligible}"
            )

        total_eligible += eligible
        total_print += print_pixels

        for field in (
            "probability_exact",
            "provisional_exact",
            "final_exact",
            "page_exact",
        ):
            if not isinstance(check.get(field), bool):
                raise RuntimeError(
                    f"A5 replay page {page!r}: {field} must be a boolean"
                )
            _require_true(check, field, f"A5 replay page {page!r}")

    if total_eligible <= 0:
        raise RuntimeError("A5 replay tested no eligible pixels")

    replay_directory = result.get("replay_directory")
    if not isinstance(replay_directory, str) or not replay_directory.strip():
        raise RuntimeError("A5 replay did not record its replay directory")

    whole_page_tested = _require_boolean(
        result,
        "whole_page_assembly_tested",
        "A5 replay",
    )

    if not whole_page_tested:
        raise RuntimeError(
            "The current A5 replay contract contains exact expected A5 page "
            "bytes, so whole_page_assembly_tested must be True"
        )

    for field in (
        "a1_fixture_available",
        "expected_a5_fixture_available",
        "complete_eligible_page_population",
        "print_equals_observed",
        "non_print_equals_a1",
        "outside_mask_equals_observed",
        "delta_equals_effective_print_restoration",
        "expected_a5_exact",
        "expected_final_decisions_exact",
    ):
        _require_true(result, field, "A5 whole-page replay")

    mask_fixture_available = _require_boolean(
        result,
        "expected_print_mask_fixture_available",
        "A5 whole-page replay",
    )
    mask_exact = result.get("expected_print_mask_exact")

    if mask_fixture_available:
        if mask_exact is not True:
            raise RuntimeError(
                "A5 replay declares an expected PRINT-mask fixture but did "
                "not confirm expected_print_mask_exact"
            )
    elif mask_exact is not None:
        raise RuntimeError(
            "A5 replay has no separate expected PRINT-mask fixture but "
            f"returned expected_print_mask_exact={mask_exact!r}"
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


def _validate_v78_replay(result: dict) -> None:
    _require_dict(result, "V7.8 equivalence result")
    verdict = str(result.get("verdict", "")).upper()
    passed = result.get("passed") is True or verdict == "PASS"
    if not passed:
        raise RuntimeError("V7.8 equivalence did not pass")


# ---------------------------------------------------------------------------
# Evidence writers
# ---------------------------------------------------------------------------

def _write_a1_evidence(startup_root: Path, result: dict) -> None:
    directory = startup_root / "v89_a1_replay"
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "replay_result.json", result)

    array_hashes = result.get("array_hashes")
    if isinstance(array_hashes, dict):
        _write_json(directory / "array_hashes.json", array_hashes)

    page_comparison = result.get("page_comparison")
    if isinstance(page_comparison, list):
        _write_csv(
            directory / "page_comparison.csv",
            [row for row in page_comparison if isinstance(row, dict)],
        )


def _write_a5_evidence(startup_root: Path, result: dict) -> None:
    directory = startup_root / "ardross_a5_replay"
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "replay_result.json", result)

    checks = result.get("checks")
    if isinstance(checks, list):
        _write_csv(
            directory / "page_replay_checks.csv",
            [row for row in checks if isinstance(row, dict)],
        )

    replay_file_hashes = result.get("replay_file_hashes")
    if isinstance(replay_file_hashes, dict):
        _write_json(
            directory / "replay_file_hashes.json",
            replay_file_hashes,
        )

    array_hashes = result.get("array_hashes")
    if isinstance(array_hashes, dict):
        _write_json(directory / "array_hashes.json", array_hashes)


def _write_v78_evidence(startup_root: Path, result: dict) -> None:
    directory = startup_root / "v78_routing_equivalence"
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "replay_result.json", result)
    _write_json(directory / "summary.json", result)

    rows = result.get("channel_comparison")
    if rows is None:
        rows = result.get("channels")
    if not isinstance(rows, list):
        rows = []

    _write_csv(
        directory / "channel_comparison.csv",
        [row for row in rows if isinstance(row, dict)],
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

    The current A5 replay verifies exact frozen probabilities,
    provisional decisions, final page-global decisions and assembled A5 page
    bytes for every page listed in the replay contract.

    The exact final-decision arrays define the expected final PRINT
    population. A separate expected PRINT-mask artefact is optional and is not
    required when exact final-decision replay has passed.

    Commissioning proves technical integrity and replay only. It does not
    create production authorisation.
    """
    startup_root = Path(startup_root)
    startup_root.mkdir(parents=True, exist_ok=True)

    if a1_replay_result is None:
        raise RuntimeError(
            "HF-04.5.1 commissioning requires a V8.9/Notebook87 A1 "
            "replay result"
        )

    if production_authorised is not None and not isinstance(
        production_authorised,
        bool,
    ):
        raise RuntimeError(
            "production_authorised must be a boolean or None"
        )

    if independent_validation_completed is not None and not isinstance(
        independent_validation_completed,
        bool,
    ):
        raise RuntimeError(
            "independent_validation_completed must be a boolean or None"
        )

    if policy_status is not None and not isinstance(policy_status, str):
        raise RuntimeError("policy_status must be a string or None")

    _validate_a1_replay(a1_replay_result)
    _validate_a5_replay(a5_replay_result)
    _validate_v78_replay(v78_summary)

    page_checks = _page_checks(a5_replay_result)
    a5_whole_page_tested = bool(
        a5_replay_result["whole_page_assembly_tested"]
    )

    probability_exact = _all_page_checks_true(
        page_checks,
        "probability_exact",
    )
    provisional_exact = _all_page_checks_true(
        page_checks,
        "provisional_exact",
    )
    final_exact = _all_page_checks_true(page_checks, "final_exact")
    page_exact = _all_page_checks_true(page_checks, "page_exact")

    # Persist each gate independently.
    _write_a1_evidence(startup_root, a1_replay_result)
    _write_a5_evidence(startup_root, a5_replay_result)
    _write_v78_evidence(startup_root, v78_summary)

    runtime = runtime_identities()
    runtime.update(
        {
            "a1_policy_id": A1_POLICY_ID,
            "v89_operator_sha256": a1_replay_result.get(
                "v89_operator_sha256"
            ),
            "v89_p_print_bundle_id": a1_replay_result.get(
                "p_print_bundle_id"
            ),
            "v89_remove_bundle_id": a1_replay_result.get(
                "remove_bundle_id"
            ),
            "a5_policy_id": A5_POLICY_ID,
            "a5_bundle_id": a5_replay_result.get("bundle_id"),
            "a5_model_sha256": a5_replay_result.get("model_sha256"),
            "a5_calibrator_sha256": a5_replay_result.get(
                "calibrator_sha256"
            ),
            "a5_tensor_replay_tested": bool(
                a5_replay_result.get("tensor_replay_tested", False)
            ),
            "a5_model_replay_tested": bool(
                a5_replay_result.get("model_replay_tested", False)
            ),
            "a5_calibration_replay_tested": bool(
                a5_replay_result.get("calibration_replay_tested", False)
            ),
            "a5_decision_replay_tested": bool(
                a5_replay_result.get("decision_replay_tested", False)
            ),
            "a5_whole_page_assembly_tested": a5_whole_page_tested,
            "a5_whole_page_assembly_limitation": a5_replay_result.get(
                "whole_page_assembly_limitation"
            ),
            "a5_replay_page_count": a5_replay_result.get("page_count"),
            "a5_validated_eligible_pixels": a5_replay_result.get(
                "validated_eligible_pixels"
            ),
            "a5_validated_print_pixels": a5_replay_result.get(
                "validated_print_pixels"
            ),
        }
    )
    _write_json(
        startup_root / "runtime_software_identities.json",
        runtime,
    )

    if a5_whole_page_tested:
        a5_scope = (
            "Frozen A5 tensor construction, model inference, isotonic "
            "calibration, provisional decisions, final page-global "
            "decisions and whole-page A1-to-A5 assembly were "
            "replay-tested exactly."
        )
    else:
        a5_scope = (
            "Frozen A5 inference was replay-tested, but whole-page A1-to-A5 "
            "assembly was not tested by the available fixture."
        )

    verdict = {
        "version": HF_VERSION,
        "app_version": APP_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "integrity_gate_passed": True,
        "runtime_replay_passed": True,
        "candidate_bundle_loadable": True,
        "production_authorised": production_authorised,
        "independent_validation_completed": (
            independent_validation_completed
        ),
        "policy_status": policy_status,
        "authorisation_interpretation": (
            "Commissioning PASS proves only that the declared frozen runtime "
            "lineage passed its integrity and replay checks. It does not "
            "itself authorise production deployment."
        ),
        "a5_replay_scope": a5_scope,
        "a5_whole_page_assembly_tested": a5_whole_page_tested,
        "a5_whole_page_assembly_limitation": a5_replay_result.get(
            "whole_page_assembly_limitation"
        ),
        "lineage": [
            "V7.8 geometry_t80",
            "frozen V8.9 corrected",
            "Notebook 87 A1",
            "A5-v1",
        ],
        "gates": {
            "v78_equivalence": {
                "passed": True,
                "name": v78_summary.get(
                    "name",
                    "ardross_v78_startup_equivalence",
                ),
                "verdict": v78_summary.get("verdict", "PASS"),
                "channels": v78_summary.get("channels"),
                "exact_channels": v78_summary.get("exact_channels"),
                "mismatched_channels": v78_summary.get(
                    "mismatched_channels",
                    [],
                ),
                "evidence": [
                    "v78_routing_equivalence/replay_result.json",
                    "v78_routing_equivalence/summary.json",
                    "v78_routing_equivalence/channel_comparison.csv",
                ],
            },
            "v89_notebook87_a1_replay": {
                "passed": True,
                "name": a1_replay_result.get(
                    "name",
                    "v89_notebook87_a1_replay",
                ),
                "a1_policy_id": a1_replay_result["a1_policy_id"],
                "v89_operator_sha256": a1_replay_result[
                    "v89_operator_sha256"
                ],
                "p_print_bundle_id": a1_replay_result[
                    "p_print_bundle_id"
                ],
                "remove_bundle_id": a1_replay_result[
                    "remove_bundle_id"
                ],
                "v89_operator_identity_verified": a1_replay_result[
                    "v89_operator_identity_verified"
                ],
                "v89_bundle_identities_verified": a1_replay_result[
                    "v89_bundle_identities_verified"
                ],
                "v89_page_replay_tested": a1_replay_result.get(
                    "v89_page_replay_tested"
                ),
                "a1_assembly_tested": a1_replay_result.get(
                    "a1_assembly_tested"
                ),
                "a1_ambiguous_equals_observed": a1_replay_result.get(
                    "a1_ambiguous_equals_observed"
                ),
                "a1_non_ambiguous_equals_v89": a1_replay_result.get(
                    "a1_non_ambiguous_equals_v89"
                ),
                "a1_outside_mask_equals_observed": a1_replay_result.get(
                    "a1_outside_mask_equals_observed"
                ),
                "a1_delta_exact": a1_replay_result.get("a1_delta_exact"),
                "limitation": a1_replay_result.get("limitation"),
                "evidence": "v89_a1_replay/replay_result.json",
            },
            "a5_frozen_replay": {
                "passed": True,
                "name": a5_replay_result.get(
                    "name",
                    "ardross_a5_startup_replay",
                ),
                "policy_id": a5_replay_result.get("policy_id"),
                "bundle_id": a5_replay_result.get("bundle_id"),
                "model_sha256": a5_replay_result.get("model_sha256"),
                "calibrator_sha256": a5_replay_result.get(
                    "calibrator_sha256"
                ),
                "page_count": a5_replay_result.get("page_count"),
                "validated_page_count": a5_replay_result.get(
                    "validated_page_count"
                ),
                "validated_eligible_pixels": a5_replay_result.get(
                    "validated_eligible_pixels"
                ),
                "validated_print_pixels": a5_replay_result.get(
                    "validated_print_pixels"
                ),
                "probability_exact": probability_exact,
                "provisional_exact": provisional_exact,
                "final_exact": final_exact,
                "page_exact": page_exact,
                "tensor_replay_tested": a5_replay_result.get(
                    "tensor_replay_tested"
                ),
                "model_replay_tested": a5_replay_result.get(
                    "model_replay_tested"
                ),
                "calibration_replay_tested": a5_replay_result.get(
                    "calibration_replay_tested"
                ),
                "decision_replay_tested": a5_replay_result.get(
                    "decision_replay_tested"
                ),
                "whole_page_assembly_tested": a5_whole_page_tested,
                "a1_fixture_available": a5_replay_result.get(
                    "a1_fixture_available"
                ),
                "expected_a5_fixture_available": a5_replay_result.get(
                    "expected_a5_fixture_available"
                ),
                "complete_eligible_page_population": a5_replay_result.get(
                    "complete_eligible_page_population"
                ),
                "expected_final_decisions_exact": a5_replay_result.get(
                    "expected_final_decisions_exact"
                ),
                "expected_print_mask_fixture_available": (
                    a5_replay_result.get(
                        "expected_print_mask_fixture_available"
                    )
                ),
                "print_equals_observed": a5_replay_result.get(
                    "print_equals_observed"
                ),
                "non_print_equals_a1": a5_replay_result.get(
                    "non_print_equals_a1"
                ),
                "outside_mask_equals_observed": a5_replay_result.get(
                    "outside_mask_equals_observed"
                ),
                "delta_equals_effective_print_restoration": (
                    a5_replay_result.get(
                        "delta_equals_effective_print_restoration"
                    )
                ),
                "expected_a5_exact": a5_replay_result.get(
                    "expected_a5_exact"
                ),
                "expected_print_mask_exact": a5_replay_result.get(
                    "expected_print_mask_exact"
                ),
                "whole_page_assembly_limitation": a5_replay_result.get(
                    "whole_page_assembly_limitation"
                ),
                "page_checks": page_checks,
                "evidence": [
                    "ardross_a5_replay/replay_result.json",
                    "ardross_a5_replay/page_replay_checks.csv",
                ],
            },
        },
        "runtime_evidence": "runtime_software_identities.json",
    }

    _write_json(startup_root / "startup_verdict.json", verdict)
    return verdict
