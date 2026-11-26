from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import platform
import re
from importlib import metadata

from app.commissioning_evidence import (
    write_commissioning_evidence,
)


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "frozen" / "a5_v1"

POLICY_ID = "A5-v1-print-p0980-margin0960-run2"

IGNORE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

REQUIRED_BUNDLE_FILES = [
    "policy.json",
    "channel_schema.json",
    "model.pt",
    "model_config.json",
    "isotonic_calibrator.joblib",
    "inference_runtime.py",
    "artifact_manifest.csv",
    "freeze_manifest.json",
]

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
    "policy_id": POLICY_ID,
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

EXPECTED_PACKAGES = {
    "scikit-learn": "1.6.1",
    "joblib": "1.6.0",
}

SHA256_PATTERN = re.compile(
    r"^[0-9a-fA-F]{64}$"
)


def generated(
    relative_path: str | Path,
) -> bool:
    path = Path(relative_path)

    return (
        "__pycache__" in path.parts
        or path.suffix.lower() in {
            ".pyc",
            ".pyo",
        }
        or path.name in IGNORE_NAMES
    )


def sha256_file(
    path: Path,
) -> str:
    if not path.is_file():
        raise RuntimeError(
            f"Cannot hash missing file: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1048576),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json_object(
    path: Path,
    name: str,
) -> dict:
    if not path.is_file():
        raise RuntimeError(
            f"Missing {name}: {path}"
        )

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


def assert_float_equal(
    field: str,
    actual,
    expected: float,
) -> None:
    try:
        actual_number = float(actual)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"A5 policy field {field!r} is not numeric: "
            f"{actual!r}"
        ) from exc

    if actual_number != float(expected):
        raise RuntimeError(
            f"A5 policy mismatch for {field}: "
            f"{actual_number!r} != {expected!r}"
        )


def validate_policy(
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
            assert_float_equal(
                key,
                actual,
                expected,
            )
        elif actual != expected:
            raise RuntimeError(
                f"A5 policy mismatch for {key}: "
                f"{actual!r} != {expected!r}"
            )

    effective_threshold = max(
        float(policy["print_threshold"]),
        (
            1.0
            + float(policy["print_margin"])
        )
        / 2.0,
    )

    if effective_threshold != float(
        policy["effective_print_threshold"]
    ):
        raise RuntimeError(
            "A5 effective PRINT threshold is "
            "internally inconsistent: "
            f"calculated={effective_threshold!r}, "
            "recorded="
            f"{policy['effective_print_threshold']!r}"
        )


def validate_freeze_manifest(
    freeze_manifest: dict,
    policy: dict,
) -> None:
    manifest_policy_id = (
        freeze_manifest.get("policy_id")
        or freeze_manifest.get("a5_policy_id")
    )

    if manifest_policy_id is None:
        raise RuntimeError(
            "freeze_manifest.json does not declare "
            "a policy_id"
        )

    if str(manifest_policy_id) != POLICY_ID:
        raise RuntimeError(
            "Freeze-manifest policy mismatch: "
            f"{manifest_policy_id!r} != {POLICY_ID!r}"
        )

    status = freeze_manifest.get("status")

    if (
        status is not None
        and str(status)
        != str(policy["status"])
    ):
        raise RuntimeError(
            "Freeze-manifest status differs from policy: "
            f"{status!r} != {policy['status']!r}"
        )

    production_authorised = (
        freeze_manifest.get(
            "production_authorised"
        )
    )

    if (
        production_authorised is not None
        and bool(production_authorised)
        is not bool(
            policy["production_authorised"]
        )
    ):
        raise RuntimeError(
            "Freeze-manifest production-authorisation "
            "value differs from policy"
        )

    independent_validation = (
        freeze_manifest.get(
            "independent_validation_completed"
        )
    )

    if (
        independent_validation is not None
        and bool(independent_validation)
        is not bool(
            policy[
                "independent_validation_completed"
            ]
        )
    ):
        raise RuntimeError(
            "Freeze-manifest independent-validation "
            "value differs from policy"
        )


def validate_artifact_manifest(
    manifest_path: Path,
) -> list[dict]:
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Missing artifact manifest: {manifest_path}"
        )

    verified = []

    with manifest_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                "artifact_manifest.csv has no header"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            relative_path = (
                row.get("relative_path")
                or row.get("path")
                or row.get("file")
            )

            expected_hash = (
                row.get("sha256")
                or row.get("SHA-256")
            )

            relative_path = (
                str(relative_path).strip()
                if relative_path is not None
                else ""
            )

            expected_hash = (
                str(expected_hash).strip().lower()
                if expected_hash is not None
                else ""
            )

            if not relative_path:
                raise RuntimeError(
                    "artifact_manifest.csv row "
                    f"{row_number} has no relative path"
                )

            if generated(relative_path):
                continue

            relative = Path(relative_path)

            if relative.is_absolute():
                raise RuntimeError(
                    "Artifact manifest contains an absolute "
                    f"path at row {row_number}: {relative}"
                )

            if ".." in relative.parts:
                raise RuntimeError(
                    "Artifact manifest path escapes the "
                    f"bundle at row {row_number}: {relative}"
                )

            if not SHA256_PATTERN.fullmatch(
                expected_hash
            ):
                raise RuntimeError(
                    "Artifact manifest contains an invalid "
                    f"SHA-256 at row {row_number}: "
                    f"{expected_hash!r}"
                )

            artifact_path = BUNDLE / relative

            if not artifact_path.is_file():
                raise RuntimeError(
                    "A5 artefact is missing: "
                    f"{relative_path}"
                )

            actual_hash = sha256_file(
                artifact_path
            )

            print(
                f"Filename: {artifact_path}"
            )
            print(
                f"SHA-256: {actual_hash}"
            )
            print("")

            if actual_hash.lower() != expected_hash:
                raise RuntimeError(
                    "A5 artefact hash failure: "
                    f"{relative_path}: "
                    f"{actual_hash} != {expected_hash}"
                )

            verified.append(
                {
                    "relative_path": relative_path,
                    "sha256": actual_hash,
                    "bytes": artifact_path.stat().st_size,
                }
            )

    if not verified:
        raise RuntimeError(
            "artifact_manifest.csv contains no "
            "verifiable frozen artefacts"
        )

    return verified


def validate_package_versions() -> dict:
    actual_versions = {}

    for package, expected in (
        EXPECTED_PACKAGES.items()
    ):
        try:
            actual = metadata.version(
                package
            )
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required package is not installed: "
                f"{package}=={expected}"
            ) from exc

        actual_versions[package] = actual

        if actual != expected:
            raise RuntimeError(
                f"{package} must be {expected}; "
                f"found {actual}"
            )

    return actual_versions


def validate_a5_replay(
    replay,
) -> None:
    if not isinstance(replay, dict):
        raise RuntimeError(
            "A5 startup replay must return a dictionary"
        )

    if replay.get("passed") is not True:
        raise RuntimeError(
            "A5 startup replay did not pass"
        )

    if replay.get("policy_id") != POLICY_ID:
        raise RuntimeError(
            "A5 startup replay policy mismatch: "
            f"{replay.get('policy_id')!r}"
        )

    required_true = [
        "tensor_replay_tested",
        "model_replay_tested",
        "calibration_replay_tested",
        "decision_replay_tested",
    ]

    for field in required_true:
        if replay.get(field) is not True:
            raise RuntimeError(
                "A5 startup replay did not confirm "
                f"{field}"
            )

    # The current 5,000-coordinate Ardross fixture does not
    # contain a frozen geometry_t80 page and expected A5 page.
    # It must therefore report this honestly.
    if replay.get(
        "whole_page_assembly_tested"
    ) not in {
        False,
        None,
    }:
        raise RuntimeError(
            "Unexpected whole-page replay declaration"
        )


def validate_v78_equivalence(
    result,
) -> None:
    if not isinstance(result, dict):
        raise RuntimeError(
            "V7.8 equivalence gate must return "
            "a dictionary"
        )

    verdict = str(
        result.get(
            "verdict",
            "",
        )
    ).upper()

    passed = (
        result.get("passed") is True
        or verdict == "PASS"
    )

    if not passed:
        raise RuntimeError(
            "Mandatory V7.8 equivalence gate failed: "
            f"{json.dumps(result, default=str)}"
        )


def gate():
    missing = [
        name
        for name in REQUIRED_BUNDLE_FILES
        if not (BUNDLE / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            "A5 bundle incomplete: "
            + ", ".join(missing)
        )

    policy = load_json_object(
        BUNDLE / "policy.json",
        "A5 policy",
    )

    validate_policy(policy)

    freeze_manifest = load_json_object(
        BUNDLE / "freeze_manifest.json",
        "A5 freeze manifest",
    )

    validate_freeze_manifest(
        freeze_manifest,
        policy,
    )

    verified_artifacts = (
        validate_artifact_manifest(
            BUNDLE / "artifact_manifest.csv"
        )
    )

    package_versions = (
        validate_package_versions()
    )

    from app.a5_operator import A5Operator

    operator = A5Operator(
        run_startup_replay=True,
    )

    validate_a5_replay(
        operator.startup_replay
    )

    from app.hf044_v78_startup import (
        run_ardross_v78_equivalence,
    )

    v78 = run_ardross_v78_equivalence()

    validate_v78_equivalence(v78)

    gate_record = {
        "passed": True,
        "policy_id": POLICY_ID,
        "policy_status": policy["status"],
        "production_authorised": bool(
            policy["production_authorised"]
        ),
        "independent_validation_completed": bool(
            policy[
                "independent_validation_completed"
            ]
        ),
        "verified_artifacts": len(
            verified_artifacts
        ),
        "package_versions": package_versions,
        "a5_inference_replay_passed": True,
        "a5_whole_page_replay_tested": bool(
            operator.startup_replay.get(
                "whole_page_assembly_tested",
                False,
            )
        ),
        "v78_equivalence_passed": True,
    }

    return (
        policy,
        operator.startup_replay,
        v78,
        gate_record,
    )


# ------------------------------------------------------------
# RUN MANDATORY STARTUP GATES
# ------------------------------------------------------------

(
    POLICY,
    ARDROSS_REPLAY,
    ARDROSS_V78_EQUIVALENCE,
    STARTUP_GATE_RECORD,
) = gate()


# ------------------------------------------------------------
# WRITE COMMISSIONING EVIDENCE
# ------------------------------------------------------------

STARTUP_EVIDENCE = (
    ROOT
    / "startup_evidence"
)

STARTUP_VERDICT = write_commissioning_evidence(
    STARTUP_EVIDENCE,
    ARDROSS_REPLAY,
    ARDROSS_V78_EQUIVALENCE,
)

if not isinstance(
    STARTUP_VERDICT,
    dict,
):
    raise RuntimeError(
        "write_commissioning_evidence() must return "
        "a dictionary"
    )

if STARTUP_VERDICT.get("passed") is False:
    raise RuntimeError(
        "Commissioning evidence writer returned a "
        "failed verdict: "
        f"{json.dumps(STARTUP_VERDICT, default=str)}"
    )

print(
    "HF-04.4 commissioning startup verdict:",
    json.dumps(
        STARTUP_VERDICT,
        indent=2,
        default=str,
    ),
)


# ------------------------------------------------------------
# COMMAND-LINE / DOCKER BUILD REPORT
# ------------------------------------------------------------

if __name__ == "__main__":
    print(
        json.dumps(
            {
                "startup_gate_passed": True,

                # Correct authorisation language:
                # this is a frozen candidate for independent
                # testing, not a production-authorised release.
                "independent_test_candidate_ready": True,
                "production_authorised": bool(
                    POLICY[
                        "production_authorised"
                    ]
                ),
                "independent_validation_completed": bool(
                    POLICY[
                        "independent_validation_completed"
                    ]
                ),

                "version": "HF-04.4",
                "policy_id": POLICY_ID,
                "policy_status": POLICY["status"],
                "python": platform.python_version(),

                "startup_gate_record": (
                    STARTUP_GATE_RECORD
                ),
                "ardross_a5_replay": (
                    ARDROSS_REPLAY
                ),
                "ardross_v78_equivalence": (
                    ARDROSS_V78_EQUIVALENCE
                ),
                "startup_verdict": (
                    STARTUP_VERDICT
                ),
                "startup_evidence_directory": str(
                    STARTUP_EVIDENCE.resolve()
                ),

                # Explicit limitation until a genuine
                # frozen whole-page replay pack is added.
                "whole_page_a5_replay_tested": bool(
                    ARDROSS_REPLAY.get(
                        "whole_page_assembly_tested",
                        False,
                    )
                ),
            },
            indent=2,
            default=str,
        )
    )