from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import platform
import re
from importlib import metadata
from typing import Any

from app.commissioning_evidence import (
    write_commissioning_evidence,
)


# ---------------------------------------------------------------------------
# Application and bundle identity
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

APP_VERSION = "4.5.1"
HF_VERSION = "HF-04.5.1"

A5_BUNDLE = (
    ROOT
    / "frozen"
    / "a5_v1"
)

STARTUP_EVIDENCE = (
    ROOT
    / "startup_evidence"
)

A1_POLICY_ID = (
    "NOTEBOOK87_A1_FROM_FROZEN_V8_9"
)

A5_POLICY_ID = (
    "A5-v1-print-p0980-margin0960-run2"
)

IGNORE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

REQUIRED_A5_BUNDLE_FILES = [
    "policy.json",
    "channel_schema.json",
    "model.pt",
    "model_config.json",
    "isotonic_calibrator.joblib",
    "inference_runtime.py",
    "artifact_manifest.csv",
    "freeze_manifest.json",
]

EXPECTED_A5_POLICY = {
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
    "policy_id": A5_POLICY_ID,
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

REQUIRED_EXACT_PACKAGES = {
    "scikit-learn": "1.6.1",
    "joblib": "1.6.0",
}

RECORDED_PACKAGES = [
    "numpy",
    "torch",
    "opencv-python-headless",
]

SHA256_PATTERN = re.compile(
    r"^[0-9a-fA-F]{64}$"
)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def generated(
    relative_path: str | Path,
) -> bool:
    path = Path(
        relative_path
    )

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


def write_json(
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
        )
    )

    temporary.replace(
        path
    )


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
            f"Unable to read {name} "
            f"from {path}: {exc}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"{name} must contain a JSON object"
        )

    return value


def assert_json_boolean(
    value: Any,
    field: str,
) -> bool:
    if not isinstance(
        value,
        bool,
    ):
        raise RuntimeError(
            f"{field} must be a JSON boolean; "
            f"found {value!r}"
        )

    return value


def assert_float_equal(
    field: str,
    actual: Any,
    expected: float,
) -> None:
    try:
        actual_number = float(
            actual
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "A5 policy field "
            f"{field!r} is not numeric: "
            f"{actual!r}"
        ) from exc

    if actual_number != float(
        expected
    ):
        raise RuntimeError(
            "A5 policy mismatch for "
            f"{field}: "
            f"{actual_number!r} "
            f"!= {expected!r}"
        )


# ---------------------------------------------------------------------------
# Frozen A5 policy and manifest validation
# ---------------------------------------------------------------------------

def validate_a5_policy(
    policy: dict,
) -> None:
    if not isinstance(
        policy,
        dict,
    ):
        raise RuntimeError(
            "Frozen A5-v1 policy must be a dictionary"
        )

    missing = sorted(
        set(
            EXPECTED_A5_POLICY
        )
        - set(
            policy
        )
    )

    if missing:
        raise RuntimeError(
            "Frozen A5-v1 policy is missing fields: "
            f"{missing}"
        )

    for key, expected in (
        EXPECTED_A5_POLICY.items()
    ):
        actual = policy[
            key
        ]

        if isinstance(
            expected,
            float,
        ):
            assert_float_equal(
                key,
                actual,
                expected,
            )

            continue

        if actual != expected:
            raise RuntimeError(
                "A5 policy mismatch for "
                f"{key}: "
                f"{actual!r} != {expected!r}"
            )

    print_threshold = float(
        policy[
            "print_threshold"
        ]
    )

    print_margin = float(
        policy[
            "print_margin"
        ]
    )

    annotation_threshold = float(
        policy[
            "annotation_threshold"
        ]
    )

    effective_threshold = max(
        print_threshold,
        (
            1.0
            + print_margin
        )
        / 2.0,
    )

    if effective_threshold != float(
        policy[
            "effective_print_threshold"
        ]
    ):
        raise RuntimeError(
            "A5 effective PRINT threshold is "
            "internally inconsistent: "
            f"calculated={effective_threshold!r}, "
            "recorded="
            f"{policy['effective_print_threshold']!r}"
        )

    for field, value in (
        (
            "annotation_threshold",
            annotation_threshold,
        ),
        (
            "print_threshold",
            print_threshold,
        ),
        (
            "print_margin",
            print_margin,
        ),
    ):
        if not (
            0.0
            <= value
            <= 1.0
        ):
            raise RuntimeError(
                f"{field} must be within 0..1"
            )

    if int(
        policy[
            "minimum_print_run"
        ]
    ) < 1:
        raise RuntimeError(
            "minimum_print_run must be at least 1"
        )

    if int(
        policy[
            "connectivity"
        ]
    ) not in {
        4,
        8,
    }:
        raise RuntimeError(
            "connectivity must be 4 or 8"
        )


def validate_freeze_manifest(
    freeze_manifest: dict,
    policy: dict,
) -> None:
    required = {
        "freeze_status",
        "production_authorised",
        "independent_validation_completed",
    }

    missing = sorted(
        required
        - set(
            freeze_manifest
        )
    )

    manifest_policy_id = (
        freeze_manifest.get(
            "policy_id"
        )
        or freeze_manifest.get(
            "a5_policy_id"
        )
    )

    if manifest_policy_id is None:
        missing.append(
            "policy_id or a5_policy_id"
        )

    if missing:
        raise RuntimeError(
            "freeze_manifest.json is missing "
            "required fields: "
            f"{sorted(set(missing))}"
        )

    if str(
        manifest_policy_id
    ) != A5_POLICY_ID:
        raise RuntimeError(
            "Freeze-manifest policy mismatch: "
            f"{manifest_policy_id!r} "
            f"!= {A5_POLICY_ID!r}"
        )

    if (
        str(
            freeze_manifest[
                "freeze_status"
            ]
        )
        != str(
            policy[
                "status"
            ]
        )
    ):
        raise RuntimeError(
            "Freeze-manifest status differs "
            "from policy: "
            f"{freeze_manifest['status']!r} "
            f"!= {policy['status']!r}"
        )

    manifest_production = assert_json_boolean(
        freeze_manifest[
            "production_authorised"
        ],
        (
            "freeze_manifest."
            "production_authorised"
        ),
    )

    policy_production = assert_json_boolean(
        policy[
            "production_authorised"
        ],
        (
            "policy.production_authorised"
        ),
    )

    if (
        manifest_production
        != policy_production
    ):
        raise RuntimeError(
            "Freeze-manifest production-authorisation "
            "value differs from policy"
        )

    manifest_validation = assert_json_boolean(
        freeze_manifest[
            "independent_validation_completed"
        ],
        (
            "freeze_manifest."
            "independent_validation_completed"
        ),
    )

    policy_validation = assert_json_boolean(
        policy[
            "independent_validation_completed"
        ],
        (
            "policy."
            "independent_validation_completed"
        ),
    )

    if (
        manifest_validation
        != policy_validation
    ):
        raise RuntimeError(
            "Freeze-manifest independent-validation "
            "value differs from policy"
        )


# ---------------------------------------------------------------------------
# Frozen bundle artifact-manifest validation
# ---------------------------------------------------------------------------

def normalised_relative_path(
    value: str,
    row_number: int,
) -> str:
    relative = Path(
        value
    )

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

    if not relative.parts:
        raise RuntimeError(
            "Artifact manifest contains an empty "
            f"path at row {row_number}"
        )

    return relative.as_posix()


def actual_a5_bundle_roster() -> set[str]:
    roster = set()

    for path in A5_BUNDLE.rglob(
        "*"
    ):
        if not path.is_file():
            continue

        relative = path.relative_to(
            A5_BUNDLE
        )

        if generated(
            relative
        ):
            continue

        # A manifest ordinarily cannot contain a stable hash
        # of itself because editing the stored hash changes
        # the file being hashed.
        if relative.as_posix() == (
            "artifact_manifest.csv"
        ):
            continue

        roster.add(
            relative.as_posix()
        )

    return roster


def validate_artifact_manifest(
    manifest_path: Path,
) -> list[dict]:
    if not manifest_path.is_file():
        raise RuntimeError(
            "Missing artifact manifest: "
            f"{manifest_path}"
        )

    verified: list[dict] = []
    listed_paths: set[str] = set()
    casefold_paths: dict[str, str] = {}

    with manifest_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                "artifact_manifest.csv has no header"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            relative_path = (
                row.get(
                    "relative_path"
                )
                or row.get(
                    "path"
                )
                or row.get(
                    "file"
                )
            )

            expected_hash = (
                row.get(
                    "sha256"
                )
                or row.get(
                    "SHA-256"
                )
            )

            relative_path = (
                str(
                    relative_path
                ).strip()
                if relative_path is not None
                else ""
            )

            expected_hash = (
                str(
                    expected_hash
                ).strip().lower()
                if expected_hash is not None
                else ""
            )

            if not relative_path:
                raise RuntimeError(
                    "artifact_manifest.csv row "
                    f"{row_number} has no relative path"
                )

            if generated(
                relative_path
            ):
                continue

            normalised = normalised_relative_path(
                relative_path,
                row_number,
            )

            if normalised == (
                "artifact_manifest.csv"
            ):
                raise RuntimeError(
                    "artifact_manifest.csv must not "
                    "attempt to hash itself"
                )

            if normalised in listed_paths:
                raise RuntimeError(
                    "Duplicate artifact-manifest path: "
                    f"{normalised}"
                )

            case_key = normalised.casefold()

            previous_case = casefold_paths.get(
                case_key
            )

            if (
                previous_case is not None
                and previous_case != normalised
            ):
                raise RuntimeError(
                    "Case-colliding artifact paths: "
                    f"{previous_case!r} and "
                    f"{normalised!r}"
                )

            listed_paths.add(
                normalised
            )

            casefold_paths[
                case_key
            ] = normalised

            if not SHA256_PATTERN.fullmatch(
                expected_hash
            ):
                raise RuntimeError(
                    "Artifact manifest contains an "
                    "invalid SHA-256 at row "
                    f"{row_number}: "
                    f"{expected_hash!r}"
                )

            artifact_path = (
                A5_BUNDLE
                / normalised
            )

            if not artifact_path.is_file():
                raise RuntimeError(
                    "A5 artefact is missing: "
                    f"{normalised}"
                )

            actual_hash = sha256_file(
                artifact_path
            )

            if (
                actual_hash.lower()
                != expected_hash
            ):
                raise RuntimeError(
                    "A5 artefact hash failure: "
                    f"{normalised}: "
                    f"{actual_hash} "
                    f"!= {expected_hash}"
                )

            verified.append(
                {
                    "relative_path": normalised,
                    "sha256": actual_hash,
                    "bytes": (
                        artifact_path
                        .stat()
                        .st_size
                    ),
                }
            )

    if not verified:
        raise RuntimeError(
            "artifact_manifest.csv contains no "
            "verifiable frozen artefacts"
        )

    actual_paths = actual_a5_bundle_roster()

    missing_from_manifest = sorted(
        actual_paths
        - listed_paths
    )

    absent_from_bundle = sorted(
        listed_paths
        - actual_paths
    )

    if missing_from_manifest:
        raise RuntimeError(
            "Frozen A5 bundle contains unmanifested "
            "files: "
            f"{missing_from_manifest}"
        )

    if absent_from_bundle:
        raise RuntimeError(
            "Artifact manifest lists files absent "
            "from the frozen A5 bundle: "
            f"{absent_from_bundle}"
        )

    required_manifested = (
        set(
            REQUIRED_A5_BUNDLE_FILES
        )
        - {
            "artifact_manifest.csv",
        }
    )

    missing_required = sorted(
        required_manifested
        - listed_paths
    )

    if missing_required:
        raise RuntimeError(
            "Mandatory A5 bundle files are not "
            "content-addressed in the artifact "
            "manifest: "
            f"{missing_required}"
        )

    return verified


# ---------------------------------------------------------------------------
# Software identity
# ---------------------------------------------------------------------------

def package_version(
    package: str,
) -> str:
    try:
        return metadata.version(
            package
        )

    except metadata.PackageNotFoundError:
        # Some OpenCV installations expose the module while
        # package metadata uses a different distribution name.
        if package == (
            "opencv-python-headless"
        ):
            for alternative in (
                "opencv-python",
                "opencv-contrib-python-headless",
                "opencv-contrib-python",
            ):
                try:
                    return metadata.version(
                        alternative
                    )

                except metadata.PackageNotFoundError:
                    continue

        raise RuntimeError(
            "Required package is not installed: "
            f"{package}"
        )


def validate_package_versions() -> dict:
    exact_versions: dict[str, str] = {}
    recorded_versions: dict[str, str] = {}

    for package, expected in (
        REQUIRED_EXACT_PACKAGES.items()
    ):
        actual = package_version(
            package
        )

        exact_versions[
            package
        ] = actual

        if actual != expected:
            raise RuntimeError(
                f"{package} must be {expected}; "
                f"found {actual}"
            )

    for package in RECORDED_PACKAGES:
        recorded_versions[
            package
        ] = package_version(
            package
        )

    return {
        "exact": exact_versions,
        "recorded": recorded_versions,
    }


# ---------------------------------------------------------------------------
# Replay validators
# ---------------------------------------------------------------------------

def validate_a5_replay(
    replay: Any,
) -> None:
    """
    Validate the frozen A5 inference replay.

    The current Ardross fixture proves tensor construction,
    model inference, isotonic calibration and decisions over
    the frozen 5,000-coordinate replay population.

    Whole-page A1-to-A5 assembly is validated only when the
    replay explicitly declares that a genuine A1 page,
    expected A5 page, expected PRINT mask and complete eligible
    population were available.
    """

    if not isinstance(
        replay,
        dict,
    ):
        raise RuntimeError(
            "A5 startup replay must return "
            "a dictionary"
        )

    if replay.get(
        "passed"
    ) is not True:
        raise RuntimeError(
            "A5 startup replay did not pass: "
            f"{json.dumps(replay, default=str)}"
        )

    replay_policy_id = replay.get(
        "policy_id"
    )

    if str(
        replay_policy_id
    ) != A5_POLICY_ID:
        raise RuntimeError(
            "A5 startup replay policy mismatch: "
            f"{replay_policy_id!r} "
            f"!= {A5_POLICY_ID!r}"
        )

    required_inference_checks = [
        "tensor_replay_tested",
        "model_replay_tested",
        "calibration_replay_tested",
        "decision_replay_tested",
    ]

    for field in required_inference_checks:
        if replay.get(
            field
        ) is not True:
            raise RuntimeError(
                "A5 startup replay did not confirm "
                f"{field}"
            )

    whole_page_tested = replay.get(
        "whole_page_assembly_tested"
    )

    if not isinstance(
        whole_page_tested,
        bool,
    ):
        raise RuntimeError(
            "A5 startup replay must explicitly declare "
            "whole_page_assembly_tested as a boolean"
        )

    if whole_page_tested:
        required_fixture_flags = [
            "a1_fixture_available",
            "expected_a5_fixture_available",
            "expected_print_mask_fixture_available",
            "complete_eligible_page_population",
        ]

        for field in required_fixture_flags:
            if replay.get(
                field
            ) is not True:
                raise RuntimeError(
                    "A5 whole-page replay claimed assembly "
                    "testing without confirming "
                    f"{field}"
                )

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

        for field in assembly_checks:
            if replay.get(
                field
            ) is not True:
                raise RuntimeError(
                    "A5 whole-page replay did not confirm "
                    f"{field}"
                )

    else:
        # A coordinate-only replay must not fabricate positive
        # whole-page assembly assertions.

        unavailable_assembly_checks = [
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

        for field in unavailable_assembly_checks:
            if replay.get(
                field
            ) is not None:
                raise RuntimeError(
                    "A5 replay declared whole-page assembly "
                    "untested but returned a non-null value "
                    f"for {field}: {replay.get(field)!r}"
                )

        if replay.get(
            "a1_fixture_available"
        ) is not False:
            raise RuntimeError(
                "Coordinate-only A5 replay must declare "
                "a1_fixture_available=False"
            )

        if replay.get(
            "expected_a5_fixture_available"
        ) is not False:
            raise RuntimeError(
                "Coordinate-only A5 replay must declare "
                "expected_a5_fixture_available=False"
            )

        if replay.get(
            "expected_print_mask_fixture_available"
        ) is not False:
            raise RuntimeError(
                "Coordinate-only A5 replay must declare "
                "expected_print_mask_fixture_available=False"
            )

        if replay.get(
            "complete_eligible_page_population"
        ) is not False:
            raise RuntimeError(
                "Coordinate-only A5 replay must declare "
                "complete_eligible_page_population=False"
            )

        limitation = replay.get(
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
                "Coordinate-only A5 replay must record "
                "its whole-page assembly limitation"
            )

    zero_checks = [
        "outside_mask_drift",
        "outside_authoritative_mask_drift",
        "non_print_baseline_regression",
        "non_print_v78_regression",
        "print_not_equal_observed",
        "changed_byte_not_equal_observed",
        "unexpected_changed_byte",
    ]

    for field in zero_checks:
        value = replay.get(
            field
        )

        # None means the assembly diagnostic was not tested.
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
                    "A5 startup replay returned a "
                    "non-integral diagnostic for "
                    f"{field}: {value!r}"
                )

        else:
            raise RuntimeError(
                "A5 startup replay returned a "
                "non-numeric diagnostic for "
                f"{field}: {value!r}"
            )

        if numeric_value != 0:
            raise RuntimeError(
                "A5 startup replay failed "
                f"{field}: {numeric_value}"
            )


def validate_a1_replay(
    replay: Any,
) -> None:
    if not isinstance(
        replay,
        dict,
    ):
        raise RuntimeError(
            "V8.9/A1 startup replay must return "
            "a dictionary"
        )

    if replay.get(
        "passed"
    ) is not True:
        raise RuntimeError(
            "V8.9/A1 startup replay did not pass"
        )

    if replay.get(
        "a1_policy_id"
    ) != A1_POLICY_ID:
        raise RuntimeError(
            "V8.9/A1 replay policy mismatch: "
            f"{replay.get('a1_policy_id')!r}"
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
        if replay.get(
            field
        ) is not True:
            raise RuntimeError(
                "V8.9/A1 startup replay did not "
                f"confirm {field}"
            )

    required_identities = [
        "v89_operator_sha256",
        "p_print_bundle_id",
        "remove_bundle_id",
    ]

    missing_identities = [
        field
        for field in required_identities
        if not replay.get(
            field
        )
    ]

    if missing_identities:
        raise RuntimeError(
            "V8.9/A1 startup replay omitted "
            "required identities: "
            f"{missing_identities}"
        )


def validate_v78_equivalence(
    result: Any,
) -> None:
    if not isinstance(
        result,
        dict,
    ):
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
        result.get(
            "passed"
        ) is True
        or verdict == "PASS"
    )

    if not passed:
        raise RuntimeError(
            "Mandatory V7.8 equivalence gate failed: "
            f"{json.dumps(result, default=str)}"
        )



# ---------------------------------------------------------------------------
# Mandatory startup gate
# ---------------------------------------------------------------------------

def gate():
    missing = [
        name
        for name in REQUIRED_A5_BUNDLE_FILES
        if not (
            A5_BUNDLE
            / name
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            "A5 bundle incomplete: "
            + ", ".join(
                missing
            )
        )

    policy = load_json_object(
        A5_BUNDLE
        / "policy.json",
        "A5 policy",
    )

    validate_a5_policy(
        policy
    )

    freeze_manifest = load_json_object(
        A5_BUNDLE
        / "freeze_manifest.json",
        "A5 freeze manifest",
    )

    validate_freeze_manifest(
        freeze_manifest,
        policy,
    )

    verified_artifacts = (
        validate_artifact_manifest(
            A5_BUNDLE
            / "artifact_manifest.csv"
        )
    )

    package_versions = (
        validate_package_versions()
    )

    # ------------------------------------------------------------
    # Construct and verify Notebook 87 A1.
    #
    # app.hf045_a1_startup must expose:
    #
    #     run_v89_a1_replay(
    #         operator=A1Operator,
    #     ) -> dict
    #
    # The returned dictionary must satisfy validate_a1_replay().
    # ------------------------------------------------------------

    from app.a1_operator import (
        A1Operator,
    )

    from app.hf045_a1_startup import (
        run_v89_a1_replay,
    )

    a1_operator = A1Operator()

    a1_replay = run_v89_a1_replay(
        operator=a1_operator,
    )

    validate_a1_replay(
        a1_replay
    )

    # ------------------------------------------------------------
    # Construct and verify A5.
    #
    # The current frozen Ardross replay proves tensor
    # construction, model inference, isotonic calibration and
    # final decisions over its frozen 5,000-coordinate replay
    # population.
    #
    # Whole-page A1-to-A5 replay is accepted only when the replay
    # pack explicitly contains and tests the required whole-page
    # fixtures. The operational A1 fallback and whole-page byte
    # contracts remain enforced by A5Operator.run_from_a1().
    # ------------------------------------------------------------

    from app.a5_operator import (
        A5Operator,
    )

    a5_operator = A5Operator(
        run_startup_replay=True,
    )

    validate_a5_replay(
        a5_operator.startup_replay
    )

    # ------------------------------------------------------------
    # Verify the frozen V7.8 engine.
    # ------------------------------------------------------------

    from app.hf044_v78_startup import (
        run_ardross_v78_equivalence,
    )

    v78 = run_ardross_v78_equivalence()

    validate_v78_equivalence(
        v78
    )

    production_authorised = (
        assert_json_boolean(
            policy[
                "production_authorised"
            ],
            (
                "policy."
                "production_authorised"
            ),
        )
    )

    independent_validation_completed = (
        assert_json_boolean(
            policy[
                "independent_validation_completed"
            ],
            (
                "policy."
                "independent_validation_completed"
            ),
        )
    )

    gate_record = {
        "integrity_gate_passed": True,
        
        "runtime_replay_passed": True,

        "a5_tensor_replay_passed": bool(
            a5_operator.startup_replay.get(
                "tensor_replay_tested",
                False,
            )
        ),

        "a5_model_replay_passed": bool(
            a5_operator.startup_replay.get(
                "model_replay_tested",
                False,
            )
        ),

        "a5_calibration_replay_passed": bool(
            a5_operator.startup_replay.get(
                "calibration_replay_tested",
                False,
            )
        ),

        "a5_decision_replay_passed": bool(
            a5_operator.startup_replay.get(
                "decision_replay_tested",
                False,
            )
        ),
    
        "candidate_bundle_loadable": True,

        "app_version": APP_VERSION,
        "hf_version": HF_VERSION,

        "policy_id": A5_POLICY_ID,
        "policy_status": policy[
            "status"
        ],

        "production_authorised": (
            production_authorised
        ),

        "independent_validation_completed": (
            independent_validation_completed
        ),

        "verified_artifacts": len(
            verified_artifacts
        ),

        "package_versions": package_versions,

        "v78_equivalence_passed": True,

        "v89_a1_replay_passed": True,
        "a1_policy_id": A1_POLICY_ID,

        "v89_operator_sha256": (
            a1_replay.get(
                "v89_operator_sha256"
            )
        ),

        "v89_p_print_bundle_id": (
            a1_replay.get(
                "p_print_bundle_id"
            )
        ),

        "v89_remove_bundle_id": (
            a1_replay.get(
                "remove_bundle_id"
            )
        ),

        "a5_inference_replay_passed": True,

        "a5_whole_page_replay_tested": bool(
            a5_operator.startup_replay.get(
                "whole_page_assembly_tested",
                False,
            )
        ),

        "a5_whole_page_replay_limitation": (
            a5_operator.startup_replay.get(
                "whole_page_assembly_limitation"
            )
        ),

        "a5_policy_id": A5_POLICY_ID,

        "authorisation_interpretation": (
            "Integrity and replay gates passed. "
            "This does not override the frozen "
            "production_authorised or "
            "independent_validation_completed fields."
        ),
    }

    return (
        policy,
        a1_operator,
        a1_replay,
        a5_operator,
        a5_operator.startup_replay,
        v78,
        gate_record,
        package_versions,
        verified_artifacts,
    )


# ---------------------------------------------------------------------------
# Run mandatory startup gates
# ---------------------------------------------------------------------------

(
    POLICY,
    A1_OPERATOR,
    V89_A1_REPLAY,
    A5_OPERATOR,
    ARDROSS_A5_REPLAY,
    ARDROSS_V78_EQUIVALENCE,
    STARTUP_GATE_RECORD,
    PACKAGE_VERSIONS,
    VERIFIED_A5_ARTIFACTS,
) = gate()


# ---------------------------------------------------------------------------
# Write commissioning evidence
# ---------------------------------------------------------------------------

STARTUP_EVIDENCE.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------------------------
# Write HF-04.5.1 commissioning evidence
#
# The commissioning writer now owns the complete evidence pack:
#
#   V7.8 equivalence
#       -> frozen V8.9 / Notebook 87 A1 replay
#       -> A5 frozen inference replay
#
# Whole-page A5 assembly is included only where the frozen replay
# pack actually contains the necessary A1, expected-A5 and PRINT-mask
# fixtures. A technical commissioning PASS proves only the checks
# that were genuinely performed.
# ---------------------------------------------------------------------------

STARTUP_VERDICT = write_commissioning_evidence(
    STARTUP_EVIDENCE,
    ARDROSS_A5_REPLAY,
    ARDROSS_V78_EQUIVALENCE,
    V89_A1_REPLAY,
    production_authorised=POLICY[
        "production_authorised"
    ],
    independent_validation_completed=POLICY[
        "independent_validation_completed"
    ],
    policy_status=str(
        POLICY["status"]
    ),
)

if not isinstance(
    STARTUP_VERDICT,
    dict,
):
    raise RuntimeError(
        "write_commissioning_evidence() must "
        "return a dictionary"
    )

if STARTUP_VERDICT.get(
    "passed"
) is not True:
    raise RuntimeError(
        "HF-04.5.1 commissioning evidence "
        "did not return a passing integrity verdict: "
        f"{json.dumps(STARTUP_VERDICT, default=str)}"
    )

if STARTUP_VERDICT.get(
    "integrity_gate_passed"
) is not True:
    raise RuntimeError(
        "HF-04.5.1 commissioning evidence "
        "did not confirm the integrity gate"
    )

if STARTUP_VERDICT.get(
    "runtime_replay_passed"
) is not True:
    raise RuntimeError(
        "HF-04.5.1 commissioning evidence "
        "did not confirm runtime replay"
    )

print(
    f"{HF_VERSION} commissioning startup verdict:",
    json.dumps(
        STARTUP_VERDICT,
        indent=2,
        default=str,
    ),
)


# ---------------------------------------------------------------------------
# Command-line / Docker build report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        json.dumps(
            {
                "startup_integrity_gate_passed": True,
                "runtime_replay_passed": True,
                "candidate_bundle_loadable": True,

                "independent_test_candidate_ready": (
                    bool(
                        POLICY[
                            "status"
                        ]
                        == (
                            "FROZEN_CANDIDATE_"
                            "FOR_INDEPENDENT_TEST"
                        )
                    )
                ),

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

                "app_version": APP_VERSION,
                "hf_version": HF_VERSION,

                "a1_policy_id": A1_POLICY_ID,
                "a5_policy_id": A5_POLICY_ID,
                "policy_status": POLICY[
                    "status"
                ],

                "python": (
                    platform.python_version()
                ),

                "packages": PACKAGE_VERSIONS,

                "startup_gate_record": (
                    STARTUP_GATE_RECORD
                ),

                "v89_a1_replay": (
                    V89_A1_REPLAY
                ),

                "ardross_a5_replay": (
                    ARDROSS_A5_REPLAY
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

                "whole_page_a5_replay_tested": bool(
                    ARDROSS_A5_REPLAY.get(
                        "whole_page_assembly_tested",
                        False,
                    )
                ),

                "authorisation_warning": (
                    "Successful integrity and replay "
                    "gates do not constitute production "
                    "authorisation."
                ),
            },
            indent=2,
            default=str,
        )
    )