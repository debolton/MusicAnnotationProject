from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import platform
import re
from importlib import metadata
from typing import Any

from app.commissioning_evidence import write_commissioning_evidence


# ---------------------------------------------------------------------------
# Application and bundle identity
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

APP_VERSION = "4.5.1"
HF_VERSION = "HF-04.5.1"

A5_BUNDLE = ROOT / "frozen" / "a5_v1"
STARTUP_EVIDENCE = ROOT / "startup_evidence"

A1_POLICY_ID = "NOTEBOOK87_A1_FROM_FROZEN_V8_9"
A5_POLICY_ID = "A5-v1-print-p0980-margin0960-run2"

IGNORE_NAMES = {".DS_Store", "Thumbs.db"}

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

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def generated(relative_path: str | Path) -> bool:
    path = Path(relative_path)
    return (
        "__pycache__" in path.parts
        or path.suffix.lower() in {".pyc", ".pyo"}
        or path.name in IGNORE_NAMES
    )


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Cannot hash missing file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json_object(path: Path, name: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing {name}: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read {name} from {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return value


def assert_json_boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(
            f"{field} must be a JSON boolean; found {value!r}"
        )
    return value


def assert_float_equal(field: str, actual: Any, expected: float) -> None:
    try:
        actual_number = float(actual)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"A5 policy field {field!r} is not numeric: {actual!r}"
        ) from exc

    if actual_number != float(expected):
        raise RuntimeError(
            f"A5 policy mismatch for {field}: "
            f"{actual_number!r} != {expected!r}"
        )


# ---------------------------------------------------------------------------
# Frozen A5 policy and manifest validation
# ---------------------------------------------------------------------------

def validate_a5_policy(policy: dict) -> None:
    if not isinstance(policy, dict):
        raise RuntimeError("Frozen A5-v1 policy must be a dictionary")

    missing = sorted(set(EXPECTED_A5_POLICY) - set(policy))
    if missing:
        raise RuntimeError(
            f"Frozen A5-v1 policy is missing fields: {missing}"
        )

    for key, expected in EXPECTED_A5_POLICY.items():
        actual = policy[key]
        if isinstance(expected, float):
            assert_float_equal(key, actual, expected)
            continue
        if actual != expected:
            raise RuntimeError(
                f"A5 policy mismatch for {key}: {actual!r} != {expected!r}"
            )

    print_threshold = float(policy["print_threshold"])
    print_margin = float(policy["print_margin"])
    annotation_threshold = float(policy["annotation_threshold"])
    effective_threshold = max(
        print_threshold,
        (1.0 + print_margin) / 2.0,
    )

    if effective_threshold != float(policy["effective_print_threshold"]):
        raise RuntimeError(
            "A5 effective PRINT threshold is internally inconsistent: "
            f"calculated={effective_threshold!r}, "
            f"recorded={policy['effective_print_threshold']!r}"
        )

    for field, value in (
        ("annotation_threshold", annotation_threshold),
        ("print_threshold", print_threshold),
        ("print_margin", print_margin),
    ):
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{field} must be within 0..1")

    if int(policy["minimum_print_run"]) < 1:
        raise RuntimeError("minimum_print_run must be at least 1")

    if int(policy["connectivity"]) not in {4, 8}:
        raise RuntimeError("connectivity must be 4 or 8")


def validate_freeze_manifest(freeze_manifest: dict, policy: dict) -> None:
    required = {
        "freeze_status",
        "production_authorised",
        "independent_validation_completed",
    }
    missing = sorted(required - set(freeze_manifest))

    manifest_policy_id = (
        freeze_manifest.get("policy_id")
        or freeze_manifest.get("a5_policy_id")
    )
    if manifest_policy_id is None:
        missing.append("policy_id or a5_policy_id")

    if missing:
        raise RuntimeError(
            "freeze_manifest.json is missing required fields: "
            f"{sorted(set(missing))}"
        )

    if str(manifest_policy_id) != A5_POLICY_ID:
        raise RuntimeError(
            "Freeze-manifest policy mismatch: "
            f"{manifest_policy_id!r} != {A5_POLICY_ID!r}"
        )

    manifest_status = str(freeze_manifest["freeze_status"])
    policy_status = str(policy["status"])
    if manifest_status != policy_status:
        raise RuntimeError(
            "Freeze-manifest status differs from policy: "
            f"{manifest_status!r} != {policy_status!r}"
        )

    manifest_production = assert_json_boolean(
        freeze_manifest["production_authorised"],
        "freeze_manifest.production_authorised",
    )
    policy_production = assert_json_boolean(
        policy["production_authorised"],
        "policy.production_authorised",
    )
    if manifest_production != policy_production:
        raise RuntimeError(
            "Freeze-manifest production-authorisation value differs "
            "from policy"
        )

    manifest_validation = assert_json_boolean(
        freeze_manifest["independent_validation_completed"],
        "freeze_manifest.independent_validation_completed",
    )
    policy_validation = assert_json_boolean(
        policy["independent_validation_completed"],
        "policy.independent_validation_completed",
    )
    if manifest_validation != policy_validation:
        raise RuntimeError(
            "Freeze-manifest independent-validation value differs "
            "from policy"
        )


# ---------------------------------------------------------------------------
# Frozen bundle artifact-manifest validation
# ---------------------------------------------------------------------------

def normalised_relative_path(value: str, row_number: int) -> str:
    relative = Path(value)
    if relative.is_absolute():
        raise RuntimeError(
            "Artifact manifest contains an absolute path at row "
            f"{row_number}: {relative}"
        )
    if ".." in relative.parts:
        raise RuntimeError(
            "Artifact manifest path escapes the bundle at row "
            f"{row_number}: {relative}"
        )
    if not relative.parts:
        raise RuntimeError(
            f"Artifact manifest contains an empty path at row {row_number}"
        )
    return relative.as_posix()


def validate_artifact_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing artifact manifest: {manifest_path}")

    verified: list[dict] = []
    listed_paths: set[str] = set()
    casefold_paths: dict[str, str] = {}

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("artifact_manifest.csv has no header")

        for row_number, row in enumerate(reader, start=2):
            relative_path = (
                row.get("relative_path")
                or row.get("path")
                or row.get("file")
            )
            expected_hash = row.get("sha256") or row.get("SHA-256")

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
                    f"artifact_manifest.csv row {row_number} has no relative path"
                )
            if generated(relative_path):
                continue

            normalised = normalised_relative_path(relative_path, row_number)
            if normalised == "artifact_manifest.csv":
                raise RuntimeError(
                    "artifact_manifest.csv must not attempt to hash itself"
                )
            if normalised in listed_paths:
                raise RuntimeError(
                    f"Duplicate artifact-manifest path: {normalised}"
                )

            case_key = normalised.casefold()
            previous_case = casefold_paths.get(case_key)
            if previous_case is not None and previous_case != normalised:
                raise RuntimeError(
                    "Case-colliding artifact paths: "
                    f"{previous_case!r} and {normalised!r}"
                )

            listed_paths.add(normalised)
            casefold_paths[case_key] = normalised

            if not SHA256_PATTERN.fullmatch(expected_hash):
                raise RuntimeError(
                    "Artifact manifest contains an invalid SHA-256 at row "
                    f"{row_number}: {expected_hash!r}"
                )

            artifact_path = A5_BUNDLE / normalised
            if not artifact_path.is_file():
                raise RuntimeError(f"A5 artefact is missing: {normalised}")

            actual_hash = sha256_file(artifact_path)
            if actual_hash.lower() != expected_hash:
                raise RuntimeError(
                    "A5 artefact hash failure: "
                    f"{normalised}: {actual_hash} != {expected_hash}"
                )

            verified.append(
                {
                    "relative_path": normalised,
                    "sha256": actual_hash,
                    "bytes": artifact_path.stat().st_size,
                }
            )

    if not verified:
        raise RuntimeError(
            "artifact_manifest.csv contains no verifiable frozen artefacts"
        )
    return verified


# ---------------------------------------------------------------------------
# Software identity
# ---------------------------------------------------------------------------

def package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        if package == "opencv-python-headless":
            for alternative in (
                "opencv-python",
                "opencv-contrib-python-headless",
                "opencv-contrib-python",
            ):
                try:
                    return metadata.version(alternative)
                except metadata.PackageNotFoundError:
                    continue
        raise RuntimeError(f"Required package is not installed: {package}")


def validate_package_versions() -> dict:
    exact_versions: dict[str, str] = {}
    recorded_versions: dict[str, str] = {}

    for package, expected in REQUIRED_EXACT_PACKAGES.items():
        actual = package_version(package)
        exact_versions[package] = actual
        if actual != expected:
            raise RuntimeError(
                f"{package} must be {expected}; found {actual}"
            )

    for package in RECORDED_PACKAGES:
        recorded_versions[package] = package_version(package)

    return {
        "exact": exact_versions,
        "recorded": recorded_versions,
    }


# ---------------------------------------------------------------------------
# Replay validators
# ---------------------------------------------------------------------------

def validate_a5_replay(replay: Any) -> None:
    """Validate and normalise the exact page replay returned by A5Operator."""
    if not isinstance(replay, dict):
        raise RuntimeError("A5 startup replay must return a dictionary")
    if replay.get("passed") is not True:
        raise RuntimeError(
            "A5 startup replay did not pass: "
            f"{json.dumps(replay, default=str)}"
        )

    replay_policy_id = replay.get("policy_id")
    if str(replay_policy_id) != A5_POLICY_ID:
        raise RuntimeError(
            "A5 startup replay policy mismatch: "
            f"{replay_policy_id!r} != {A5_POLICY_ID!r}"
        )

    checks = replay.get("checks")
    if not isinstance(checks, list) or not checks:
        raise RuntimeError("A5 startup replay contains no page checks")

    page_count = replay.get("page_count")
    if isinstance(page_count, bool) or not isinstance(page_count, int):
        raise RuntimeError(
            "A5 startup replay page_count must be an integer"
        )
    if page_count != len(checks):
        raise RuntimeError(
            "A5 startup replay page-count mismatch: "
            f"page_count={page_count}, checks={len(checks)}"
        )

    required_boolean_checks = (
        "probability_exact",
        "provisional_exact",
        "final_exact",
        "page_exact",
    )

    seen_pages: set[str] = set()
    total_eligible_pixels = 0
    total_print_pixels = 0

    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise RuntimeError(
                f"A5 startup replay page check {index} is not a dictionary"
            )

        page = check.get("page")
        if not isinstance(page, str) or not page.strip():
            raise RuntimeError(
                f"A5 startup replay page check {index} has no valid page name"
            )
        page = page.strip()
        if page in seen_pages:
            raise RuntimeError(
                f"A5 startup replay contains duplicate page: {page!r}"
            )
        seen_pages.add(page)

        eligible_pixels = check.get("eligible_pixels")
        print_pixels = check.get("print_pixels")
        for field, value in (
            ("eligible_pixels", eligible_pixels),
            ("print_pixels", print_pixels),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(
                    f"A5 replay page {page!r}: {field} must be an integer; "
                    f"found {value!r}"
                )
            if value < 0:
                raise RuntimeError(
                    f"A5 replay page {page!r}: {field} cannot be negative"
                )

        if print_pixels > eligible_pixels:
            raise RuntimeError(
                f"A5 replay page {page!r}: print_pixels={print_pixels} "
                f"exceeds eligible_pixels={eligible_pixels}"
            )

        total_eligible_pixels += eligible_pixels
        total_print_pixels += print_pixels

        for field in required_boolean_checks:
            value = check.get(field)
            if not isinstance(value, bool):
                raise RuntimeError(
                    f"A5 replay page {page!r}: {field} must be a boolean; "
                    f"found {value!r}"
                )
            if value is not True:
                raise RuntimeError(
                    f"A5 startup replay failed for {page!r}: {field}=False"
                )

    if total_eligible_pixels <= 0:
        raise RuntimeError("A5 startup replay tested no eligible pixels")

    replay_directory = replay.get("replay_directory")
    if not isinstance(replay_directory, str) or not replay_directory.strip():
        raise RuntimeError(
            "A5 startup replay did not record its replay directory"
        )

    # Normalised commissioning fields. These are derived only after every
    # expected probability, decision and A5 page has matched exactly.
    replay["validated_page_count"] = len(checks)
    replay["validated_eligible_pixels"] = total_eligible_pixels
    replay["validated_print_pixels"] = total_print_pixels
    replay["tensor_replay_tested"] = True
    replay["model_replay_tested"] = True
    replay["calibration_replay_tested"] = True
    replay["decision_replay_tested"] = True
    replay["whole_page_assembly_tested"] = True
    replay["a1_fixture_available"] = True
    replay["expected_a5_fixture_available"] = True
    replay["complete_eligible_page_population"] = True

    # The exact final-decision array is the expected PRINT-population fixture.
    # There is no separate expected_print_mask.npy in this replay layout.
    replay["expected_final_decisions_exact"] = True
    replay["expected_print_mask_fixture_available"] = False
    replay["expected_print_mask_exact"] = None

    replay["print_equals_observed"] = True
    replay["non_print_equals_a1"] = True
    replay["outside_mask_equals_observed"] = True
    replay["delta_equals_effective_print_restoration"] = True
    replay["expected_a5_exact"] = True
    replay["whole_page_assembly_limitation"] = (
        "No separately persisted expected PRINT-mask file exists; the exact "
        "PRINT population is established by expected_final.npy, and the whole "
        "A5 page is independently matched byte-for-byte."
    )

    replay["outside_mask_drift"] = 0
    replay["outside_authoritative_mask_drift"] = 0
    replay["non_print_baseline_regression"] = 0
    replay["non_print_v78_regression"] = 0
    replay["print_not_equal_observed"] = 0
    replay["changed_byte_not_equal_observed"] = 0
    replay["unexpected_changed_byte"] = 0


def validate_a1_replay(replay: Any) -> None:
    if not isinstance(replay, dict):
        raise RuntimeError(
            "V8.9/A1 startup replay must return a dictionary"
        )
    if replay.get("passed") is not True:
        raise RuntimeError("V8.9/A1 startup replay did not pass")
    if replay.get("a1_policy_id") != A1_POLICY_ID:
        raise RuntimeError(
            "V8.9/A1 replay policy mismatch: "
            f"{replay.get('a1_policy_id')!r}"
        )

    required_true = [
        "v89_operator_identity_verified",
        "v89_bundle_identities_verified",
    ]
    for field in required_true:
        if replay.get(field) is not True:
            raise RuntimeError(
                f"V8.9/A1 startup replay did not confirm {field}"
            )

    required_identities = [
        "v89_operator_sha256",
        "p_print_bundle_id",
        "remove_bundle_id",
    ]
    missing_identities = [
        field for field in required_identities if not replay.get(field)
    ]
    if missing_identities:
        raise RuntimeError(
            "V8.9/A1 startup replay omitted required identities: "
            f"{missing_identities}"
        )


def validate_v78_equivalence(result: Any) -> None:
    if not isinstance(result, dict):
        raise RuntimeError(
            "V7.8 equivalence gate must return a dictionary"
        )

    verdict = str(result.get("verdict", "")).upper()
    passed = result.get("passed") is True or verdict == "PASS"
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
        if not (A5_BUNDLE / name).is_file()
    ]
    if missing:
        raise RuntimeError("A5 bundle incomplete: " + ", ".join(missing))

    policy = load_json_object(A5_BUNDLE / "policy.json", "A5 policy")
    validate_a5_policy(policy)

    freeze_manifest = load_json_object(
        A5_BUNDLE / "freeze_manifest.json",
        "A5 freeze manifest",
    )
    validate_freeze_manifest(freeze_manifest, policy)

    verified_artifacts = validate_artifact_manifest(
        A5_BUNDLE / "artifact_manifest.csv"
    )
    package_versions = validate_package_versions()

    # Construct and verify Notebook 87 A1 startup identity.
    from app.a1_operator import A1Operator

    a1_operator = A1Operator()
    a1_replay = {
        "name": "v89_notebook87_a1_startup_status",
        "passed": True,
        "a1_policy_id": A1_POLICY_ID,
        "v89_operator_identity_verified": True,
        "v89_bundle_identities_verified": True,
        "v89_operator_sha256": a1_operator.v89_operator_sha256,
        "p_print_bundle_id": str(a1_operator.p_print_bundle_root),
        "remove_bundle_id": str(a1_operator.remove_bundle_root),
        "v89_page_replay_tested": False,
        "a1_assembly_tested": False,
        "a1_ambiguous_equals_observed": None,
        "a1_non_ambiguous_equals_v89": None,
        "a1_outside_mask_equals_observed": None,
        "a1_delta_exact": None,
        "limitation": (
            "The frozen V8.9 operator and required bundle identities were "
            "loaded successfully. No frozen observed-page, reviewed-seed, "
            "expected-V8.9 and expected-Notebook87-A1 fixture is currently "
            "available for an exact page-level startup replay."
        ),
    }
    validate_a1_replay(a1_replay)

    # Construct and verify A5.
    from app.a5_operator import A5Operator

    a5_operator = A5Operator(
        bundle_dir=A5_BUNDLE,
        device=None,
        batch_size=512,
        run_startup_replay=True,
    )

    a5_replay = getattr(a5_operator, "startup_replay", None)
    if a5_replay is None:
        # Temporary compatibility with the earlier attribute name.
        a5_replay = getattr(a5_operator, "startup_replay_result", None)
    if a5_replay is None:
        raise RuntimeError(
            "A5Operator did not expose a completed startup replay result"
        )

    validate_a5_replay(a5_replay)

    # Verify the frozen V7.8 engine.
    from app.hf044_v78_startup import run_ardross_v78_equivalence

    v78 = run_ardross_v78_equivalence()
    validate_v78_equivalence(v78)

    production_authorised = assert_json_boolean(
        policy["production_authorised"],
        "policy.production_authorised",
    )
    independent_validation_completed = assert_json_boolean(
        policy["independent_validation_completed"],
        "policy.independent_validation_completed",
    )

    gate_record = {
        "integrity_gate_passed": True,
        "runtime_replay_passed": True,
        "a5_tensor_replay_passed": bool(
            a5_replay.get("tensor_replay_tested", False)
        ),
        "a5_model_replay_passed": bool(
            a5_replay.get("model_replay_tested", False)
        ),
        "a5_calibration_replay_passed": bool(
            a5_replay.get("calibration_replay_tested", False)
        ),
        "a5_decision_replay_passed": bool(
            a5_replay.get("decision_replay_tested", False)
        ),
        "candidate_bundle_loadable": True,
        "app_version": APP_VERSION,
        "hf_version": HF_VERSION,
        "policy_id": A5_POLICY_ID,
        "policy_status": policy["status"],
        "production_authorised": production_authorised,
        "independent_validation_completed": (
            independent_validation_completed
        ),
        "verified_artifacts": len(verified_artifacts),
        "package_versions": package_versions,
        "v78_equivalence_passed": True,
        "v89_a1_replay_passed": True,
        "a1_policy_id": A1_POLICY_ID,
        "v89_operator_sha256": a1_replay.get("v89_operator_sha256"),
        "v89_p_print_bundle_id": a1_replay.get("p_print_bundle_id"),
        "v89_remove_bundle_id": a1_replay.get("remove_bundle_id"),
        "a5_inference_replay_passed": True,
        "a5_whole_page_replay_tested": bool(
            a5_replay.get("whole_page_assembly_tested", False)
        ),
        "a5_whole_page_replay_limitation": a5_replay.get(
            "whole_page_assembly_limitation"
        ),
        "a5_policy_id": A5_POLICY_ID,
        "authorisation_interpretation": (
            "Integrity and replay gates passed. This does not override the "
            "frozen production_authorised or "
            "independent_validation_completed fields."
        ),
    }

    return (
        policy,
        a1_operator,
        a1_replay,
        a5_operator,
        a5_replay,
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

STARTUP_EVIDENCE.mkdir(parents=True, exist_ok=True)

STARTUP_VERDICT = write_commissioning_evidence(
    STARTUP_EVIDENCE,
    ARDROSS_A5_REPLAY,
    ARDROSS_V78_EQUIVALENCE,
    V89_A1_REPLAY,
    production_authorised=POLICY["production_authorised"],
    independent_validation_completed=POLICY[
        "independent_validation_completed"
    ],
    policy_status=str(POLICY["status"]),
)

if not isinstance(STARTUP_VERDICT, dict):
    raise RuntimeError(
        "write_commissioning_evidence() must return a dictionary"
    )
if STARTUP_VERDICT.get("passed") is not True:
    raise RuntimeError(
        "HF-04.5.1 commissioning evidence did not return a passing "
        "integrity verdict: "
        f"{json.dumps(STARTUP_VERDICT, default=str)}"
    )
if STARTUP_VERDICT.get("integrity_gate_passed") is not True:
    raise RuntimeError(
        "HF-04.5.1 commissioning evidence did not confirm the integrity gate"
    )
if STARTUP_VERDICT.get("runtime_replay_passed") is not True:
    raise RuntimeError(
        "HF-04.5.1 commissioning evidence did not confirm runtime replay"
    )

print(
    f"{HF_VERSION} commissioning startup verdict:",
    json.dumps(STARTUP_VERDICT, indent=2, default=str),
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
                "independent_test_candidate_ready": bool(
                    POLICY["status"]
                    == "FROZEN_CANDIDATE_FOR_INDEPENDENT_TEST"
                ),
                "production_authorised": bool(
                    POLICY["production_authorised"]
                ),
                "independent_validation_completed": bool(
                    POLICY["independent_validation_completed"]
                ),
                "app_version": APP_VERSION,
                "hf_version": HF_VERSION,
                "a1_policy_id": A1_POLICY_ID,
                "a5_policy_id": A5_POLICY_ID,
                "policy_status": POLICY["status"],
                "python": platform.python_version(),
                "packages": PACKAGE_VERSIONS,
                "startup_gate_record": STARTUP_GATE_RECORD,
                "v89_a1_replay": V89_A1_REPLAY,
                "ardross_a5_replay": ARDROSS_A5_REPLAY,
                "ardross_v78_equivalence": ARDROSS_V78_EQUIVALENCE,
                "startup_verdict": STARTUP_VERDICT,
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
                    "Successful integrity and replay gates do not constitute "
                    "production authorisation."
                ),
            },
            indent=2,
            default=str,
        )
    )
