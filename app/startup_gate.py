from pathlib import Path
import hashlib, json, platform, sys
import numpy as np
import pandas as pd
import sklearn
import joblib

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "frozen" / "structural_policy_v7_7f" / "artefacts"
REPORT = ROOT / "startup_gate_report.json"
REQUIRED = [
    "policy.json", "predictor_schema.json", "structural_model.joblib",
    "isotonic_calibrator.joblib", "replay_test.json",
]
EXPECTED_RUNTIME = {
    "numpy": "2.1.3", "pandas": "2.2.3",
    "scikit_learn": "1.6.1", "joblib": "1.6.0",
}
REQUIRED_REPLAY_CHECKS = [
    "calibrator_hash", "decisions_replay", "model_hash",
    "predictor_order", "probabilities_replay", "raw_scores_replay",
    "schema_hash",
]
LEGACY_ALIASES = {
    "component_pixels_x": "component_pixels",
    "component_pixels_y": "component_pixels",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def write_and_fail(report, message):
    report.update(status="FAIL", error=message)
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    raise RuntimeError(message)

def software_value(software, *names):
    lowered = {str(k).lower().replace("-", "_"): str(v) for k, v in software.items()}
    for name in names:
        key = name.lower().replace("-", "_")
        if key in lowered:
            return lowered[key]
    return None

def schema_valid_smoke_row(predictors):
    # Finite, deliberately ordinary values. This is a deserialisation/execution
    # smoke test, not a frozen replay and not a policy-performance assertion.
    defaults = {
        "w": 12.0, "h": 8.0, "area": 64.0, "aspect_ratio": 1.5,
        "fill_ratio": 0.67, "touches_page_border": 0.0,
        "context_pixels": 1200.0, "context_edge_density": 0.08,
        "context_dark_density": 0.05, "context_mean_intensity": 235.0,
        "context_intensity_std": 18.0, "maximum_mask_depth": 4.0,
        "horizontal_fraction": 0.10, "vertical_fraction": 0.05,
        "compact_fraction": 0.04, "preserve_fraction": 0.10,
        "remove_fraction": 0.75, "ambiguous_fraction": 0.15,
        "reconstruction_dependency_fraction": 0.15,
        "component_pixels": 64.0, "component_pixels_x": 64.0,
        "component_pixels_y": 64.0,
    }
    row = {}
    for predictor in predictors:
        if predictor in defaults:
            row[predictor] = defaults[predictor]
        elif predictor in LEGACY_ALIASES:
            row[predictor] = defaults[LEGACY_ALIASES[predictor]]
        else:
            row[predictor] = 0.0
    return pd.DataFrame([row], columns=predictors)

def main():
    report = {
        "gate": "V7.7F_STARTUP",
        "gate_version": "HF-03.2-corrected",
        "status": "RUNNING",
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
    }
    missing = [name for name in REQUIRED if not (POLICY_DIR / name).exists()]
    if missing:
        write_and_fail(report, f"Missing frozen V7.7F files: {missing}")

    policy = json.loads((POLICY_DIR / "policy.json").read_text())
    schema = json.loads((POLICY_DIR / "predictor_schema.json").read_text())
    replay = json.loads((POLICY_DIR / "replay_test.json").read_text())
    software = policy.get("software", {})

    actual = {
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
    }
    for package, expected in EXPECTED_RUNTIME.items():
        if actual[package] != expected:
            write_and_fail(report, f"Software mismatch {package}: installed={actual[package]} expected={expected}")
    if sys.version_info[:2] != (3, 13):
        write_and_fail(report, f"Python major.minor mismatch: installed={platform.python_version()} expected=3.13")

    recorded = {
        "python": software_value(software, "python"),
        "numpy": software_value(software, "numpy"),
        "pandas": software_value(software, "pandas"),
        "scikit_learn": software_value(software, "scikit_learn", "scikit-learn"),
        "joblib": software_value(software, "joblib"),
    }
    report["policy_software_record"] = recorded
    # Exact package pins are enforced above. The policy record is also checked
    # whenever that field is present, while Python is matched at major.minor.
    for package in ("numpy", "pandas", "scikit_learn", "joblib"):
        if recorded[package] and recorded[package] != EXPECTED_RUNTIME[package]:
            write_and_fail(report, f"policy.json software.{package}={recorded[package]} conflicts with expected {EXPECTED_RUNTIME[package]}")
    if recorded["python"] and not recorded["python"].startswith("3.13"):
        write_and_fail(report, f"policy.json Python record is incompatible: {recorded['python']}")

    identity = {
        "model_hash": policy.get("model_sha256") == sha256(POLICY_DIR / "structural_model.joblib"),
        "calibrator_hash": policy.get("calibrator_sha256") == sha256(POLICY_DIR / "isotonic_calibrator.joblib"),
        "schema_hash": policy.get("schema_sha256") == sha256(POLICY_DIR / "predictor_schema.json"),
        "policy_id_match": replay.get("policy_id") == policy.get("policy_id"),
    }
    report["identity_checks"] = identity
    if not all(identity.values()):
        write_and_fail(report, f"Frozen identity gate failed: {identity}")

    replay_checks = replay.get("checks", {})
    historical = {
        "passed": replay.get("passed") is True,
        "policy_id_match": replay.get("policy_id") == policy.get("policy_id"),
        "development_replay_hash_present": bool(replay.get("development_replay_hash")),
        **{name: replay_checks.get(name) is True for name in REQUIRED_REPLAY_CHECKS},
    }
    report["historical_freeze_replay_record"] = historical
    if not all(historical.values()):
        write_and_fail(report, f"Historical replay verification record failed: {historical}")

    predictors = list(schema.get("ordered_predictors", []))
    if not predictors or len(predictors) != len(set(predictors)):
        write_and_fail(report, "Frozen predictor schema is empty or contains duplicates.")

    model = joblib.load(POLICY_DIR / "structural_model.joblib")
    calibrator = joblib.load(POLICY_DIR / "isotonic_calibrator.joblib")
    X = schema_valid_smoke_row(predictors)
    if list(X.columns) != predictors or not np.isfinite(X.to_numpy(dtype=float)).all():
        write_and_fail(report, "Schema-valid smoke row construction failed.")
    raw = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    probability = np.asarray(calibrator.predict(raw), dtype=float)
    threshold = float(policy["threshold"])
    decision = probability >= threshold
    if raw.shape != (1,) or probability.shape != (1,) or not np.isfinite(raw).all() or not np.isfinite(probability).all():
        write_and_fail(report, "Runtime model/calibrator smoke execution returned invalid output.")

    report.update(
        status="PASS", policy_id=policy.get("policy_id"), threshold=threshold,
        historical_freeze_replay_verified=True,
        runtime_deserialisation_smoke_test={
            "passed": True, "predictor_count": len(predictors),
            "raw_score": float(raw[0]), "calibrated_probability": float(probability[0]),
            "threshold_decision": bool(decision[0]),
            "interpretation": "Runtime execution smoke test only; not a replay or performance assertion.",
        },
    )
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
