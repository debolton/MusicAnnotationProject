"""Frozen V8.8 REMOVE-only page operator.

Adapter contract for Notebook 72. This module contains no clean/truth access.
It implements the frozen V8.8 policy:

    M5_REMOVE = M3 & ~SAFE_REMOVE

SAFE_REMOVE pixels receive the local paper estimate. Remaining M5_REMOVE pixels
are reconstructed with the configured frozen inpainting method. Pixels outside
M3 are immutable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
import hashlib
import json
import platform
import sys
import time

import cv2
import joblib
import numpy as np

OPERATOR_VERSION = "8.8-notebook72-adapter-1"
DECISION_M3_FALLBACK = np.uint8(0)
DECISION_SAFE_REMOVE = np.uint8(1)


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _binary(array: np.ndarray) -> np.ndarray:
    return np.asarray(array) > 0


@dataclass(frozen=True)
class FrozenArtifacts:
    policy_manifest_path: Path
    predictor_schema_path: Path
    scorer_path: Optional[Path] = None
    scorer_threshold: Optional[float] = None
    replay_decisions_path: Optional[Path] = None
    expected_policy_id: Optional[str] = None
    expected_hashes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeContext:
    artifacts: FrozenArtifacts
    page_id: str
    feature_table: Any = None
    registered_decisions: Any = None
    mask_is_prepared_m3: bool = True
    mask_dilation: int = 0
    reconstruction_method: str = "telea"
    inpaint_radius: int = 3
    crop_margin: int = 48
    minimum_component_area: int = 8
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PageResult:
    corrected: np.ndarray
    baseline: np.ndarray
    prepared_mask: np.ndarray
    decisions: np.ndarray
    probabilities: dict[str, np.ndarray]
    changed: dict[str, np.ndarray]
    audit: dict[str, Any]
    safe_remove: np.ndarray
    m5_remove: np.ndarray
    paper_estimate: np.ndarray


def verify_artifacts(artifacts: FrozenArtifacts) -> dict[str, Any]:
    policy_path = Path(artifacts.policy_manifest_path)
    schema_path = Path(artifacts.predictor_schema_path)
    _require(policy_path.is_file(), f"Missing policy manifest: {policy_path}")
    _require(schema_path.is_file(), f"Missing predictor schema: {schema_path}")

    policy = _json(policy_path)
    schema = _json(schema_path)
    predictors = schema.get("ordered_predictors", schema.get("predictors", schema.get("ordered_features", [])))
    _require(bool(predictors), "Frozen ordered predictor schema is empty.")
    _require(policy.get("policy") == "M5_REMOVE", "Frozen policy is not M5_REMOVE.")
    _require(policy.get("safe_protect_runtime_enabled") is False, "SAFE_PROTECT must remain disabled.")
    _require(policy.get("retraining_performed") is False, "Policy indicates retraining.")
    _require(policy.get("recalibration_performed") is False, "Policy indicates recalibration.")
    _require(policy.get("threshold_changed") is False, "Policy indicates threshold movement.")
    if artifacts.expected_policy_id is not None:
        _require(policy.get("policy_id") == artifacts.expected_policy_id, "Frozen policy ID mismatch.")

    paths = {"policy_manifest": policy_path, "predictor_schema": schema_path}
    if artifacts.scorer_path is not None:
        paths["scorer"] = Path(artifacts.scorer_path)
    if artifacts.replay_decisions_path is not None:
        paths["replay_decisions"] = Path(artifacts.replay_decisions_path)
    for name, path in paths.items():
        _require(path.is_file(), f"Missing frozen artefact {name}: {path}")
        expected = artifacts.expected_hashes.get(name)
        if expected:
            _require(_file_hash(path) == expected, f"Frozen artefact hash mismatch: {name}")

    return {
        "policy": policy,
        "schema": schema,
        "ordered_predictors": list(predictors),
        "hashes": {name: _file_hash(path) for name, path in paths.items()},
    }


def prepare_mask(seed: np.ndarray, dilation: int = 0) -> np.ndarray:
    mask = _binary(seed)
    _require(mask.ndim == 2, "Authoritative mask must be a 2-D array.")
    _require(dilation >= 0, "mask_dilation must be non-negative.")
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask


def _padded_components(mask: np.ndarray, margin: int, minimum_area: int) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    height, width = mask.shape
    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component]
        if int(area) >= minimum_area:
            boxes.append((max(0, x - margin), max(0, y - margin), min(width, x + box_width + margin), min(height, y + box_height + margin)))
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        while boxes:
            ax0, ay0, ax1, ay1 = boxes.pop()
            remaining = []
            for bx0, by0, bx1, by1 in boxes:
                if not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0):
                    ax0, ay0, ax1, ay1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
                    changed = True
                else:
                    remaining.append((bx0, by0, bx1, by1))
            boxes = remaining
            merged.append((ax0, ay0, ax1, ay1))
        boxes = merged
    return boxes


def _reconstruct_residual(observed: np.ndarray, residual: np.ndarray, method: str, radius: int, margin: int, minimum_area: int) -> np.ndarray:
    result = observed.copy()
    mask_u8 = residual.astype(np.uint8) * 255
    for x0, y0, x1, y1 in _padded_components(residual, margin, minimum_area):
        image_crop = observed[y0:y1, x0:x1]
        mask_crop = mask_u8[y0:y1, x0:x1]
        if method == "telea":
            candidate = cv2.inpaint(image_crop, mask_crop, radius, cv2.INPAINT_TELEA)
        elif method == "navier_stokes":
            candidate = cv2.inpaint(image_crop, mask_crop, radius, cv2.INPAINT_NS)
        else:
            raise ValueError(f"Unsupported frozen reconstruction method: {method}")
        local = result[y0:y1, x0:x1]
        local[mask_crop > 0] = candidate[mask_crop > 0]
        result[y0:y1, x0:x1] = local
    _require(np.array_equal(result[~residual], observed[~residual]), "Residual reconstruction changed non-residual pixels.")
    return result


def _background_surface(observed: np.ndarray, mask: np.ndarray, sigma: float = 45.0) -> np.ndarray:
    visible = (~mask) & (observed >= 200)
    values = observed.astype(np.float32) * visible
    weights = visible.astype(np.float32)
    numerator = cv2.GaussianBlur(values, (0, 0), sigma)
    denominator = cv2.GaussianBlur(weights, (0, 0), sigma)
    fallback = cv2.GaussianBlur(observed, (0, 0), sigma)
    return np.clip(np.where(denominator > 0.02, numerator / np.maximum(denominator, 1e-4), fallback), 0, 255).astype(np.uint8)


def _normalise_decision_table(table: Any, page_id: str) -> Any:
    import pandas as pd
    if isinstance(table, (str, Path)):
        path = Path(table)
        table = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    _require(isinstance(table, pd.DataFrame), "Decision/feature input must be a pandas DataFrame or CSV/Parquet path.")
    frame = table.copy()
    if "page" in frame.columns:
        frame = frame[frame.page.astype(str) == str(page_id)].copy()
    _require({"y", "x"}.issubset(frame.columns), "Decision/feature table requires y and x columns.")
    frame["y"] = pd.to_numeric(frame.y, errors="raise").astype(int)
    frame["x"] = pd.to_numeric(frame.x, errors="raise").astype(int)
    _require(not frame.duplicated(["y", "x"]).any(), "Duplicate decision coordinates.")
    return frame


def _resolve_decisions(context: RuntimeContext, verified: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import pandas as pd
    if context.registered_decisions is not None:
        frame = _normalise_decision_table(context.registered_decisions, context.page_id)
        _require("safe_remove" in frame.columns, "Registered decisions require safe_remove.")
        selected = frame.safe_remove.astype(bool).to_numpy()
        probability_column = next((name for name in ("p_remove", "remove_probability") if name in frame.columns), None)
        probabilities = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(float) if probability_column else np.full(len(frame), np.nan)
        mode = "registered_decisions"
    else:
        _require(context.feature_table is not None, "Runtime requires registered_decisions or a feature_table.")
        _require(context.artifacts.scorer_path is not None, "Feature mode requires a frozen deployable scorer.")
        _require(context.artifacts.scorer_threshold is not None, "Feature mode requires the frozen scorer threshold.")
        frame = _normalise_decision_table(context.feature_table, context.page_id)
        predictors = verified["ordered_predictors"]
        missing = [name for name in predictors if name not in frame.columns]
        _require(not missing, f"Feature table is missing frozen predictors: {missing}")
        matrix = frame.loc[:, predictors].apply(pd.to_numeric, errors="coerce")
        _require(not np.isinf(matrix.to_numpy(float)).any(), "Infinite predictor value.")
        scorer = joblib.load(context.artifacts.scorer_path)
        _require(hasattr(scorer, "predict_proba") or hasattr(scorer, "predict"), "Frozen scorer lacks prediction interface.")
        if hasattr(scorer, "predict_proba"):
            probabilities = np.asarray(scorer.predict_proba(matrix), dtype=float)[:, 1]
            selected = probabilities >= float(context.artifacts.scorer_threshold)
        else:
            selected = np.asarray(scorer.predict(matrix)).astype(bool)
            probabilities = np.full(len(selected), np.nan)
        mode = "deployable_scorer"

    height, width = shape
    ys, xs = frame.y.to_numpy(int), frame.x.to_numpy(int)
    _require((ys >= 0).all() and (xs >= 0).all() and (ys < height).all() and (xs < width).all(), "Decision coordinate outside page.")
    safe_remove = np.zeros(shape, bool)
    safe_remove[ys[selected], xs[selected]] = True
    probability_map = np.full(shape, np.nan, np.float32)
    probability_map[ys, xs] = probabilities.astype(np.float32)
    return safe_remove, probability_map, {"decision_mode": mode, "decision_rows": int(len(frame)), "selected_rows": int(selected.sum())}


def run_page(observed: np.ndarray, authoritative_seed_mask: np.ndarray, runtime_context: RuntimeContext | Mapping[str, Any]) -> PageResult:
    started = time.perf_counter()
    context = runtime_context if isinstance(runtime_context, RuntimeContext) else RuntimeContext(**runtime_context)
    verified = verify_artifacts(context.artifacts)

    observed = np.asarray(observed)
    seed = np.asarray(authoritative_seed_mask)
    _require(observed.ndim == 2, "Observed page must be single-channel.")
    _require(observed.dtype == np.uint8, "Observed page dtype must be uint8.")
    _require(seed.shape == observed.shape, "Observed page and authoritative seed mask shape mismatch.")
    _require(seed.ndim == 2, "Authoritative seed mask must be 2-D.")

    dilation = 0 if context.mask_is_prepared_m3 else context.mask_dilation
    prepared = prepare_mask(seed, dilation)
    safe_remove, p_remove, decision_audit = _resolve_decisions(context, verified, observed.shape)
    _require(not np.any(safe_remove & ~prepared), "SAFE_REMOVE is not a subset of authoritative M3.")

    m5_remove = prepared & ~safe_remove
    baseline = _reconstruct_residual(observed, prepared, context.reconstruction_method, context.inpaint_radius, context.crop_margin, context.minimum_component_area)
    residual = _reconstruct_residual(observed, m5_remove, context.reconstruction_method, context.inpaint_radius, context.crop_margin, context.minimum_component_area)
    paper = _background_surface(observed, prepared)
    corrected = residual.copy()
    corrected[safe_remove] = paper[safe_remove]

    decisions = np.full(observed.shape, DECISION_M3_FALLBACK, np.uint8)
    decisions[safe_remove] = DECISION_SAFE_REMOVE
    changed_from_observed = corrected != observed
    changed_from_baseline = corrected != baseline
    valid_intervention = prepared.copy()

    outside_drift = int(np.sum(corrected[~prepared] != observed[~prepared]))
    unjustified = int(np.sum(changed_from_observed & ~valid_intervention))
    _require(outside_drift == 0, "Corrected page changed outside authoritative mask.")
    _require(unjustified == 0, "Changed pixel lacks authorised intervention.")
    _require(corrected.shape == observed.shape and corrected.dtype == observed.dtype, "Corrected page shape/dtype contract failed.")

    audit = {
        "operator_version": OPERATOR_VERSION,
        "page_id": str(context.page_id),
        "policy_id": verified["policy"].get("policy_id"),
        "policy_definition": "M5_REMOVE = M3 & ~SAFE_REMOVE",
        "safe_protect_runtime_enabled": False,
        "clean_or_truth_input_used": False,
        "artifact_hashes": verified["hashes"],
        "observed_sha256": _array_hash(observed),
        "prepared_mask_sha256": _array_hash(prepared.astype(np.uint8)),
        "safe_remove_sha256": _array_hash(safe_remove.astype(np.uint8)),
        "m5_remove_sha256": _array_hash(m5_remove.astype(np.uint8)),
        "baseline_sha256": _array_hash(baseline),
        "corrected_sha256": _array_hash(corrected),
        "decision_sha256": _array_hash(decisions),
        "outside_mask_drift": outside_drift,
        "unjustified_changed_pixels": unjustified,
        "prepared_mask_pixels": int(prepared.sum()),
        "safe_remove_pixels": int(safe_remove.sum()),
        "m5_remove_pixels": int(m5_remove.sum()),
        "changed_from_observed_pixels": int(changed_from_observed.sum()),
        "changed_from_baseline_pixels": int(changed_from_baseline.sum()),
        "reconstruction_method": context.reconstruction_method,
        "inpaint_radius": int(context.inpaint_radius),
        "crop_margin": int(context.crop_margin),
        "mask_dilation_applied": int(dilation),
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        **decision_audit,
        **dict(context.policy_metadata),
    }
    return PageResult(
        corrected=corrected,
        baseline=baseline,
        prepared_mask=prepared,
        decisions=decisions,
        probabilities={"p_remove": p_remove},
        changed={"from_observed": changed_from_observed, "from_baseline": changed_from_baseline, "valid_intervention": valid_intervention},
        audit=audit,
        safe_remove=safe_remove,
        m5_remove=m5_remove,
        paper_estimate=paper,
    )


def replay_page(observed: np.ndarray, authoritative_seed_mask: np.ndarray, runtime_context: RuntimeContext | Mapping[str, Any], expected: Mapping[str, str]) -> dict[str, Any]:
    first = run_page(observed, authoritative_seed_mask, runtime_context)
    second = run_page(observed, authoritative_seed_mask, runtime_context)
    current = {
        "corrected_sha256": _array_hash(first.corrected),
        "baseline_sha256": _array_hash(first.baseline),
        "prepared_mask_sha256": _array_hash(first.prepared_mask.astype(np.uint8)),
        "decision_sha256": _array_hash(first.decisions),
        "safe_remove_sha256": _array_hash(first.safe_remove.astype(np.uint8)),
        "m5_remove_sha256": _array_hash(first.m5_remove.astype(np.uint8)),
    }
    deterministic = (
        np.array_equal(first.corrected, second.corrected)
        and np.array_equal(first.baseline, second.baseline)
        and np.array_equal(first.decisions, second.decisions)
        and np.array_equal(first.safe_remove, second.safe_remove)
    )
    mismatches = {key: {"expected": value, "current": current.get(key)} for key, value in expected.items() if current.get(key) != value}
    return {"passed": bool(deterministic and not mismatches), "deterministic": deterministic, "hashes": current, "mismatches": mismatches}


__all__ = ["FrozenArtifacts", "RuntimeContext", "PageResult", "verify_artifacts", "prepare_mask", "run_page", "replay_page"]
