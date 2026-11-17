"""Frozen V8.8 observed-print operator for Notebook 72.

Public API
==========
    operator = load_operator(v88_bundle_root)
    replay = replay_page(operator, v88_bundle_root)
    result = run_page(observed, authoritative_seed_mask, operator, runtime_context=None)

Bundle contract
===============
`v88_bundle_root` is a DIRECTORY containing `operator_bundle.json` either at
its root or below `policy_bundle/`.

The manifest must identify a deployable, all-development model rather than an
OOF decision table.  Example:

{
  "operator_id": "v8.8-frozen-remove-operator",
  "policy_id": "...",
  "implementation_id": "...",
  "architecture": "topology_cnn_v8_2",
  "model": {"path": "artefacts/model.pt", "sha256": "..."},
  "calibrator": {"path": "artefacts/calibrator.joblib", "sha256": "..."},
  "channel_schema": {"path": "artefacts/channel_schema.json", "sha256": "..."},
  "policy": {"path": "artefacts/policy.json", "sha256": "..."},
  "replay": {"path": "replay/replay_test.npz", "sha256": "..."},
  "constants": {
    "mask_dilation": 2,
    "patch": 33,
    "dark_threshold": 200,
    "print_threshold": 0.995,
    "annotation_threshold": 0.995,
    "print_margin": 0.990,
    "minimum_print_run": 2,
    "batch_size": 512,
    "crop_margin": 48
  }
}

The channel schema must contain the ordered V8 channels. The supported frozen
architecture is the compact 11-channel V8 TopologyCNN used in the durable V8.2
lineage. The module fails closed if the Notebook 52 directory contains only
registered Notebook 50 OOF decisions. OOF rows are replay evidence, not an
unseen-page decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import hashlib
import json
import platform
import sys
import time

import cv2
import joblib
import numpy as np

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


OPERATOR_ID = "v8.8-frozen-observed-print-operator"
OPERATOR_VERSION = "3.0.0"

DECISION_INELIGIBLE = np.uint8(0)
DECISION_ANNOTATION = np.uint8(1)
DECISION_UNCERTAIN = np.uint8(2)
DECISION_PRINT = np.uint8(3)

DEFAULT_CHANNELS = (
    "obs", "norm", "mask", "depth", "horizontal", "vertical", "compact",
    "preserve", "remove", "ambiguous", "edge",
)


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON file: {path}")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Unreadable JSON file {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _resolve_file(root: Path, record: Mapping[str, Any], label: str) -> Path:
    raw = record.get("path")
    _require(raw, f"operator_bundle.json is missing {label}.path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    _require(path.is_file(), f"Missing frozen {label}: {path}")
    expected = str(record.get("sha256", "")).strip().lower()
    _require(expected, f"operator_bundle.json is missing {label}.sha256")
    actual = _sha256_file(path)
    _require(actual == expected, f"Frozen {label} hash mismatch: {path}")
    return path


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)


def _immutable(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = original.copy()
    selected = np.asarray(mask) > 0
    result[selected] = candidate[selected]
    _require(np.array_equal(result[~selected], original[~selected]),
             "Candidate changed pixels outside its authorised mask.")
    return result


if nn is not None:
    class TopologyCNN(nn.Module):
        """Exact compact CNN architecture used by the V8.2 durable lineage."""

        def __init__(self, nc: int = 11):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(nc, 24, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(24),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, 2)

        def forward(self, value):
            return self.head(self.net(value).flatten(1))
else:  # pragma: no cover
    class TopologyCNN:  # type: ignore
        pass


@dataclass(frozen=True)
class OperatorContext:
    bundle_root: Path
    bundle_manifest_path: Path
    bundle_manifest: Mapping[str, Any]
    policy: Mapping[str, Any]
    channel_schema: Mapping[str, Any]
    model_path: Path
    calibrator_path: Path
    policy_path: Path
    schema_path: Path
    replay_path: Path
    model: Any
    calibrator: Any
    device: str
    channels: tuple[str, ...]
    patch: int
    mask_dilation: int
    dark_threshold: int
    print_threshold: float
    annotation_threshold: float
    print_margin: float
    minimum_print_run: int
    batch_size: int
    crop_margin: int
    audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "unseen_page_inference": True,
            "requires_external_decisions": False,
            "uses_clean_truth": False,
        }

    def run_page(
        self,
        observed: np.ndarray,
        authoritative_seed_mask: np.ndarray,
        runtime_context: Optional[Mapping[str, Any]] = None,
    ) -> "PageResult":
        return run_page(observed, authoritative_seed_mask, self, runtime_context)

    def run_replay_test(self) -> dict[str, Any]:
        return replay_page(self, self.bundle_root)


@dataclass
class PageResult:
    corrected: np.ndarray
    baseline: np.ndarray
    prepared_mask: np.ndarray
    decisions: np.ndarray
    probabilities: dict[str, np.ndarray]
    changed: dict[str, np.ndarray]
    audit: dict[str, Any]
    print_mask: np.ndarray
    annotation_mask: np.ndarray
    uncertain_mask: np.ndarray
    eligible_mask: np.ndarray


def load_operator(
    v88_bundle_root: Any,
) -> OperatorContext:
    """
    Load the completed Notebook 52 V8.8A freeze and its authoritative
    Notebook 50 registered-decision source.

    This loader supports registered replay. It does not claim that an
    all-development deployable scorer for previously unseen pages exists.
    """

    root = Path(v88_bundle_root).expanduser().resolve()

    _require(
        root.is_dir(),
        f"v88_bundle_root must be a Notebook 52 run directory: {root}",
    )

    verdict_path = root / "notebook52_verdict.json"
    policy_path = root / "policy_bundle" / "policy_manifest.json"

    _require(
        verdict_path.is_file(),
        f"Missing Notebook 52 verdict: {verdict_path}",
    )

    _require(
        policy_path.is_file(),
        f"Missing Notebook 52 policy manifest: {policy_path}",
    )

    verdict = _read_json(verdict_path)
    policy = _read_json(policy_path)

    _require(
        verdict.get("verdict")
        == "PASS_M5_REMOVE_FREEZE_READY_FOR_OPERATIONAL_BUILDER",
        f"Notebook 52 did not pass its operational-builder gate: {verdict}",
    )

    _require(
        policy.get("policy") == "M5_REMOVE",
        f"Frozen policy is not M5_REMOVE: {policy.get('policy')!r}",
    )

    _require(
        policy.get("safe_protect_runtime_enabled") is False,
        "SAFE_PROTECT must remain disabled.",
    )

    _require(
        policy.get("retraining_performed") is False,
        "Notebook 52 records unexpected retraining.",
    )

    _require(
        policy.get("recalibration_performed") is False,
        "Notebook 52 records unexpected recalibration.",
    )

    _require(
        policy.get("threshold_changed") is False,
        "Notebook 52 records an unexpected threshold change.",
    )

    notebook50_raw = policy.get("authoritative_notebook50_run")

    _require(
        notebook50_raw,
        "Policy manifest does not identify authoritative_notebook50_run.",
    )

    notebook50_run = Path(
        notebook50_raw
    ).expanduser().resolve()

    _require(
        notebook50_run.is_dir(),
        "Authoritative Notebook 50 run does not exist: "
        f"{notebook50_run}",
    )

    schema_path = (
        notebook50_run
        / "inputs"
        / "predictor_schema.json"
    )

    decision_table_path = (
        notebook50_run
        / "tables"
        / "all_m3_oof_predictions_and_decisions.parquet"
    )

    _require(
        schema_path.is_file(),
        f"Missing Notebook 50 predictor schema: {schema_path}",
    )

    _require(
        decision_table_path.is_file(),
        f"Missing Notebook 50 registered decision table: "
        f"{decision_table_path}",
    )

    schema = _read_json(schema_path)
    decision_table = pd.read_parquet(
        decision_table_path
    )

    required_columns = {
        "page",
        "y",
        "x",
        "safe_remove",
    }

    missing_columns = (
        required_columns
        - set(decision_table.columns)
    )

    _require(
        not missing_columns,
        "Notebook 50 decision table is missing fields: "
        f"{sorted(missing_columns)}",
    )

    decision_table = decision_table.copy()
    decision_table["page"] = (
        decision_table["page"].astype(str)
    )
    decision_table["y"] = pd.to_numeric(
        decision_table["y"],
        errors="raise",
    ).astype(int)
    decision_table["x"] = pd.to_numeric(
        decision_table["x"],
        errors="raise",
    ).astype(int)
    decision_table["safe_remove"] = (
        decision_table["safe_remove"]
        .astype(bool)
    )

    _require(
        not decision_table.duplicated(
            ["page", "y", "x"]
        ).any(),
        "Notebook 50 decision table contains duplicate "
        "page/y/x coordinates.",
    )

    audit = {
        "operator_id":
            "v8.8-registered-m5-remove-replay",

        "operator_version":
            OPERATOR_VERSION,

        "policy_id":
            str(policy["policy_id"]),

        "notebook52_run":
            str(root),

        "notebook52_verdict_sha256":
            _sha256_file(verdict_path),

        "policy_manifest_sha256":
            _sha256_file(policy_path),

        "authoritative_notebook50_run":
            str(notebook50_run),

        "predictor_schema_sha256":
            _sha256_file(schema_path),

        "decision_table_sha256":
            _sha256_file(decision_table_path),

        "registered_pages":
            int(decision_table["page"].nunique()),

        "registered_coordinates":
            int(len(decision_table)),

        "unseen_page_inference":
            False,

        "requires_external_decisions":
            True,

        "clean_truth_used":
            False,
    }

    return OperatorContext(
        v88_bundle_root=root,
        notebook50_run=notebook50_run,
        policy_id=str(policy["policy_id"]),
        policy_manifest_path=policy_path,
        predictor_schema_path=schema_path,
        decision_table_path=decision_table_path,
        policy=policy,
        schema=schema,
        decision_table=decision_table,
        audit=audit,
    )

def prepare_mask(seed: np.ndarray, dilation: int = 2) -> np.ndarray:
    mask = _binary(seed)
    _require(mask.ndim == 2, "Authoritative seed mask must be 2-D.")
    _require(dilation >= 0, "mask_dilation must be non-negative.")
    if dilation:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation + 1, 2 * dilation + 1),
        )
        mask = cv2.dilate(mask, kernel)
    return mask


def _padded_components(mask: np.ndarray, margin: int) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    height, width = mask.shape
    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, n):
        x, y, box_width, box_height, area = stats[component]
        if int(area) >= 8:
            boxes.append((
                max(0, int(x) - margin),
                max(0, int(y) - margin),
                min(width, int(x + box_width) + margin),
                min(height, int(y + box_height) + margin),
            ))
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        while boxes:
            ax0, ay0, ax1, ay1 = boxes.pop()
            rest = []
            for bx0, by0, bx1, by1 in boxes:
                if not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0):
                    ax0, ay0 = min(ax0, bx0), min(ay0, by0)
                    ax1, ay1 = max(ax1, bx1), max(ay1, by1)
                    changed = True
                else:
                    rest.append((bx0, by0, bx1, by1))
            boxes = rest
            merged.append((ax0, ay0, ax1, ay1))
        boxes = merged
    return boxes


def _telea(observed: np.ndarray, mask: np.ndarray, margin: int) -> np.ndarray:
    result = observed.copy()
    for x0, y0, x1, y1 in _padded_components(mask, margin):
        image_crop = observed[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]
        candidate = cv2.inpaint(image_crop, mask_crop, 3, cv2.INPAINT_TELEA)
        local = result[y0:y1, x0:x1]
        local[mask_crop > 0] = candidate[mask_crop > 0]
        result[y0:y1, x0:x1] = local
    return _immutable(observed, result, mask)


def _background_surface(observed: np.ndarray, mask_bool: np.ndarray, sigma: float = 45.0) -> np.ndarray:
    visible = (~mask_bool) & (observed >= 200)
    numerator = cv2.GaussianBlur(observed.astype(np.float32) * visible, (0, 0), sigma)
    denominator = cv2.GaussianBlur(visible.astype(np.float32), (0, 0), sigma)
    fallback = cv2.GaussianBlur(observed, (0, 0), sigma)
    return np.clip(
        np.where(denominator > .02, numerator / np.maximum(denominator, 1e-4), fallback),
        0,
        255,
    ).astype(np.uint8)


def _support_maps(observed: np.ndarray, mask: np.ndarray, band: int = 4):
    dark = (observed < 190) & (mask == 0)
    horizontal = cv2.morphologyEx(
        dark.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((1, 35), np.uint8),
    ) > 0
    vertical = cv2.morphologyEx(
        dark.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((15, 1), np.uint8),
    ) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    compact = np.zeros_like(dark)
    for component in range(1, n):
        x, y, width, height, area = stats[component]
        if 3 <= area <= 650 and width <= 40 and height <= 40:
            compact[labels == component] = True
    kernel = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    return (
        cv2.dilate(horizontal.astype(np.uint8), kernel) > 0,
        cv2.dilate(vertical.astype(np.uint8), kernel) > 0,
        cv2.dilate(compact.astype(np.uint8), kernel) > 0,
    )


def _v78_arrays(observed: np.ndarray, mask: np.ndarray, crop_margin: int):
    mask_bool = mask > 0
    telea = _telea(observed, mask, crop_margin)
    paper = _background_surface(observed, mask_bool)
    normalised = cv2.divide(observed, np.maximum(paper, 1), scale=255)
    horizontal, vertical, compact = _support_maps(observed, mask)
    preserve = mask_bool & (horizontal | vertical | compact) & (observed < 200)
    remove = mask_bool & ~preserve & (observed >= 80) & (normalised >= 80)
    ambiguous = mask_bool & ~preserve & ~remove
    baseline = observed.copy()
    baseline[remove] = paper[remove]
    baseline[ambiguous] = telea[ambiguous]
    baseline[preserve] = observed[preserve]
    baseline = _immutable(observed, baseline, mask)
    return {
        "obs": observed,
        "norm": normalised,
        "mask": mask_bool,
        "depth": cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5),
        "horizontal": horizontal,
        "vertical": vertical,
        "compact": compact,
        "preserve": preserve,
        "remove": remove,
        "ambiguous": ambiguous,
        "edge": cv2.Canny(observed, 70, 170) > 0,
        "baseline": baseline,
    }


def _build_padded_channels(arrays: Mapping[str, np.ndarray], channels: Sequence[str], half: int):
    padded = {}
    for key in channels:
        value = arrays[key].astype(np.float32)
        if key in {"obs", "norm"}:
            value /= 255.0
        padded[key] = np.pad(value, half, mode="reflect")
    return padded


def _score(
    operator: OperatorContext,
    arrays: Mapping[str, np.ndarray],
    ys: np.ndarray,
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    half = operator.patch // 2
    padded = _build_padded_channels(arrays, operator.channels, half)
    raw = np.empty(len(ys), dtype=np.float64)
    operator.model.eval()
    with torch.no_grad():
        for start in range(0, len(ys), operator.batch_size):
            stop = min(start + operator.batch_size, len(ys))
            batch = np.stack([
                np.stack([
                    padded[key][int(y):int(y) + operator.patch,
                                int(x):int(x) + operator.patch]
                    for key in operator.channels
                ])
                for y, x in zip(ys[start:stop], xs[start:stop])
            ]).astype(np.float32)
            tensor = torch.from_numpy(batch).to(operator.device)
            logits = operator.model(tensor)
            raw[start:stop] = torch.softmax(logits, 1)[:, 1].cpu().numpy()
    calibrated = np.asarray(operator.calibrator.predict(raw), dtype=np.float64)
    _require(np.isfinite(raw).all(), "Non-finite raw PRINT score.")
    _require(np.isfinite(calibrated).all(), "Non-finite calibrated PRINT probability.")
    _require(((calibrated >= 0) & (calibrated <= 1)).all(),
             "Calibrated PRINT probability outside [0, 1].")
    return raw, calibrated


def _decide(operator: OperatorContext, p_print: np.ndarray) -> np.ndarray:
    p_annotation = 1.0 - p_print
    decision = np.full(len(p_print), DECISION_UNCERTAIN, np.uint8)
    decision[p_annotation >= operator.annotation_threshold] = DECISION_ANNOTATION
    print_rule = (
        (p_print >= operator.print_threshold)
        & ((p_print - p_annotation) >= operator.print_margin)
    )
    decision[print_rule] = DECISION_PRINT
    return decision


def _apply_run_rule(
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
    decisions: np.ndarray,
    minimum_run: int,
) -> int:
    if minimum_run <= 1:
        return 0
    print_map = np.zeros(shape, np.uint8)
    selected = decisions == DECISION_PRINT
    print_map[ys[selected], xs[selected]] = 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(print_map, 8)
    demoted = 0
    for component in range(1, n):
        if int(stats[component, cv2.CC_STAT_AREA]) < minimum_run:
            component_map = labels == component
            hit = selected & component_map[ys, xs]
            decisions[hit] = DECISION_UNCERTAIN
            demoted += int(hit.sum())
    return demoted


def run_page(
    observed: np.ndarray,
    authoritative_seed_mask: np.ndarray,
    runtime_context: OperatorContext,
    page_context: Optional[Mapping[str, Any]] = None,
) -> PageResult:
    """Infer V8.8 decisions for one previously unseen reviewed page."""
    started = time.perf_counter()
    _require(isinstance(runtime_context, OperatorContext),
             "runtime_context must be returned by load_operator().")
    operator = runtime_context
    context = dict(page_context or {})

    observed = np.asarray(observed)
    seed = np.asarray(authoritative_seed_mask)
    _require(observed.ndim == 2 and observed.dtype == np.uint8,
             "observed must be a 2-D uint8 array.")
    _require(seed.ndim == 2 and seed.shape == observed.shape,
             "authoritative_seed_mask must be 2-D and match observed shape.")

    prepared = prepare_mask(seed, operator.mask_dilation)
    arrays = _v78_arrays(observed, prepared, operator.crop_margin)
    baseline = arrays["baseline"]
    eligible = (prepared > 0) & (observed < operator.dark_threshold) & ~arrays["preserve"]
    ys, xs = np.where(eligible)

    raw_map = np.full(observed.shape, np.nan, np.float32)
    print_map = np.full(observed.shape, np.nan, np.float32)
    annotation_map = np.full(observed.shape, np.nan, np.float32)
    decisions = np.full(observed.shape, DECISION_INELIGIBLE, np.uint8)

    if len(ys):
        raw, p_print = _score(operator, arrays, ys, xs)
        local_decisions = _decide(operator, p_print)
        demoted = _apply_run_rule(
            observed.shape, ys, xs, local_decisions, operator.minimum_print_run
        )
        raw_map[ys, xs] = raw.astype(np.float32)
        print_map[ys, xs] = p_print.astype(np.float32)
        annotation_map[ys, xs] = (1.0 - p_print).astype(np.float32)
        decisions[ys, xs] = local_decisions
    else:
        demoted = 0

    print_mask = decisions == DECISION_PRINT
    annotation_mask = decisions == DECISION_ANNOTATION
    uncertain_mask = decisions == DECISION_UNCERTAIN

    corrected = baseline.copy()
    corrected[print_mask] = observed[print_mask]
    corrected[prepared == 0] = observed[prepared == 0]

    outside_drift = int(np.sum(corrected[prepared == 0] != observed[prepared == 0]))
    nonprint_regression = int(np.sum(corrected[~print_mask] != baseline[~print_mask]))
    changed_not_observed = int(np.sum((corrected != baseline) & (corrected != observed)))
    _require(outside_drift == 0, "Outside-mask drift detected.")
    _require(nonprint_regression == 0, "Non-PRINT regression from frozen V7.8 detected.")
    _require(changed_not_observed == 0,
             "A V8.8-changed pixel does not equal the observed input byte.")

    changed_from_observed = corrected != observed
    changed_from_baseline = corrected != baseline
    audit = {
        **dict(operator.audit),
        **context,
        "clean_truth_used": False,
        "observed_sha256": _sha256_array(observed),
        "seed_mask_sha256": _sha256_array(_binary(seed)),
        "prepared_mask_sha256": _sha256_array(prepared),
        "baseline_sha256": _sha256_array(baseline),
        "corrected_sha256": _sha256_array(corrected),
        "decisions_sha256": _sha256_array(decisions),
        "eligible_pixels": int(eligible.sum()),
        "print_pixels": int(print_mask.sum()),
        "annotation_pixels": int(annotation_mask.sum()),
        "uncertain_pixels": int(uncertain_mask.sum()),
        "singleton_or_short_run_demotions": int(demoted),
        "outside_mask_drift": outside_drift,
        "nonprint_v78_regression": nonprint_regression,
        "changed_not_equal_observed": changed_not_observed,
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "device": operator.device,
    }

    return PageResult(
        corrected=corrected,
        baseline=baseline,
        prepared_mask=prepared,
        decisions=decisions,
        probabilities={
            "raw_print": raw_map,
            "p_print": print_map,
            "p_annotation": annotation_map,
        },
        changed={
            "from_observed": changed_from_observed,
            "from_baseline": changed_from_baseline,
            "valid_intervention": prepared > 0,
        },
        audit=audit,
        print_mask=print_mask,
        annotation_mask=annotation_mask,
        uncertain_mask=uncertain_mask,
        eligible_mask=eligible,
    )


def replay_page(operator: OperatorContext, bundle_root: Any = None) -> dict[str, Any]:
    """Run the exact frozen replay pack and verify semantic hashes."""
    _require(isinstance(operator, OperatorContext),
             "operator must be returned by load_operator().")
    if bundle_root is not None:
        supplied = Path(bundle_root).expanduser().resolve()
        _require(supplied == operator.bundle_root,
                 "replay bundle_root differs from the loaded operator bundle.")

    try:
        with np.load(operator.replay_path, allow_pickle=False) as replay:
            required = {
                "observed", "authoritative_seed_mask", "corrected_sha256",
                "baseline_sha256", "decisions_sha256", "prepared_mask_sha256",
            }
            missing = sorted(required - set(replay.files))
            _require(not missing, f"Replay pack missing arrays/fields: {missing}")
            observed = replay["observed"]
            seed = replay["authoritative_seed_mask"]
            expected = {
                "corrected_sha256": str(replay["corrected_sha256"].item()),
                "baseline_sha256": str(replay["baseline_sha256"].item()),
                "decisions_sha256": str(replay["decisions_sha256"].item()),
                "prepared_mask_sha256": str(replay["prepared_mask_sha256"].item()),
            }

        first = run_page(observed, seed, operator, {"replay": True})
        second = run_page(observed, seed, operator, {"replay": True})
        actual = {
            "corrected_sha256": _sha256_array(first.corrected),
            "baseline_sha256": _sha256_array(first.baseline),
            "decisions_sha256": _sha256_array(first.decisions),
            "prepared_mask_sha256": _sha256_array(first.prepared_mask),
        }
        deterministic = (
            np.array_equal(first.corrected, second.corrected)
            and np.array_equal(first.baseline, second.baseline)
            and np.array_equal(first.decisions, second.decisions)
            and np.array_equal(first.prepared_mask, second.prepared_mask)
        )
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        return {
            "passed": bool(deterministic and not mismatches),
            "operator_id": operator.audit["operator_id"],
            "policy_id": operator.audit["policy_id"],
            "implementation_id": operator.audit["implementation_id"],
            "deterministic": bool(deterministic),
            "expected": expected,
            "actual": actual,
            "mismatches": mismatches,
        }
    except Exception as exc:
        return {
            "passed": False,
            "operator_id": operator.audit.get("operator_id"),
            "policy_id": operator.audit.get("policy_id"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


__all__ = [
    "OPERATOR_ID",
    "OPERATOR_VERSION",
    "OperatorContext",
    "PageResult",
    "TopologyCNN",
    "load_operator",
    "replay_page",
    "run_page",
    "prepare_mask",
]
