"""a5_n88_operator.py

Alpha-only inference wrapper for the Notebook 88 contextual safe-cleanup CNNs.

Contract
--------
* A1 is the fallback and remains byte-for-byte unchanged except at final CLEAN pixels.
* CLEAN is permitted only inside ``eligible_mask`` supplied by the caller.
* Final CLEAN pixels are replaced by the supplied local ``paper`` estimate.
* Pixels outside the authoritative mask are asserted unchanged from ``observed``.
* The original Notebook 88 fold ``.pt`` and isotonic ``.joblib`` files are loaded
  directly. They are not refitted, recalibrated or re-saved.

Deployment note
---------------
Notebook 88 persisted out-of-fold models rather than a single all-development
inference model. For unseen Hugging Face pages this alpha wrapper uses a
transparent fold ensemble: each fold probability is calibrated by its matching
isotonic calibrator and the calibrated probabilities are averaged. This is an
Alpha test adaptation, not a claim that Notebook 88 froze a production policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
import csv
import json
import re

import cv2
import joblib
import numpy as np
import torch
from torch import nn


Array = np.ndarray
PathLike = Union[str, Path]


DEFAULT_CHANNELS: Tuple[str, ...] = (
    "obs",
    "norm",
    "mask",
    "depth",
    "ambiguous",
    "p_annotation",
    "p_print_value",
    "p_print_available",
    "edge",
    "horizontal",
    "vertical",
    "compact",
)


@dataclass(frozen=True)
class A5N88Config:
    bundle_dir: PathLike = Path("frozen/a5_notebook88")
    patch_size: int = 33
    batch_size: int = 512
    clean_threshold: float = 0.995
    minimum_clean_run: int = 2
    ensemble_reducer: str = "mean"
    device: Optional[str] = None
    channel_names: Tuple[str, ...] = DEFAULT_CHANNELS


@dataclass
class A5N88Result:
    output: Array
    clean_mask: Array
    eligible_mask: Array
    cleanup_probability: Array
    fold_probability_min: Array
    fold_probability_max: Array
    fold_probability_std: Array
    fold_files: Tuple[str, ...]
    diagnostics: Dict[str, object]


class Notebook88CNN(nn.Module):
    """Compact Notebook 88 patch CNN reconstructed from its saved state_dict schema."""

    def __init__(self, input_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 24, 3, padding=1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1))


def _as_gray_u8(name: str, value: Array, shape: Optional[Tuple[int, int]] = None) -> Array:
    a = np.asarray(value)
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        else:
            a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D greyscale array; got shape {a.shape}.")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} does not match required shape {shape}.")
    if a.dtype != np.uint8:
        if not np.isfinite(a).all():
            raise ValueError(f"{name} contains non-finite values.")
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return a


def _as_bool(name: str, value: Array, shape: Tuple[int, int]) -> Array:
    a = np.asarray(value)
    if a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} does not match required shape {shape}.")
    return a.astype(bool)


def _as_probability(name: str, value: Array, shape: Tuple[int, int]) -> Array:
    a = np.asarray(value, dtype=np.float32)
    if a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} does not match required shape {shape}.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values.")
    if a.min(initial=0.0) < 0.0 or a.max(initial=0.0) > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return a


def _read_channel_schema(bundle: Path, fallback: Sequence[str]) -> Tuple[str, ...]:
    candidates = (bundle / "channel_schema.json", bundle / "inputs" / "channel_schema.json")
    for path in candidates:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            channels = obj.get("channels") or obj.get("ordered_channels") or obj.get("channel_names")
            if channels:
                return tuple(str(x) for x in channels)
    return tuple(fallback)


def _fold_number(path: Path) -> int:
    m = re.search(r"fold[_-]?(\d+)", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot determine fold number from {path.name!r}.")
    return int(m.group(1))


def _inventory_paths(bundle: Path, subdir: str, suffix: str, inventory_name: str) -> Sequence[Path]:
    folder = bundle / subdir
    if not folder.exists():
        raise FileNotFoundError(f"Missing Notebook 88 directory: {folder}")

    inventory_candidates = (bundle / inventory_name, bundle / "audit" / inventory_name)
    for inventory in inventory_candidates:
        if not inventory.exists():
            continue
        rows = list(csv.DictReader(inventory.open("r", encoding="utf-8-sig", newline="")))
        names = []
        for row in rows:
            for key in ("checkpoint", "joblib", "path", "file", "filename", "source_file"):
                raw = row.get(key)
                if raw and str(raw).lower().endswith(suffix):
                    names.append(Path(str(raw)).name)
                    break
        paths = [folder / name for name in names]
        if paths and all(p.exists() for p in paths):
            return sorted(paths, key=_fold_number)

    paths = sorted(folder.glob(f"*{suffix}"), key=_fold_number)
    if not paths:
        raise FileNotFoundError(f"No {suffix} files found in {folder}")
    return paths


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise ValueError("Checkpoint is not a plain state_dict or supported wrapper dictionary.")


def _normalise_state_keys(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = str(key)
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = value
    return out


def _build_channels(
    observed: Array,
    authoritative_mask: Array,
    ambiguous: Array,
    paper: Array,
    horizontal: Array,
    vertical: Array,
    compact: Array,
    p_annotation: Array,
    p_print_value: Array,
    p_print_available: Array,
    channel_names: Sequence[str],
) -> Dict[str, Array]:
    shape = observed.shape
    mask = authoritative_mask.astype(bool)
    paper_safe = np.maximum(paper, 1)
    norm = cv2.divide(observed, paper_safe, scale=255).astype(np.float32) / 255.0
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    max_depth = float(depth.max())
    if max_depth > 0:
        depth /= max_depth
    edge = (cv2.Canny(observed, 70, 170) > 0).astype(np.float32)

    all_channels: Dict[str, Array] = {
        # 1. observed
        "obs": observed.astype(np.float32) / 255.0,
        "observed": observed.astype(np.float32) / 255.0,

        # 2. local-normalised observed intensity
        "norm": norm,
        "normalised": norm,
        "normalized": norm,
        "local_normalised": norm,
        "local_normalized": norm,

        # 3. authoritative reviewed mask
        "mask": mask.astype(np.float32),
        "reviewed_mask": mask.astype(np.float32),
        "authoritative_mask": mask.astype(np.float32),

        # 4. mask depth
        "depth": depth,
        "mask_depth": depth,

        # 5. A1 / V7.8 ambiguous route
        "ambiguous": ambiguous.astype(np.float32),

        # 6. annotation probability
        "p_annotation": p_annotation.astype(np.float32),

        # 7. print probability value
        "p_print_value": p_print_value.astype(np.float32),
        "p_print": p_print_value.astype(np.float32),

        # 8. print-probability availability
        "p_print_available": p_print_available.astype(np.float32),

        # 9. observed edge
        "edge": edge,

        # 10. horizontal structural support
        "horizontal": horizontal.astype(np.float32),
        "horizontal_support": horizontal.astype(np.float32),

        # 11. vertical structural support
        "vertical": vertical.astype(np.float32),
        "vertical_support": vertical.astype(np.float32),

        # 12. compact structural support
        "compact": compact.astype(np.float32),
        "compact_support": compact.astype(np.float32),
    }
    missing = [name for name in channel_names if name not in all_channels]
    if missing:
        raise RuntimeError(f"Unsupported Notebook 88 channel name(s): {missing}")
    for name in channel_names:
        if all_channels[name].shape != shape or not np.isfinite(all_channels[name]).all():
            raise RuntimeError(f"Invalid channel {name!r}.")
    return all_channels


def _extract_patches(channels: Mapping[str, Array], names: Sequence[str], ys: Array, xs: Array, patch: int) -> Array:
    if patch % 2 != 1:
        raise ValueError("patch_size must be odd.")
    half = patch // 2
    padded = {name: np.pad(channels[name], half, mode="reflect") for name in names}
    return np.stack(
        [np.stack([padded[name][int(y):int(y)+patch, int(x):int(x)+patch] for name in names]) for y, x in zip(ys, xs)],
        axis=0,
    ).astype(np.float32)


def _predict_raw(model: nn.Module, patches: Array, batch_size: int, device: str) -> Array:
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            xb = torch.from_numpy(patches[start:start + batch_size]).to(device)
            p = torch.softmax(model(xb), dim=1)[:, 1]
            out.append(p.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def _calibrate(calibrator: object, raw_probability: Array) -> Array:
    if hasattr(calibrator, "predict"):
        calibrated = calibrator.predict(raw_probability)
    elif hasattr(calibrator, "transform"):
        calibrated = calibrator.transform(raw_probability)
    else:
        raise TypeError(f"Unsupported calibrator type: {type(calibrator).__name__}")
    calibrated = np.asarray(calibrated, dtype=np.float64)
    if calibrated.shape != raw_probability.shape or not np.isfinite(calibrated).all():
        raise RuntimeError("Calibrator returned invalid probabilities.")
    return np.clip(calibrated, 0.0, 1.0)


def _remove_small_runs(mask: Array, minimum_run: int) -> Tuple[Array, int]:
    mask = mask.astype(bool)
    if minimum_run <= 1 or not mask.any():
        return mask, 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    demoted = 0
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        component = labels == component_id
        if area >= minimum_run:
            keep[component] = True
        else:
            demoted += area
    return keep, demoted


def run_a5_n88(
    observed: Array,
    a1_output: Array,
    authoritative_mask: Array,
    ambiguous_mask: Array,
    paper_estimate: Array,
    horizontal_support: Array,
    vertical_support: Array,
    compact_support: Array,
    p_annotation: Array,
    p_print_value: Optional[Array] = None,
    p_print_available: Optional[Array] = None,
    eligible_mask: Optional[Array] = None,
    config: Optional[A5N88Config] = None,
) -> A5N88Result:
    """Run the Notebook 88 alpha cleaner on one page.

    ``eligible_mask`` should be the exact population that Notebook 88 is allowed to
    alter. If omitted, the conservative default is authoritative_mask AND
    ambiguous_mask. If the Notebook 88 notebook used a narrower condition in the
    HF application, pass that exact array explicitly.
    """

    cfg = config or A5N88Config()
    bundle = Path(cfg.bundle_dir)
    if not bundle.exists():
        raise FileNotFoundError(f"Notebook 88 bundle not found: {bundle}")

    observed = _as_gray_u8("observed", observed)
    shape = observed.shape
    a1 = _as_gray_u8("a1_output", a1_output, shape)
    paper = _as_gray_u8("paper_estimate", paper_estimate, shape)
    authority = _as_bool("authoritative_mask", authoritative_mask, shape)
    ambiguous = _as_bool("ambiguous_mask", ambiguous_mask, shape)
    horizontal = _as_bool("horizontal_support", horizontal_support, shape)
    vertical = _as_bool("vertical_support", vertical_support, shape)
    compact = _as_bool("compact_support", compact_support, shape)
    p_ann = _as_probability("p_annotation", p_annotation, shape)

    if p_print_value is None:
        p_print = np.zeros(shape, dtype=np.float32)
    else:
        p_print = _as_probability("p_print_value", p_print_value, shape)
    if p_print_available is None:
        p_print_avail = np.zeros(shape, dtype=np.float32)
    else:
        p_print_avail = _as_probability("p_print_available", p_print_available, shape)

    eligible = authority & ambiguous if eligible_mask is None else _as_bool("eligible_mask", eligible_mask, shape)
    if np.any(eligible & ~authority):
        raise ValueError("eligible_mask contains pixels outside authoritative_mask.")
    if np.any(eligible & ~ambiguous):
        raise ValueError("eligible_mask contains pixels outside ambiguous_mask.")

    schema_channels = _read_channel_schema(bundle, cfg.channel_names)
    if len(schema_channels) != 12:
        raise RuntimeError(f"Notebook 88 requires 12 channels; schema contains {len(schema_channels)}: {schema_channels}")

    model_paths = _inventory_paths(bundle, "models", ".pt", "model_inventory.csv")
    calibrator_paths = _inventory_paths(bundle, "calibrators", ".joblib", "calibrator_inventory.csv")
    model_by_fold = {_fold_number(p): p for p in model_paths}
    calibrator_by_fold = {_fold_number(p): p for p in calibrator_paths}
    if set(model_by_fold) != set(calibrator_by_fold):
        raise RuntimeError(
            f"Model/calibrator fold mismatch: models={sorted(model_by_fold)}, calibrators={sorted(calibrator_by_fold)}"
        )

    ys, xs = np.where(eligible)
    probability = np.full(shape, np.nan, dtype=np.float32)
    p_min = np.full(shape, np.nan, dtype=np.float32)
    p_max = np.full(shape, np.nan, dtype=np.float32)
    p_std = np.full(shape, np.nan, dtype=np.float32)

    if len(ys) == 0:
        return A5N88Result(
            output=a1.copy(),
            clean_mask=np.zeros(shape, bool),
            eligible_mask=eligible,
            cleanup_probability=probability,
            fold_probability_min=p_min,
            fold_probability_max=p_max,
            fold_probability_std=p_std,
            fold_files=tuple(),
            diagnostics={"eligible_pixels": 0, "clean_pixels": 0, "folds": 0},
        )

    channels = _build_channels(
        observed, authority, ambiguous, paper, horizontal, vertical, compact,
        p_ann, p_print, p_print_avail, schema_channels,
    )
    patches = _extract_patches(channels, schema_channels, ys, xs, cfg.patch_size)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fold_probabilities = []
    fold_files = []

    for fold in sorted(model_by_fold):
        model_path = model_by_fold[fold]
        calibrator_path = calibrator_by_fold[fold]
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = _normalise_state_keys(_extract_state_dict(checkpoint))
        model = Notebook88CNN(input_channels=len(schema_channels))
        model.load_state_dict(state, strict=True)
        model.to(device)
        raw = _predict_raw(model, patches, cfg.batch_size, device)
        calibrated = _calibrate(joblib.load(calibrator_path), raw)
        fold_probabilities.append(calibrated)
        fold_files.append(f"fold{fold}:{model_path.name}|{calibrator_path.name}")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    fold_stack = np.stack(fold_probabilities, axis=0)
    if cfg.ensemble_reducer == "mean":
        combined = fold_stack.mean(axis=0)
    elif cfg.ensemble_reducer == "median":
        combined = np.median(fold_stack, axis=0)
    elif cfg.ensemble_reducer == "minimum":
        combined = fold_stack.min(axis=0)
    else:
        raise ValueError("ensemble_reducer must be 'mean', 'median' or 'minimum'.")

    probability[ys, xs] = combined.astype(np.float32)
    p_min[ys, xs] = fold_stack.min(axis=0).astype(np.float32)
    p_max[ys, xs] = fold_stack.max(axis=0).astype(np.float32)
    p_std[ys, xs] = fold_stack.std(axis=0).astype(np.float32)

    provisional_clean = np.zeros(shape, dtype=bool)
    provisional_clean[ys, xs] = combined >= float(cfg.clean_threshold)
    clean_mask, demoted = _remove_small_runs(provisional_clean, int(cfg.minimum_clean_run))
    clean_mask &= eligible

    output = a1.copy()
    output[clean_mask] = paper[clean_mask]

    if not np.array_equal(output[~clean_mask], a1[~clean_mask]):
        raise AssertionError("A5-N88 changed A1 outside final CLEAN pixels.")
    if not np.array_equal(output[~authority], observed[~authority]):
        raise AssertionError("A5-N88 output differs from observed outside authoritative mask.")
    if np.any(clean_mask & ~eligible):
        raise AssertionError("A5-N88 cleaned an ineligible pixel.")

    diagnostics: Dict[str, object] = {
        "bundle_dir": str(bundle),
        "device": device,
        "channels": list(schema_channels),
        "patch_size": int(cfg.patch_size),
        "folds": len(fold_stack),
        "ensemble_reducer": cfg.ensemble_reducer,
        "clean_threshold": float(cfg.clean_threshold),
        "minimum_clean_run": int(cfg.minimum_clean_run),
        "eligible_pixels": int(eligible.sum()),
        "provisional_clean_pixels": int(provisional_clean.sum()),
        "singleton_or_short_run_demotions": int(demoted),
        "clean_pixels": int(clean_mask.sum()),
        "outside_mask_drift": int(np.sum(output[~authority] != observed[~authority])),
        "non_clean_a1_regression": int(np.sum(output[~clean_mask] != a1[~clean_mask])),
    }

    return A5N88Result(
        output=output,
        clean_mask=clean_mask,
        eligible_mask=eligible,
        cleanup_probability=probability,
        fold_probability_min=p_min,
        fold_probability_max=p_max,
        fold_probability_std=p_std,
        fold_files=tuple(fold_files),
        diagnostics=diagnostics,
    )


__all__ = [
    "A5N88Config",
    "A5N88Result",
    "Notebook88CNN",
    "run_a5_n88",
]
