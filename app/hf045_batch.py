from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import shutil
import tempfile
import zipfile

import cv2
import numpy as np

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_BYTES = 1_000_000_000
MAX_FILES = 5000
MAX_UNCOMPRESSED_BYTES = 4_000_000_000


class BatchFailure(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_members(archive: zipfile.ZipFile):
    members = archive.infolist()
    if len(members) > MAX_FILES:
        raise BatchFailure(f"ZIP contains too many entries: {len(members)}")
    total = sum(x.file_size for x in members)
    if total > MAX_UNCOMPRESSED_BYTES:
        raise BatchFailure("ZIP uncompressed size exceeds the batch limit")
    for item in members:
        p = PurePosixPath(item.filename)
        if p.is_absolute() or ".." in p.parts:
            raise BatchFailure(f"Unsafe ZIP member path: {item.filename}")
        if item.is_dir():
            continue
        yield item


def _role(path: PurePosixPath):
    parts = [x.lower() for x in path.parts]
    stem = path.stem.lower()
    if path.suffix.lower() not in SUPPORTED:
        return None
    if any(x in {"masks", "mask", "reviewed_masks", "corrected_masks", "authoritative_masks"} for x in parts[:-1]):
        return "mask"
    if stem.endswith(("_mask", "_reviewed_mask", "_corrected_mask", "_authoritative_mask")):
        return "mask"
    if any(x in {"observed", "annotated", "annotated_pages", "images", "pages", "input"} for x in parts[:-1]):
        return "observed"
    return "observed"


def _pair_key(path: PurePosixPath, role: str):
    stem = path.stem
    if role == "mask":
        for suffix in ("_reviewed_authoritative_seed_mask", "_authoritative_mask", "_reviewed_mask", "_corrected_mask", "_mask"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
    return stem


def _decode_gray(data: bytes, name: str):
    a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise BatchFailure(f"Unreadable image: {name}")
    return a


def _write_png(path: Path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(array)):
        raise BatchFailure(f"Could not write PNG: {path.name}")


def run_batch(upload: bytes, operator, canonical_v78, output_root: Path) -> Path:
    if not upload:
        raise BatchFailure("Uploaded ZIP is empty")
    if len(upload) > MAX_UPLOAD_BYTES:
        raise BatchFailure("Uploaded ZIP exceeds the batch limit")
    if operator is None:
        raise BatchFailure("A5 operator is not initialised")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_hf045"
    run = Path(output_root) / run_id
    work = run / "work"
    pages_dir = run / "pages"
    run.mkdir(parents=True, exist_ok=False)
    work.mkdir()
    pages_dir.mkdir()

    try:
        with zipfile.ZipFile(io.BytesIO(upload), "r") as archive:
            records = []
            for item in _safe_members(archive):
                p = PurePosixPath(item.filename)
                role = _role(p)
                if role is None:
                    continue
                data = archive.read(item)
                records.append({
                    "path": str(p), "role": role, "key": _pair_key(p, role),
                    "data": data, "sha256": _sha256_bytes(data),
                })

        if not records:
            raise BatchFailure("ZIP contains no supported image files")

        observed = {}
        masks = {}
        for rec in records:
            target = masks if rec["role"] == "mask" else observed
            key = rec["key"]
            if key in target:
                raise BatchFailure(f"Duplicate {rec['role']} key '{key}': {target[key]['path']} and {rec['path']}")
            target[key] = rec

        keys = sorted(set(observed) | set(masks))
        missing_observed = [k for k in keys if k not in observed]
        missing_masks = [k for k in keys if k not in masks]
        if missing_observed or missing_masks:
            raise BatchFailure(
                "Observed/mask pairing is incomplete. "
                f"Missing observed={missing_observed[:20]}, missing masks={missing_masks[:20]}"
            )

        summary_rows = []
        for key in keys:
            obs = _decode_gray(observed[key]["data"], observed[key]["path"])
            seed = _decode_gray(masks[key]["data"], masks[key]["path"])
            if obs.shape != seed.shape:
                raise BatchFailure(f"{key}: observed/mask shape mismatch {obs.shape} vs {seed.shape}")

            # Build A1 once, then pass those exact arrays to the same A5 operator path used live.
            v = canonical_v78(obs, seed)
            result = operator.process_arrays(
                v["channels"], v["a1"], authoritative_mask=v["mask"], batch=512
            )
            out = pages_dir / key
            out.mkdir()
            _write_png(out / "observed.png", obs)
            _write_png(out / "reviewed_authoritative_seed_mask.png", v["seed"])
            _write_png(out / "prepared_dilation2_mask.png", v["mask"])
            _write_png(out / "telea.png", v["telea"])
            _write_png(out / "a1_geometry_t80.png", v["a1"])
            _write_png(out / "a5_v1_corrected.png", result["corrected"])
            _write_png(out / "a5_print_mask.png", result["print_mask"].astype(np.uint8) * 255)
            _write_png(out / "a5_delta_from_a1.png", (result["corrected"] != v["a1"]).astype(np.uint8) * 255)
            np.save(out / "a5_probability.npy", result["probability"])
            np.save(out / "a5_provisional_decision.npy", result["provisional"])
            np.save(out / "a5_final_decision.npy", result["final"])

            audit = {
                "page": key,
                "observed_source": observed[key]["path"],
                "mask_source": masks[key]["path"],
                "observed_sha256": observed[key]["sha256"],
                "mask_sha256": masks[key]["sha256"],
                "shape": list(obs.shape),
                "eligible_pixels": int(result["eligible"].sum()),
                "provisional_print": int(np.sum(result["provisional"] == "PRINT")),
                "final_print": int(np.sum(result["final"] == "PRINT")),
                "run_demotions": int(np.sum((result["provisional"] == "PRINT") & (result["final"] != "PRINT"))),
                "a5_changed_from_a1": int(np.sum(result["corrected"] != v["a1"])),
                "outside_mask_drift": int(np.sum(result["corrected"][~(v["mask"] > 0)] != obs[~(v["mask"] > 0)])),
                "non_print_a1_regression": int(np.sum(result["corrected"][~result["print_mask"]] != v["a1"][~result["print_mask"]])),
                "changed_byte_not_equal_observed": int(np.sum((result["corrected"] != v["a1"]) & (result["corrected"] != obs))),
            }
            if audit["outside_mask_drift"] or audit["non_print_a1_regression"] or audit["changed_byte_not_equal_observed"]:
                raise BatchFailure(f"{key}: A5 integrity contract failed: {audit}")
            (out / "page_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
            summary_rows.append(audit)

        summary_path = run / "batch_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

        manifest = {
            "version": "HF-04.5",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pages": len(summary_rows),
            "policy_id": operator.policy.get("policy_id"),
            "class_contract": {"model_output_0": "ANNOTATION", "model_output_1": "PRINT", "probability": "p_print"},
            "implementation": "canonical_v78 once per page; A5Operator.process_arrays; page-global run filter",
            "input_zip_sha256": _sha256_bytes(upload),
            "summary_sha256": _sha256_file(summary_path),
        }
        (run / "batch_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

        zip_path = Path(output_root) / f"{run_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(run.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(run))
        return zip_path
    except Exception:
        # Preserve the run directory for forensic review, but never return a partial ZIP.
        raise
