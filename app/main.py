from pathlib import Path
from datetime import datetime, timezone
import io
import json
import shutil
import zipfile

import cv2
import numpy as np
import torch
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
)
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import store, yolo
from .a1_operator import A1Operator
from .a5_n88_operator import A5N88Config, run_a5_n88
from .final_evidence_upload import commit_final_approval_evidence


# ---------------------------------------------------------------------------
# Application identity and paths
# ---------------------------------------------------------------------------

APP_VERSION = "4.7.0-alpha-n88"
A1_POLICY_ID = "Notebook87-A1-from-frozen-V8.9"
V78_POLICY_ID = "V7.8-geometry-t80"
A5_N88_POLICY_ID = "Notebook88-contextual-safe-cleanup-alpha"

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "frozen" / "yolo_detector" / "best.pt"
A5_N88_BUNDLE_DIR = ROOT / "frozen" / "a5_notebook88"
BATCH_OUTPUT_ROOT = ROOT / "batch_runs"
STARTUP_EVIDENCE_DIR = ROOT / "startup_evidence"

BATCH_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

_model = None
_a1_operator = None
_a5_n88_config = None

app = FastAPI(title="Music Annotation Project Alpha - A5-N88")
app.mount(
    "/static",
    StaticFiles(directory=ROOT / "app" / "static"),
    name="static",
)

# A5-N88 is the Alpha correction stage. It cleans selected Notebook 87 A1
# pixels to the frozen local-paper estimate. A1 remains the exact fallback.
A5_N88_APPLY_TO_OPERATIONAL_OUTPUT = True
OPERATIONAL_OUTPUT_POLICY_ID = (
    A5_N88_POLICY_ID
    if A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
    else A1_POLICY_ID
)
CORRECTION_CONTRACT = (
    "frozen-v8.9-notebook87-a1-plus-notebook88-safe-cleanup-alpha"
)
FINAL_PACKAGE_FILES = (
    "annotated_page.png",
    "reviewed_authoritative_seed_mask.png",
    "a1_corrected.png",
    "a5_n88_corrected.png",
    "operational_corrected.png",
    "a5_n88_decision_overlay.png",
    "a5_n88_live_pixel_diagnostics.csv",
    "a5_n88_final_clean_diagnostics.csv",
    "a5_n88_diagnostic_export_manifest.json",
    "operational_correction_audit.json",
    "whole_page_review.json",
    "training_evidence_upload.json",
    "package_manifest.json",
)
A5_N88_CHANNEL_ORDER = (
    "obs", "norm", "mask", "depth", "ambiguous",
    "p_annotation", "p_print_value", "p_print_available",
    "edge", "horizontal", "vertical", "compact",
)
# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class Stroke(BaseModel):
    mode: str
    brush: float
    points: list[list[float]]
    view_width: int
    view_height: int


class Approval(BaseModel):
    status: str
    note: str = ""

    mask_status: str | None = None
    a1_status: str | None = None
    a5_n88_status: str | None = None

    a5_n88_failure_stage: str | None = None
    a5_n88_failure_type: str | None = None


# ---------------------------------------------------------------------------
# Dependency loaders
# ---------------------------------------------------------------------------

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _model

    if _model is None:
        if not MODEL.exists():
            raise RuntimeError(
                "Missing frozen/yolo_detector/best.pt"
            )

        _model = yolo.load(MODEL)

    return _model

def get_a1_operator() -> A1Operator:
    global _a1_operator

    if _a1_operator is None:
        _a1_operator = A1Operator(
            device=get_device(),
        )

    return _a1_operator

def get_a5_n88_config() -> A5N88Config:
    global _a5_n88_config
    if _a5_n88_config is None:
        _a5_n88_config = A5N88Config(
            bundle_dir=A5_N88_BUNDLE_DIR,
            device=get_device(),
            clean_threshold=0.995,
            minimum_clean_run=2,
            ensemble_reducer="mean",
            channel_names=A5_N88_CHANNEL_ORDER,
        )
    return _a5_n88_config

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def binary_u8(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def require_image(
    value: np.ndarray | None,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"Missing image result: {name}")

    value = np.asarray(value)

    if value.ndim != 2:
        raise RuntimeError(
            f"{name} must be a two-dimensional greyscale array; "
            f"found shape {value.shape}"
        )

    if expected_shape is not None and value.shape != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: "
            f"expected {expected_shape}, found {value.shape}"
        )

    return value


def require_bool_mask(
    value: np.ndarray | None,
    name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    value = require_image(value, name, expected_shape)
    return value.astype(bool)


def write_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image)

    if image.dtype == bool:
        image = binary_u8(image)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def png(image: np.ndarray) -> Response:
    image = np.asarray(image)

    if image.ndim not in (2, 3):
        raise RuntimeError(
            "PNG response image must be either "
            "2-D greyscale or 3-D colour; "
            f"found shape {image.shape}"
        )

    if image.ndim == 3:
        if image.shape[2] not in (3, 4):
            raise RuntimeError(
                "Colour PNG response must have "
                "3 BGR channels or 4 BGRA channels; "
                f"found shape {image.shape}"
            )

    if image.dtype != np.uint8:
        if not np.issubdtype(
            image.dtype,
            np.number,
        ):
            raise RuntimeError(
                "PNG response image must contain "
                f"numeric values; found {image.dtype}"
            )

        if not np.isfinite(image).all():
            raise RuntimeError(
                "PNG response image contains "
                "non-finite values"
            )

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    image = np.ascontiguousarray(image)

    ok, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not ok:
        raise RuntimeError(
            "PNG encoding failed"
        )

    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
        },
    )


def read_session_png(
    sid: str,
    filename: str,
    colour: bool = False,
) -> np.ndarray:
    path = store.folder(sid) / filename

    flags = (
        cv2.IMREAD_COLOR
        if colour
        else cv2.IMREAD_GRAYSCALE
    )

    image = cv2.imread(
        str(path),
        flags,
    )

    if image is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session artefact not found: "
                f"{filename}"
            ),
        )

    return image

def a5_n88_decision_overlay(
    observed: np.ndarray,
    clean_mask: np.ndarray,
    eligible_mask: np.ndarray,
    alpha: float = 0.55,
    display_radius: int = 1,
) -> np.ndarray:
    """A5-N88 overlay: CLEAN green; eligible fallback yellow."""
    observed = require_image(observed, "A5-N88 overlay observed page")
    clean_mask = require_bool_mask(clean_mask, "A5-N88 CLEAN mask", observed.shape)
    eligible_mask = require_bool_mask(eligible_mask, "A5-N88 eligibility", observed.shape)
    keep_mask = eligible_mask & ~clean_mask
    if display_radius > 0:
        kernel = np.ones((2 * display_radius + 1, 2 * display_radius + 1), np.uint8)
        clean_display = cv2.dilate(clean_mask.astype(np.uint8), kernel) > 0
        keep_display = cv2.dilate(keep_mask.astype(np.uint8), kernel) > 0
    else:
        clean_display, keep_display = clean_mask, keep_mask
    base = cv2.cvtColor(observed, cv2.COLOR_GRAY2BGR)
    colour = base.copy()
    colour[clean_display] = (0, 255, 0)
    colour[keep_display] = (0, 255, 255)
    active = clean_display | keep_display
    output = base.copy()
    blended = cv2.addWeighted(base, 1.0 - alpha, colour, alpha, 0)
    output[active] = blended[active]
    return output

# ---------------------------------------------------------------------------
# Operational result validation
# ---------------------------------------------------------------------------

def validate_operational_result(observed: np.ndarray, result: dict) -> dict:
    required = {
        "mask", "telea", "paper", "routes", "geometry_t80",
        "v8_9_corrected", "a1", "a5_n88", "eligible_a5_n88",
        "cleanup_probability", "clean_mask",
    }
    missing = sorted(required - set(result))
    if missing:
        raise RuntimeError(f"Operational result omitted required fields: {missing}")
    shape = observed.shape
    mask = require_bool_mask(result["mask"], "authoritative mask", shape)
    paper = require_image(result["paper"], "paper estimate", shape)
    a1 = require_image(result["a1"], "Notebook 87 A1", shape)
    a5_n88 = require_image(result["a5_n88"], "A5-N88 corrected page", shape)
    eligible = require_bool_mask(result["eligible_a5_n88"], "A5-N88 eligibility", shape)
    clean_mask = require_bool_mask(result["clean_mask"], "A5-N88 CLEAN mask", shape)
    routes = result["routes"]
    ambiguous = require_bool_mask(routes["ambiguous"], "V7.8 AMBIGUOUS", shape)
    if not np.array_equal(eligible, mask & ambiguous):
        raise RuntimeError("A5-N88 eligibility is not exactly authoritative AMBIGUOUS")
    if np.any(clean_mask & ~eligible):
        raise RuntimeError("A5-N88 CLEAN exists outside eligibility")
    if np.any(a5_n88[clean_mask] != paper[clean_mask]):
        raise RuntimeError("A5-N88 CLEAN output does not equal paper estimate")
    if np.any(a5_n88[~clean_mask] != a1[~clean_mask]):
        raise RuntimeError("A5-N88 non-CLEAN output differs from Notebook 87 A1")
    if np.any(a5_n88[~mask] != observed[~mask]):
        raise RuntimeError("A5-N88 changed pixels outside authority")
    probability = np.asarray(result["cleanup_probability"], dtype=np.float32)
    if probability.shape != shape:
        raise RuntimeError("A5-N88 cleanup probability shape mismatch")
    if eligible.any() and not np.isfinite(probability[eligible]).all():
        raise RuntimeError("A5-N88 cleanup probability is non-finite on eligibility")
    operational = a5_n88.copy() if A5_N88_APPLY_TO_OPERATIONAL_OUTPUT else a1.copy()
    result.update({
        "mask": mask, "paper": paper, "a1": a1, "a5_n88": a5_n88,
        "eligible_a5_n88": eligible, "clean_mask": clean_mask,
        "cleanup_probability": probability, "operational": operational,
        "operational_stage": "A5_N88" if A5_N88_APPLY_TO_OPERATIONAL_OUTPUT else "NOTEBOOK87_A1",
        "a5_n88_applied_to_operational_output": bool(A5_N88_APPLY_TO_OPERATIONAL_OUTPUT),
    })
    return result

# ---------------------------------------------------------------------------
# UI endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "app" / "templates" / "index.html").read_text()


@app.post("/api/session")
async def create(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        page = np.array(
            Image.open(io.BytesIO(raw)).convert("L")
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unreadable image upload: {exc}",
        ) from exc

    confidence = yolo.confidence_map(page, get_model())
    draft = (confidence >= yolo.CONF).astype(np.uint8) * 255

    sid = store.create(
        page,
        confidence,
        draft,
        file.filename or "upload",
    )

    session = store.load(sid)

    if session["gray"].shape != session["working"].shape:
        raise RuntimeError(
            "Session creation corrupted mask dimensions: "
            f"gray={session['gray'].shape}, "
            f"working={session['working'].shape}"
        )

    if not np.array_equal(
        session["working"] > 0,
        draft > 0,
    ):
        raise RuntimeError(
            "Session working mask differs from YOLO draft "
            "immediately after session creation"
        )

    directory = store.folder(sid)

    write_png(directory / "annotated_page.png", page)
    # Revision 000 is required by the interactive undo workflow.
    # The duplicate YOLO draft PNG and confidence NPY are not persisted.
    write_png(directory / "mask_revision_000.png", draft)

    return {
        "session_id": sid,
        "image_url": f"/api/session/{sid}/image",
        "overlay_url": f"/api/session/{sid}/overlay?rev=0",
    }


@app.get("/api/session/{sid}/image")
def image(sid: str):
    return png(store.load(sid)["gray"])


def overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray)
    mask = np.asarray(mask)

    if gray.ndim != 2:
        raise RuntimeError(
            f"Overlay gray must be 2-D; got "
            f"shape={gray.shape}, dtype={gray.dtype}"
        )

    if mask.ndim != 2:
        raise RuntimeError(
            f"Overlay mask must be 2-D; got "
            f"shape={mask.shape}, dtype={mask.dtype}"
        )

    if gray.shape != mask.shape:
        raise RuntimeError(
            f"Overlay gray/mask shape mismatch: "
            f"gray={gray.shape}, mask={mask.shape}"
        )

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    active = mask > 0

    rgb = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    colour = rgb.copy()

    # Red overlay in OpenCV BGR order.
    colour[active] = (60, 60, 255)

    result = cv2.addWeighted(
        rgb,
        0.62,
        colour,
        0.38,
        0,
    )

    return result

@app.get("/api/session/{sid}/overlay")
def session_overlay(sid: str, rev: int = 0):
    try:
        session = store.load(sid)

        if "gray" not in session:
            raise RuntimeError(
                "Session has no 'gray' array"
            )

        if "working" not in session:
            raise RuntimeError(
                "Session has no 'working' mask"
            )

        return png(
            overlay(
                session["gray"],
                session["working"],
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Overlay generation failed",
                "session_id": sid,
                "revision": rev,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc


@app.post("/api/session/{sid}/stroke")
def stroke(sid: str, stroke_data: Stroke):
    session = store.load(sid)
    height, width = session["working"].shape

    sx = width / max(stroke_data.view_width, 1)
    sy = height / max(stroke_data.view_height, 1)

    points = [
        (
            round(point[0] * sx),
            round(point[1] * sy),
        )
        for point in stroke_data.points
    ]

    brush = max(
        1,
        round(stroke_data.brush * (sx + sy) / 2),
    )

    stroke_mask = np.zeros((height, width), np.uint8)

    if len(points) == 1:
        cv2.circle(
            stroke_mask,
            points[0],
            max(1, brush // 2),
            255,
            -1,
        )
    else:
        for start, end in zip(points, points[1:]):
            cv2.line(
                stroke_mask,
                start,
                end,
                255,
                brush,
                cv2.LINE_AA,
            )

    if stroke_data.mode == "add":
        session["working"][stroke_mask > 0] = 255
    elif stroke_data.mode == "erase":
        session["working"][stroke_mask > 0] = 0
    else:
        raise HTTPException(
            status_code=400,
            detail="Stroke mode must be 'add' or 'erase'",
        )

    store.save(sid, session)

    audit = store.audit(sid)
    audit["revision"] = int(audit.get("revision", 0)) + 1
    audit["operation_sequence"] = (
        int(audit.get("operation_sequence", 0)) + 1
    )

    revision = audit["revision"]

    audit.setdefault("events", []).append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "revision": revision,
            "operation": stroke_data.mode,
            "points": len(points),
        }
    )

    store.put_audit(sid, audit)

    write_png(
        store.folder(sid)
        / f"mask_revision_{revision:03d}.png",
        session["working"],
    )

    return {
        "revision": revision,
        "overlay_url": (
            f"/api/session/{sid}/overlay?rev={revision}"
        ),
    }


@app.post("/api/session/{sid}/undo")
def undo(sid: str):
    audit = store.audit(sid)
    current_revision = int(audit.get("revision", 0))

    if current_revision <= 0:
        return {
            "revision": 0,
            "overlay_url": f"/api/session/{sid}/overlay?rev=0",
        }

    revision = current_revision - 1
    path = (
        store.folder(sid)
        / f"mask_revision_{revision:03d}.png"
    )

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise HTTPException(
            status_code=409,
            detail=f"Mask revision {revision} is unavailable",
        )

    session = store.load(sid)
    session["working"] = mask
    store.save(sid, session)

    audit["revision"] = revision
    audit["operation_sequence"] = (
        int(audit.get("operation_sequence", 0)) + 1
    )
    audit.setdefault("events", []).append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "revision": revision,
            "operation": "undo",
        }
    )

    store.put_audit(sid, audit)

    return {
        "revision": revision,
        "overlay_url": (
            f"/api/session/{sid}/overlay?rev={revision}"
        ),
    }

@app.get("/api/session/{sid}/mask")
def session_mask(sid: str):
    session = store.load(sid)

    mask = np.asarray(
        session["working"]
    )

    if mask.ndim != 2:
        raise HTTPException(
            500,
            f"Working mask shape is {mask.shape}",
        )

    return png(
        np.where(
            mask > 0,
            255,
            0,
        ).astype(np.uint8)
    )

# ---------------------------------------------------------------------------
# Correction authorisation
# ---------------------------------------------------------------------------

@app.post("/api/session/{sid}/authorise")
def authorise(sid: str):
    session = store.load(sid)
    directory = store.folder(sid)
    observed = require_image(session["gray"], "observed session page")
    reviewed_seed = require_image(session["working"], "reviewed authoritative seed", observed.shape)
    write_png(directory / "reviewed_authoritative_seed_mask.png", reviewed_seed)
    try:
        a1_result = get_a1_operator().run(
            observed=observed,
            reviewed_seed=reviewed_seed,
            page=sid,
        )
        mask = require_bool_mask(a1_result["mask"], "authoritative mask", observed.shape)
        routes = a1_result["routes"]
        ambiguous = require_bool_mask(routes["ambiguous"], "V7.8 AMBIGUOUS", observed.shape)
        horizontal = require_bool_mask(routes["horizontal"], "horizontal support", observed.shape)
        vertical = require_bool_mask(routes["vertical"], "vertical support", observed.shape)
        compact = require_bool_mask(routes["compact"], "compact support", observed.shape)
        paper = require_image(a1_result["paper"], "paper estimate", observed.shape)
        a1 = require_image(a1_result["a1"], "Notebook 87 A1", observed.shape)
        p_annotation = np.asarray(session["confidence"], dtype=np.float32)
        if p_annotation.shape != observed.shape:
            raise RuntimeError("YOLO confidence map shape mismatch")
        n88 = run_a5_n88(
            observed=observed,
            a1_output=a1,
            authoritative_mask=mask,
            ambiguous_mask=ambiguous,
            paper_estimate=paper,
            horizontal_support=horizontal,
            vertical_support=vertical,
            compact_support=compact,
            p_annotation=p_annotation,
            p_print_value=np.zeros(observed.shape, np.float32),
            p_print_available=np.zeros(observed.shape, np.float32),
            eligible_mask=mask & ambiguous,
            config=get_a5_n88_config(),
        )
        result = dict(a1_result)
        result.update({
            "a5_n88": n88.output,
            "eligible_a5_n88": n88.eligible_mask,
            "cleanup_probability": n88.cleanup_probability,
            "fold_probability_min": n88.fold_probability_min,
            "fold_probability_max": n88.fold_probability_max,
            "fold_probability_std": n88.fold_probability_std,
            "clean_mask": n88.clean_mask,
            "a5_n88_diagnostics": n88.diagnostics,
            "a5_n88_fold_files": n88.fold_files,
        })
        result = validate_operational_result(observed, result)
        operational = result["operational"]
        a5_n88 = result["a5_n88"]
        clean_mask = result["clean_mask"]
        eligible = result["eligible_a5_n88"]
        probability = result["cleanup_probability"]
        overlay_image = a5_n88_decision_overlay(observed, clean_mask, eligible)
        write_png(directory / "a1_corrected.png", a1)
        write_png(directory / "a5_n88_corrected.png", a5_n88)
        write_png(directory / "operational_corrected.png", operational)
        write_png(directory / "a5_n88_decision_overlay.png", overlay_image)

        import csv
        all_path = directory / "a5_n88_live_pixel_diagnostics.csv"
        clean_path = directory / "a5_n88_final_clean_diagnostics.csv"
        fields = ["y", "x", "cleanup_probability", "clean_decision", "yolo_confidence", "observed_byte", "a1_byte", "paper_byte", "a5_n88_byte"]
        ys, xs = np.where(eligible)
        clean_rows = 0
        with all_path.open("w", newline="", encoding="utf-8") as ah, clean_path.open("w", newline="", encoding="utf-8") as ch:
            aw, cw = csv.DictWriter(ah, fieldnames=fields), csv.DictWriter(ch, fieldnames=fields)
            aw.writeheader(); cw.writeheader()
            for y, x in zip(ys, xs):
                row = {
                    "y": int(y), "x": int(x),
                    "cleanup_probability": float(probability[y, x]),
                    "clean_decision": bool(clean_mask[y, x]),
                    "yolo_confidence": float(p_annotation[y, x]),
                    "observed_byte": int(observed[y, x]),
                    "a1_byte": int(a1[y, x]),
                    "paper_byte": int(paper[y, x]),
                    "a5_n88_byte": int(a5_n88[y, x]),
                }
                aw.writerow(row)
                if clean_mask[y, x]:
                    cw.writerow(row); clean_rows += 1
        diagnostic_manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "policy_id": A5_N88_POLICY_ID,
            "page_shape": list(observed.shape),
            "channel_order": A5_N88_CHANNEL_ORDER,
            "ensemble_reducer": "mean",
            "clean_threshold": 0.995,
            "minimum_clean_run": 2,
            "eligible_rows": int(eligible.sum()),
            "final_clean_rows": int(clean_rows),
            "fold_files": list(n88.fold_files),
            "diagnostics": n88.diagnostics,
        }
        (directory / "a5_n88_diagnostic_export_manifest.json").write_text(json.dumps(diagnostic_manifest, indent=2), encoding="utf-8")
        audit = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": sid,
            "app_version": APP_VERSION,
            "correction_contract": CORRECTION_CONTRACT,
            "v7_8_policy_id": V78_POLICY_ID,
            "a1_policy_id": result.get("a1_policy_id", A1_POLICY_ID),
            "a5_n88_policy_id": A5_N88_POLICY_ID,
            "a5_n88_action": "REPLACE_SELECTED_A1_PIXEL_WITH_LOCAL_PAPER_ESTIMATE",
            "a5_n88_fallback": "Notebook87 A1",
            "operational_output_stage": result["operational_stage"],
            "a5_n88_applied_to_operational_output": bool(A5_N88_APPLY_TO_OPERATIONAL_OUTPUT),
            "prepared_mask_pixels": int(mask.sum()),
            "ambiguous_pixels": int(ambiguous.sum()),
            "a5_n88_eligible_pixels": int(eligible.sum()),
            "a5_n88_final_clean_pixels": int(clean_mask.sum()),
            "a5_n88_changed_from_a1_pixels": int(np.sum(a5_n88 != a1)),
            "a5_n88_outside_mask_drift": int(np.sum(a5_n88[~mask] != observed[~mask])),
            "a5_n88_non_clean_a1_regression": int(np.sum(a5_n88[~clean_mask] != a1[~clean_mask])),
            "a5_n88_clean_not_equal_paper": int(np.sum(a5_n88[clean_mask] != paper[clean_mask])),
            "clean_image_used_at_runtime": False,
            "notebook88_alpha_test": True,
        }
        zero_fields = ["a5_n88_outside_mask_drift", "a5_n88_non_clean_a1_regression", "a5_n88_clean_not_equal_paper"]
        failures = {k: audit[k] for k in zero_fields if audit[k] != 0}
        if failures:
            raise RuntimeError("A5-N88 correction contract failed: " + json.dumps(failures, sort_keys=True))
        (directory / "operational_correction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    except Exception as exc:
        failure = {"utc": datetime.now(timezone.utc).isoformat(), "session_id": sid, "contract": CORRECTION_CONTRACT, "error_type": type(exc).__name__, "error": str(exc)}
        (directory / "correction_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise HTTPException(status_code=500, detail=f"A5-N88 correction failed: {type(exc).__name__}: {exc}") from exc
    return {
        "ok": True,
        "corrected_url": f"/api/session/{sid}/result",
        "operational_stage": audit["operational_output_stage"],
        "a5_n88_url": f"/api/session/{sid}/a5-n88",
        "original_url": f"/api/session/{sid}/image",
        "diagnostics": {
            "prepared_mask_pixels": audit["prepared_mask_pixels"],
            "ambiguous_pixels": audit["ambiguous_pixels"],
            "a5_n88_eligible_pixels": audit["a5_n88_eligible_pixels"],
            "a5_n88_clean_pixels": audit["a5_n88_final_clean_pixels"],
            "a5_n88_changed_from_a1": audit["a5_n88_changed_from_a1_pixels"],
            "outside_mask_drift": audit["a5_n88_outside_mask_drift"],
            "non_clean_a1_regression": audit["a5_n88_non_clean_a1_regression"],
        },
    }

@app.get("/api/session/{sid}/result")
def result(sid: str):
    return png(
        read_session_png(
            sid,
            "operational_corrected.png",
        )
    )

@app.get("/api/session/{sid}/a5-n88")
def a5_n88_result(sid: str):
    return png(
        read_session_png(
            sid,
            "a5_n88_corrected.png",
        )
    )

@app.get("/api/session/{sid}/diagnostic/{name}")
def diagnostic(
    sid: str,
    name: str,
):
    names = {
        "a1": "a1_corrected.png",
        "operational": "operational_corrected.png",
        "a5_n88": "a5_n88_corrected.png",
        "a5_n88": "a5_n88_corrected.png",
        "decisions": "a5_n88_decision_overlay.png",
        "m5": "operational_corrected.png",
    }
    filename = names.get(name)

    if filename is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown diagnostic",
        )

    return png(
        read_session_png(
            sid,
            filename,
            colour=(
                name == "decisions"
            ),
        )
    )


# ---------------------------------------------------------------------------
# Review and packaging
# ---------------------------------------------------------------------------

@app.post("/api/session/{sid}/approval")
def approval(
    sid: str,
    value: Approval,
    request: Request,
):
    allowed = {
        "approved",
        "return_to_mask",
        "flag_failure",
    }

    if value.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {value.status}",
        )

    review_statuses = {
        "approved",
        "rejected",
        "not_reviewed",
    }

    for field_name, field_value in (
        ("mask_status", value.mask_status),
        ("a1_status", value.a1_status),
        ("a5_n88_status", value.a5_n88_status),
    ):
        if (
            field_value is not None
            and field_value not in review_statuses
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid {field_name}: {field_value}"
                ),
            )

    directory = store.folder(sid)

    if not directory.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session: {sid}",
        )

    record = {
        "utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "session_id": sid,

        # Compatibility field.
        "status": value.status,

        "mask_status": (
            value.mask_status
            or (
                "approved"
                if value.status == "approved"
                else "rejected"
            )
        ),

        "a1_status": (
            value.a1_status
            or (
                "approved"
                if value.status == "approved"
                else "rejected"
            )
        ),

        "a5_n88_status": (
            value.a5_n88_status
            or "not_reviewed"
        ),

        "a5_n88_failure_stage": (
            value.a5_n88_failure_stage
        ),

        "a5_n88_failure_type": (
            value.a5_n88_failure_type
        ),

        "note": value.note,

        "operational_output_stage": (
            "A5_N88"
            if A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_applied_to_operational_output": bool(
            A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
        ),
    }

    (
        directory
        / "whole_page_review.json"
    ).write_text(
        json.dumps(
            record,
            indent=2,
        )
    )

    return {
        "ok": True,
        "status": value.status,

        "finish_url": (
            str(
                request.url_for(
                    "finish_session",
                    sid=sid,
                )
            )
            if value.status == "approved"
            else None
        ),

        "corrected_url": (
            f"/api/session/{sid}/result"
        ),

        "decision_overlay_url": (
            f"/api/session/{sid}/diagnostic/decisions"
        ),
    }


@app.post("/api/session/{sid}/approve")
def approve_compatibility(
    sid: str,
    value: Approval,
    request: Request,
):
    return approval(
        sid=sid,
        value=value,
        request=request,
    )


def evidence_zip_path(
    sid: str,
) -> Path:
    return (
        store.folder(sid)
        / "hf_04_5_training_and_audit.zip"
    )


@app.post(
    "/api/session/{sid}/finish",
    name="finish_session",
)
def finish(
    sid: str,
    request: Request,
):
    import csv

    import numpy as np

    directory = store.folder(sid)
    session = store.load(sid)

    review_path = (
        directory
        / "whole_page_review.json"
    )

    if not review_path.is_file():
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "The page has not received a "
                    "whole-page review decision."
                ),

                "approval_url": (
                    f"/api/session/{sid}/approval"
                ),
            },
        )

    try:
        review = json.loads(
            review_path.read_text(
                encoding="utf-8",
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "The whole-page review record "
                f"is unreadable: {exc}"
            ),
        ) from exc

    if review.get(
        "status"
    ) != "approved":
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "The page is not approved "
                    "for packaging."
                ),

                "review_status": review.get(
                    "status"
                ),
            },
        )

    required_approvals = {
        "mask_status": "approved",
        "a1_status": "approved",
    }
    approval_failures = {
        key: review.get(key)
        for key, expected in required_approvals.items()
        if review.get(key) != expected
    }
    if approval_failures:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "The operational mask and Notebook 87 A1 "
                    "output must both be approved before packaging."
                ),
                "approval_failures": approval_failures,
            },
        )

    write_png(
        directory
        / "reviewed_authoritative_seed_mask.png",
        session[
            "working"
        ],
    )

    # ------------------------------------------------------------
    # A5 diagnostic CSVs were written directly from the in-memory
    # authorisation result. No full-page NPY/NPZ intermediates are needed.
    diagnostic_manifest_path = (
        directory / "a5_diagnostic_export_manifest.json"
    )
    try:
        diagnostic_manifest = json.loads(
            diagnostic_manifest_path.read_text(encoding="utf-8")
        )
        eligible_rows = int(diagnostic_manifest["eligible_rows"])
        final_clean_rows = int(diagnostic_manifest["final_clean_rows"])
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"A5 diagnostic export is unavailable: {exc}",
        ) from exc

    required = [
        name
        for name in FINAL_PACKAGE_FILES
        if name not in {
            "training_evidence_upload.json",
            "package_manifest.json",
        }
    ]
    missing = [
        filename
        for filename in required
        if not (
            directory
            / filename
        ).exists()
    ]

    if missing:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "Correction package is incomplete. "
                    "Authorise the page before finishing."
                ),

                "missing": missing,
            },
        )

    output = evidence_zip_path(
        sid
    )

    a1_operator = get_a1_operator()
    a5_n88_config = get_a5_n88_config()

    manifest = {
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "version": APP_VERSION,

        "correction_contract": (
            CORRECTION_CONTRACT
        ),

        "training_pair": [
            "annotated_page.png",
            "reviewed_authoritative_seed_mask.png",
        ],

        "raw_yolo_is_training_truth": False,

        # --------------------------------------------------------
        # V7.8
        # --------------------------------------------------------

        "v78_policy_id": V78_POLICY_ID,

        "telea_separate_file_is_diagnostic": False,

        "telea_contributes_to_v78_ambiguous": True,

        # --------------------------------------------------------
        # Notebook 87 A1
        # --------------------------------------------------------

        "a1_policy_id": A1_POLICY_ID,

        "a1_source": (
            "frozen/v8_9/v8_9_operator.py"
        ),

        "v89_operator_sha256": getattr(
            a1_operator,
            "v89_operator_sha256",
            None,
        ),

        "a1_assembly": (
            "V8.9 corrected, then every V7.8 "
            "AMBIGUOUS coordinate restored to "
            "the exact observed byte"
        ),

        # --------------------------------------------------------
        # Operational output
        # --------------------------------------------------------

        "operational_output_policy_id": (
            OPERATIONAL_OUTPUT_POLICY_ID
        ),

        "operational_output_file": (
            "operational_corrected.png"
        ),

        "operational_output_stage": (
            "A5_N88"
            if A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_n88_mode": bool(False),

        "a5_applied_to_operational_output": bool(
            A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
        ),

        "a5_n88_proposal_file": (
            "a5_n88_corrected.png"
        ),

        # --------------------------------------------------------
        # A5-N88
        # --------------------------------------------------------

        "a5_policy_id": getattr(
            a5_n88_config,
            "a5_policy_id",
            A5_N88_POLICY_ID,
        ),

        "a5_bundle_id": getattr(
            a5_n88_config,
            "bundle_id",
            None,
        ),

        "a5_clean_action": (
            "REPLACE_WITH_LOCAL_PAPER_ESTIMATE"
        ),

        "a5_non_print_fallback": (
            "Notebook87 A1"
        ),

        "a5_raw_probability_exported": False,

        "a5_calibrated_probability_exported": False,

        "a5_channel_diagnostic_csv_exported": True,

        "a5_channel_order": (
            A5_N88_CHANNEL_ORDER
        ),

        "a5_patch_size": 33,

        # --------------------------------------------------------
        # Global integrity
        # --------------------------------------------------------

        "outside_authoritative_mask": (
            "IMMUTABLE_OBSERVED"
        ),

        "clean_image_used_at_runtime": False,

        "historical_notebook87_is_operational": True,

        "historical_notebook91_is_operational": False,

        "files": [],
    }

    for filename in FINAL_PACKAGE_FILES:
        if filename in {
            "training_evidence_upload.json",
            "package_manifest.json",
        }:
            continue
        path = directory / filename
        if path.is_file():
            manifest["files"].append(
                {"name": path.name, "bytes": path.stat().st_size}
            )

    # ------------------------------------------------------------
    # Durable evidence upload
    #
    # Build the manifest dictionary first, but do not write the
    # package manifest or ZIP until the remote commitment receipt
    # has been created.
    # ------------------------------------------------------------

    evidence_review = {
        "mask_status": review[
            "mask_status"
        ],

        "correction_status": review.get(
            "a1_status"
        ),

        "a1_status": review.get(
            "a1_status"
        ),

        "a5_n88_status": review.get(
            "a5_n88_status"
        ),

        "failure_stage": review.get(
            "a5_n88_failure_stage"
        ),

        "failure_type": review.get(
            "a5_n88_failure_type"
        ),

        "notes": review.get(
            "note",
            "",
        ),

        "operational_output_stage": (
            "A5_N88"
            if A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_applied_to_operational_output": bool(
            A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
        ),
    }

    training_evidence = (
        commit_final_approval_evidence(
            directory=directory,
            sid=sid,
            review=evidence_review,
            package_manifest=manifest,
        )
    )

    receipt_path = (
        directory
        / "training_evidence_upload.json"
    )

    if not receipt_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=(
                "Evidence upload returned without "
                "creating training_evidence_upload.json."
            ),
        )

    # ------------------------------------------------------------
    # Rebuild the manifest file list after the receipt exists.
    # ------------------------------------------------------------

    manifest[
        "training_evidence"
    ] = training_evidence

    manifest[
        "files"
    ] = []

    for filename in FINAL_PACKAGE_FILES:
        if filename == "package_manifest.json":
            continue
        path = directory / filename
        if path.is_file():
            manifest["files"].append(
                {"name": path.name, "bytes": path.stat().st_size}
            )

    package_manifest_path = (
        directory
        / "package_manifest.json"
    )

    package_manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Create the local package only after the remote receipt has
    # been included in package_manifest.json.
    # ------------------------------------------------------------

    with zipfile.ZipFile(
        output,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        missing_package_files = [
            filename
            for filename in FINAL_PACKAGE_FILES
            if not (directory / filename).is_file()
        ]
        if missing_package_files:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Curated evidence package is incomplete.",
                    "missing": missing_package_files,
                },
            )
        for filename in FINAL_PACKAGE_FILES:
            path = directory / filename
            archive.write(path, filename)

    if not output.is_file():
        raise HTTPException(
            status_code=500,
            detail=(
                "Evidence package creation completed "
                "without producing the ZIP file."
            ),
        )

    if output.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail=(
                "Evidence package creation produced "
                "an empty ZIP file."
            ),
        )

    download_url = request.url_for(
        "download_evidence_zip",
        sid=sid,
    )

    return {
        "ok": True,

        "package_created": True,

        "package_filename": (
            output.name
        ),

        "package_bytes": (
            output.stat().st_size
        ),

        "operational_output_stage": (
            "A5_N88"
            if A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_n88_mode": bool(False),

        "a5_applied_to_operational_output": bool(
            A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
        ),

        "training_evidence": (
            training_evidence
        ),

        "a5_eligible_diagnostic_rows": int(
            eligible_rows
        ),

        "a5_final_clean_diagnostic_rows": int(
            final_clean_rows
        ),

        "download_url": str(
            download_url
        ),
    }



@app.get(
    "/api/session/{sid}/download.zip",
    name="download_evidence_zip",
)
def download_evidence_zip(
    sid: str,
):
    path = evidence_zip_path(sid)

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    "Finish package has not been created"
                ),

                "session_id": sid,

                "expected_path": str(
                    path
                ),
            },
        )

    if path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "The evidence ZIP exists but is empty"
                ),

                "session_id": sid,
            },
        )

    return FileResponse(
        path=str(path),

        media_type="application/zip",

        filename=path.name,

        headers={
            "Cache-Control": "no-store",
        },
    )


@app.get(
    "/api/session/{sid}/download"
)
def download_evidence_zip_compatibility(
    sid: str,
):
    return download_evidence_zip(
        sid
    )


# ---------------------------------------------------------------------------
# Health and startup evidence
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    a1_operator = get_a1_operator()
    a5_n88_config = get_a5_n88_config()

    return {
        "ok": True,

        "version": APP_VERSION,

        "correction_contract": (
            CORRECTION_CONTRACT
        ),

        "device": get_device(),

        # --------------------------------------------------------
        # V7.8
        # --------------------------------------------------------

        "v78_policy_id": (
            V78_POLICY_ID
        ),

        "telea_operational": True,

        "telea_role": (
            "Frozen V7.8 AMBIGUOUS route before "
            "Notebook 87 A1 restoration"
        ),

        # --------------------------------------------------------
        # Notebook 87 A1
        # --------------------------------------------------------

        "a1_policy_id": getattr(
            a1_operator,
            "a1_policy_id",
            A1_POLICY_ID,
        ),

        "v89_operator_sha256": getattr(
            a1_operator,
            "v89_operator_sha256",
            None,
        ),

        "a1_source": (
            "Frozen V8.9 plus Notebook 87 "
            "AMBIGUOUS-to-observed assembly"
        ),

        "operational_output_policy_id": (
            OPERATIONAL_OUTPUT_POLICY_ID
        ),

        "operational_output_stage": (
            "NOTEBOOK87_A1"
            if not A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
            else "A5_N88"
        ),

        "a5_n88_mode": bool(
            False
        ),

        "a5_applied_to_operational_output": bool(
            A5_N88_APPLY_TO_OPERATIONAL_OUTPUT
        ),

        # --------------------------------------------------------
        # A5-N88
        # --------------------------------------------------------

        "a5_policy_id": getattr(
            a5_n88_config,
            "a5_policy_id",
            A5_N88_POLICY_ID,
        ),

        "bundle_id": getattr(
            a5_n88_config,
            "bundle_id",
            None,
        ),

        "a5_clean_action": (
            "REPLACE_WITH_LOCAL_PAPER_ESTIMATE"
        ),

        "a5_non_print_fallback": (
            "Notebook87 A1"
        ),

        # --------------------------------------------------------
        # Historical identity
        # --------------------------------------------------------

        "historical_notebook87_operational": True,

        "historical_notebook91_operational": False,
    }


@app.get("/startup-evidence")
def download_startup_evidence():
    root = STARTUP_EVIDENCE_DIR

    verdict = (
        root
        / "startup_verdict.json"
    )

    a5_replay = (
        root
        / "ardross_a5_replay"
        / "replay_result.json"
    )

    v78_replay = (
        root
        / "v78_routing_equivalence"
        / "replay_result.json"
    )

    v78_summary = (
        root
        / "v78_routing_equivalence"
        / "summary.json"
    )

    v78_csv = (
        root
        / "v78_routing_equivalence"
        / "channel_comparison.csv"
    )

    runtime = (
        root
        / "runtime_software_identities.json"
    )

    required = [
        verdict,
        v78_replay,
        a5_replay,
        v78_summary,
        v78_csv,
        runtime,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Startup commissioning evidence "
                    "incomplete"
                ),

                "missing": missing,
            },
        )

    value = json.loads(
        verdict.read_text()
    )

    if value.get("passed") is not True:
        raise HTTPException(
            status_code=503,
            detail=(
                "Startup commissioning verdict "
                "is not PASS"
            ),
        )

    zip_base = Path(
        "/tmp/hf_04_5_startup_evidence"
    )

    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=root,
        )
    )

    return FileResponse(
        path=zip_path,

        media_type="application/zip",

        filename=(
            "hf_04_5_startup_evidence.zip"
        ),
    )


@app.get("/startup-evidence-status")
def startup_evidence_status():
    root = STARTUP_EVIDENCE_DIR

    return {
        "root": str(
            root.resolve()
        ),

        "root_exists": root.exists(),

        "files": (
            sorted(
                str(
                    path.relative_to(root)
                )
                for path in root.rglob("*")
                if path.is_file()
            )
            if root.exists()
            else []
        ),
    }


# ---------------------------------------------------------------------------
# HF-04.5.1 batch
#
# NOTE:
# This endpoint assumes hf045_batch.run_batch() exposes:
#
#     run_batch(
#         upload=bytes,
#         a1_operator=A1Operator,
#         operator=A5N88Config,
#         output_root=Path,
#     )
#
# If hf045_batch.py has not yet been changed to this interface,
# update that file before enabling batch processing.
# ---------------------------------------------------------------------------

@app.post("/batch-v89-a1-a5-n88", response_class=FileResponse)
async def batch_v89_a1_a5_n88(file: UploadFile = File(...)):
    await file.close()
    raise HTTPException(
        status_code=501,
        detail=(
            "Batch processing is disabled in the first A5-N88 Alpha build. "
            "hf045_batch.py still implements the the legacy operator contract "
            "and must be converted separately before this endpoint is enabled."
        ),
    )
