from pathlib import Path
from collections.abc import Mapping
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
from .a5_operator import A5Operator
from .hf045_batch import BatchFailure, run_batch
from .final_evidence_upload import commit_final_approval_evidence


# ---------------------------------------------------------------------------
# Application identity and paths
# ---------------------------------------------------------------------------

APP_VERSION = "4.5.2"
A1_POLICY_ID = "Notebook87-A1-from-frozen-V8.9"
V78_POLICY_ID = "V7.8-geometry-t80"
A5_POLICY_ID = "A5-v1-print-p0980-margin0960-run2"

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "frozen" / "yolo_detector" / "best.pt"
A5_BUNDLE_DIR = ROOT / "frozen" / "a5_v1"
BATCH_OUTPUT_ROOT = ROOT / "batch_runs"
STARTUP_EVIDENCE_DIR = ROOT / "startup_evidence"

BATCH_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

_model = None
_a1_operator = None
_correction_operator = None

app = FastAPI(title="HF-04.5 Music Annotation")
app.mount(
    "/static",
    StaticFiles(directory=ROOT / "app" / "static"),
    name="static",
)

# A5 still runs and produces complete diagnostics.
A5_SHADOW_MODE = True

# Must remain False until a later A5 challenger passes the required gates.
A5_APPLY_TO_OPERATIONAL_OUTPUT = False

OPERATIONAL_OUTPUT_POLICY_ID = (
    A5_POLICY_ID
    if A5_APPLY_TO_OPERATIONAL_OUTPUT
    else A1_POLICY_ID
)

CORRECTION_CONTRACT = (
    "frozen-v8.9-notebook87-a1-operational-"
    "plus-a5-v1-shadow"
)

FINAL_PACKAGE_FILES = (
    "annotated_page.png",
    "reviewed_authoritative_seed_mask.png",
    "a1_corrected.png",
    "a5_corrected.png",
    "operational_corrected.png",
    "a5_decision_overlay.png",
    "a5_live_pixel_diagnostics.csv",
    "a5_final_print_diagnostics.csv",
    "a5_diagnostic_export_manifest.json",
    "operational_correction_audit.json",
    "whole_page_review.json",
    "training_evidence_upload.json",
    "package_manifest.json",
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
    a5_shadow_status: str | None = None

    a5_failure_stage: str | None = None
    a5_failure_type: str | None = None


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

def get_correction_operator() -> A5Operator:
    global _correction_operator

    if _correction_operator is None:
        _correction_operator = A5Operator(
            bundle_dir=A5_BUNDLE_DIR,
            device=get_device(),
            run_startup_replay=True,
        )

    return _correction_operator


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

def a5_decision_overlay(
    observed: np.ndarray,
    final: np.ndarray,
    eligible_a5_v1: np.ndarray,
    alpha: float = 0.55,
    display_radius: int = 1,
) -> np.ndarray:
    """
    Diagnostic A5-v1 decision overlay.

    PRINT      -> green
    ANNOTATION -> red
    UNCERTAIN  -> yellow

    display_radius=1 expands each classified pixel to 3x3 for
    visibility only. It does NOT change the underlying decisions.
    """

    observed = require_image(
        observed,
        "A5 decision-overlay observed page",
    )

    final = np.asarray(final)

    if final.shape != observed.shape:
        raise RuntimeError(
            "A5 decision-overlay shape mismatch: "
            f"observed={observed.shape}, "
            f"final={final.shape}"
        )

    eligible_a5_v1 = np.asarray(
        eligible_a5_v1,
        dtype=bool,
    )

    if eligible_a5_v1.shape != observed.shape:
        raise RuntimeError(
            "A5 decision-overlay eligibility shape mismatch"
        )

    print_mask = (
        eligible_a5_v1
        & (final == "PRINT")
    )

    annotation_mask = (
        eligible_a5_v1
        & (final == "ANNOTATION")
    )

    uncertain_mask = (
        eligible_a5_v1
        & (final == "UNCERTAIN")
    )

    # Optional visibility enlargement.
    # This is DISPLAY ONLY.
    if display_radius > 0:
        kernel = np.ones(
            (
                2 * display_radius + 1,
                2 * display_radius + 1,
            ),
            dtype=np.uint8,
        )

        print_display = (
            cv2.dilate(
                print_mask.astype(np.uint8),
                kernel,
            )
            > 0
        )

        annotation_display = (
            cv2.dilate(
                annotation_mask.astype(np.uint8),
                kernel,
            )
            > 0
        )

        uncertain_display = (
            cv2.dilate(
                uncertain_mask.astype(np.uint8),
                kernel,
            )
            > 0
        )
    else:
        print_display = print_mask
        annotation_display = annotation_mask
        uncertain_display = uncertain_mask

    base = cv2.cvtColor(
        observed,
        cv2.COLOR_GRAY2BGR,
    )

    colour = base.copy()

    # OpenCV uses BGR.
    colour[print_display] = (
        0,
        255,
        0,
    )

    colour[annotation_display] = (
        0,
        0,
        255,
    )

    # Yellow in BGR.
    colour[uncertain_display] = (
        0,
        255,
        255,
    )

    active = (
        print_display
        | annotation_display
        | uncertain_display
    )

    output = base.copy()

    blended = cv2.addWeighted(
        base,
        1.0 - alpha,
        colour,
        alpha,
        0,
    )

    output[active] = blended[active]

    return output

# ---------------------------------------------------------------------------
# Operational result validation
# ---------------------------------------------------------------------------

def validate_operational_result(
    observed: np.ndarray,
    result: dict,
) -> dict:
    """
    Validate the operational lineage:

        V7.8 geometry_t80
            -> frozen V8.9 corrected
            -> Notebook 87 A1
            -> A5-v1 final page

    Notebook 87 A1:

        A1 = frozen V8.9 corrected
        A1[V7.8 AMBIGUOUS] = observed[V7.8 AMBIGUOUS]

    A5-v1:

        PRINT      -> exact observed byte
        non-PRINT  -> exact A1 byte
        outside authoritative mask -> exact observed byte
    """

    required = {
        "mask",
        "telea",
        "paper",
        "routes",
        "geometry_t80",
        "v8_9_corrected",
        "a1",
        "a5",
        "eligible_a5_v1",
        "p_print",
        "provisional",
        "final",
        "print_mask",
    }

    missing = sorted(
        required - set(result)
    )

    if missing:
        raise RuntimeError(
            "Operational result omitted required fields: "
            f"{missing}"
        )

    shape = observed.shape

    mask = require_bool_mask(
        result["mask"],
        "authoritative mask",
        shape,
    )

    telea = require_image(
        result["telea"],
        "Telea diagnostic",
        shape,
    )

    paper = require_image(
        result["paper"],
        "paper estimate",
        shape,
    )

    geometry_t80 = require_image(
        result["geometry_t80"],
        "V7.8 geometry_t80",
        shape,
    )

    v8_9_corrected = require_image(
        result["v8_9_corrected"],
        "frozen V8.9 corrected page",
        shape,
    )

    a1 = require_image(
        result["a1"],
        "Notebook 87 A1",
        shape,
    )

    a5 = require_image(
        result["a5"],
        "A5-v1 corrected page",
        shape,
    )

    eligible_a5_v1 = require_bool_mask(
        result["eligible_a5_v1"],
        "A5-v1 eligibility",
        shape,
    )

    print_mask = require_bool_mask(
        result["print_mask"],
        "A5-v1 PRINT mask",
        shape,
    )

    routes = result["routes"]

    if not isinstance(routes, dict):
        raise RuntimeError(
            "result['routes'] must be a dictionary"
        )

    required_routes = {
        "preserve",
        "remove",
        "ambiguous",
    }

    missing_routes = sorted(
        required_routes - set(routes)
    )

    if missing_routes:
        raise RuntimeError(
            "Operational result omitted V7.8 routes: "
            f"{missing_routes}"
        )

    preserve = require_bool_mask(
        routes["preserve"],
        "V7.8 PRESERVE",
        shape,
    )

    remove = require_bool_mask(
        routes["remove"],
        "V7.8 REMOVE",
        shape,
    )

    ambiguous = require_bool_mask(
        routes["ambiguous"],
        "V7.8 AMBIGUOUS",
        shape,
    )

    # ------------------------------------------------------------
    # V7.8 integrity
    # ------------------------------------------------------------

    if np.any(preserve & remove):
        raise RuntimeError(
            "V7.8 PRESERVE and REMOVE overlap"
        )

    if np.any(preserve & ambiguous):
        raise RuntimeError(
            "V7.8 PRESERVE and AMBIGUOUS overlap"
        )

    if np.any(remove & ambiguous):
        raise RuntimeError(
            "V7.8 REMOVE and AMBIGUOUS overlap"
        )

    routed = (
        preserve
        | remove
        | ambiguous
    )

    if not np.array_equal(
        routed,
        mask,
    ):
        raise RuntimeError(
            "V7.8 routes do not exactly partition "
            "the authoritative mask"
        )

    if np.any(
        geometry_t80[preserve]
        != observed[preserve]
    ):
        raise RuntimeError(
            "V7.8 PRESERVE does not equal observed"
        )

    if np.any(
        geometry_t80[remove]
        != paper[remove]
    ):
        raise RuntimeError(
            "V7.8 REMOVE does not equal paper"
        )

    if np.any(
        geometry_t80[ambiguous]
        != telea[ambiguous]
    ):
        raise RuntimeError(
            "V7.8 AMBIGUOUS does not equal Telea"
        )

    if np.any(
        geometry_t80[~mask]
        != observed[~mask]
    ):
        raise RuntimeError(
            "V7.8 changed pixels outside authority"
        )

    # ------------------------------------------------------------
    # Notebook 87 A1 integrity
    # ------------------------------------------------------------

    if np.any(
        a1[ambiguous]
        != observed[ambiguous]
    ):
        raise RuntimeError(
            "Notebook 87 A1 AMBIGUOUS bytes do not "
            "equal observed"
        )

    if np.any(
        a1[~ambiguous]
        != v8_9_corrected[~ambiguous]
    ):
        raise RuntimeError(
            "Notebook 87 A1 differs from frozen V8.9 "
            "outside AMBIGUOUS"
        )

    if np.any(
        a1[~mask]
        != observed[~mask]
    ):
        raise RuntimeError(
            "Notebook 87 A1 changed pixels outside "
            "the authoritative mask"
        )

    a1_delta_from_v89 = (
        a1 != v8_9_corrected
    )

    expected_a1_delta = (
        ambiguous
        & (
            observed
            != v8_9_corrected
        )
    )

    if not np.array_equal(
        a1_delta_from_v89,
        expected_a1_delta,
    ):
        raise RuntimeError(
            "Notebook 87 A1 delta is not exactly the "
            "effective AMBIGUOUS-to-observed restoration"
        )

    # ------------------------------------------------------------
    # A5 eligibility
    # ------------------------------------------------------------

    expected_eligible = (
        mask
        & (observed < 200)
        & ~preserve
    )

    if not np.array_equal(
        eligible_a5_v1,
        expected_eligible,
    ):
        raise RuntimeError(
            "A5-v1 eligible population mismatch"
        )

    if np.any(
        print_mask
        & ~eligible_a5_v1
    ):
        raise RuntimeError(
            "A5-v1 PRINT exists outside eligibility"
        )

    # ------------------------------------------------------------
    # A5 output, now falling back to A1
    # ------------------------------------------------------------

    if np.any(
        a5[print_mask]
        != observed[print_mask]
    ):
        raise RuntimeError(
            "A5-v1 PRINT output does not equal observed"
        )

    if np.any(
        a5[~print_mask]
        != a1[~print_mask]
    ):
        raise RuntimeError(
            "A5-v1 non-PRINT output differs from "
            "Notebook 87 A1"
        )

    if np.any(
        a5[~mask]
        != observed[~mask]
    ):
        raise RuntimeError(
            "A5-v1 changed pixels outside authority"
        )

    changed_from_a1 = (
        a5 != a1
    )

    expected_changed_from_a1 = (
        print_mask
        & (
            observed != a1
        )
    )

    if not np.array_equal(
        changed_from_a1,
        expected_changed_from_a1,
    ):
        raise RuntimeError(
            "A5-v1 delta does not equal effective PRINT "
            "restoration from A1"
        )

    if np.any(
        changed_from_a1
        & (a5 != observed)
    ):
        raise RuntimeError(
            "A5-v1 changed byte does not equal observed"
        )

    # ------------------------------------------------------------
    # Probability and decision integrity
    # ------------------------------------------------------------

    p_print = np.asarray(
        result["p_print"]
    )

    provisional = np.asarray(
        result["provisional"]
    )

    final = np.asarray(
        result["final"]
    )

    for name, array in (
        ("p_print", p_print),
        ("provisional", provisional),
        ("final", final),
    ):
        if array.shape != shape:
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"expected {shape}, found {array.shape}"
            )

    if not np.isfinite(
        p_print[eligible_a5_v1]
    ).all():
        raise RuntimeError(
            "p_print contains non-finite eligible values"
        )

    if np.any(
        p_print[eligible_a5_v1] < 0.0
    ) or np.any(
        p_print[eligible_a5_v1] > 1.0
    ):
        raise RuntimeError(
            "p_print lies outside 0..1"
        )

    valid_classes = {
        "PRINT",
        "ANNOTATION",
        "UNCERTAIN",
        "INELIGIBLE",
    }

    for name, decisions in (
        ("provisional", provisional),
        ("final", final),
    ):
        unexpected = (
            set(decisions.astype(str).ravel())
            - valid_classes
        )

        if unexpected:
            raise RuntimeError(
                f"Unexpected {name} classes: "
                f"{sorted(unexpected)}"
            )

        if np.any(
            decisions[eligible_a5_v1]
            == "INELIGIBLE"
        ):
            raise RuntimeError(
                f"Eligible coordinate is {name} INELIGIBLE"
            )

        if np.any(
            decisions[~eligible_a5_v1]
            != "INELIGIBLE"
        ):
            raise RuntimeError(
                f"{name} decision exists outside eligibility"
            )

    if not np.array_equal(
        print_mask,
        final == "PRINT",
    ):
        raise RuntimeError(
            "PRINT mask does not equal final PRINT decisions"
        )

    if np.any(
        (final == "PRINT")
        & (provisional != "PRINT")
    ):
        raise RuntimeError(
            "Run filter manufactured a new PRINT decision"
        )

    if np.any(
        (provisional == "ANNOTATION")
        & (final != "ANNOTATION")
    ):
        raise RuntimeError(
            "Run filter altered an ANNOTATION decision"
        )

    # Preserve each real processing stage. Do not overwrite A1.
    result["mask"] = mask
    result["telea"] = telea
    result["paper"] = paper

    result["routes"] = {
        **routes,
        "preserve": preserve,
        "remove": remove,
        "ambiguous": ambiguous,
    }

    result["geometry_t80"] = geometry_t80
    result["v8_9_corrected"] = v8_9_corrected
    result["a1"] = a1
    result["a5"] = a5

    result["eligible_a5_v1"] = eligible_a5_v1
    result["p_print"] = p_print
    result["provisional"] = provisional
    result["final"] = final
    result["print_mask"] = print_mask

    # ------------------------------------------------------------
    # Operational publication decision
    #
    # A5 continues to run as a complete shadow proposal, but A1 is
    # the published correction until A5 has separately passed the
    # required safety and promotion gates.
    # ------------------------------------------------------------

    if A5_APPLY_TO_OPERATIONAL_OUTPUT:
        operational = a5.copy()
        operational_stage = "A5_V1"
    else:
        operational = a1.copy()
        operational_stage = "NOTEBOOK87_A1"

    if np.any(
        operational[~mask]
        != observed[~mask]
    ):
        raise RuntimeError(
            "Operational output changed pixels outside "
            "the authoritative mask"
        )

    if not A5_APPLY_TO_OPERATIONAL_OUTPUT:
        if not np.array_equal(
            operational,
            a1,
        ):
            raise RuntimeError(
                "A5 shadow mode operational output "
                "does not exactly equal Notebook 87 A1"
            )

    result["operational"] = operational
    result["operational_stage"] = operational_stage
    result["a5_shadow_mode"] = bool(
        A5_SHADOW_MODE
    )
    result["a5_applied_to_operational_output"] = bool(
        A5_APPLY_TO_OPERATIONAL_OUTPUT
    )

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

    observed = require_image(
        session["gray"],
        "observed session page",
    )

    reviewed_seed = require_image(
        session["working"],
        "reviewed authoritative seed",
        observed.shape,
    )

    write_png(
        directory / "reviewed_authoritative_seed_mask.png",
        reviewed_seed,
    )

    try:
        # --------------------------------------------------------
        # Exact Notebook 87 A1 lineage:
        #
        # reviewed seed
        #   -> frozen V8.9
        #   -> restore V7.8 AMBIGUOUS coordinates to observed
        #   -> Notebook 87 A1
        # --------------------------------------------------------

        a1_result = get_a1_operator().run(
            observed=observed,
            reviewed_seed=reviewed_seed,
            page=sid,
        )

        # --------------------------------------------------------
        # A5-v1 operates from Notebook 87 A1.
        # --------------------------------------------------------

        result = get_correction_operator().run_from_a1(
            observed=observed,
            reviewed_seed=reviewed_seed,
            a1_result=a1_result,
            page=sid,
        )

        result = validate_operational_result(
            observed=observed,
            result=result,
        )

        mask = result["mask"]
        routes = result["routes"]

        geometry_t80 = result["geometry_t80"]
        v8_9_corrected = result["v8_9_corrected"]
        a1 = result["a1"]
        a5 = result["a5"]

        operational = result["operational"]
        operational_stage = result["operational_stage"]

        print_mask = result["print_mask"]
        eligible_a5_v1 = result["eligible_a5_v1"]

        p_print = result["p_print"]
        provisional = result["provisional"]
        final = result["final"]

        preserve = routes["preserve"]
        remove = routes["remove"]
        ambiguous = routes["ambiguous"]

        outside_mask = ~mask
        non_print = ~print_mask

        # --------------------------------------------------------
        # Stage deltas
        # --------------------------------------------------------

        a1_delta_from_v89 = (
            a1 != v8_9_corrected
        )

        expected_a1_delta_from_v89 = (
            ambiguous
            & (
                observed != v8_9_corrected
            )
        )

        a5_changed_from_a1 = (
            a5 != a1
        )

        expected_a5_changed_from_a1 = (
            print_mask
            & (
                observed != a1
            )
        )

        a5_changed_from_v78 = (
            a5 != geometry_t80
        )

        # --------------------------------------------------------
        # Decision overlay
        # --------------------------------------------------------

        decision_overlay = a5_decision_overlay(
            observed=observed,
            final=final,
            eligible_a5_v1=eligible_a5_v1,
            alpha=0.55,
            display_radius=1,
        )

        # --------------------------------------------------------
        # Persist only the curated A1 -> A5 review artefacts.
        write_png(directory / "a1_corrected.png", a1)
        write_png(directory / "a5_corrected.png", a5)
        write_png(directory / "operational_corrected.png", operational)
        write_png(
            directory / "a5_decision_overlay.png",
            decision_overlay,
        )

        # Export the temporary pixel-level A5 diagnostics directly from
        # the in-memory result. Full-page NPY/NPZ intermediates are not
        # written to disk.
        import csv

        channel_order = [
            "obs", "norm", "mask", "depth", "horizontal",
            "vertical", "compact", "preserve", "remove",
            "ambiguous", "edge",
        ]
        channels = result.get("channels")
        if not isinstance(channels, Mapping):
            raise RuntimeError("A5 result omitted the channel mapping")
        missing_channels = [
            name for name in channel_order if name not in channels
        ]
        if missing_channels:
            raise RuntimeError(
                f"A5 result omitted diagnostic channels: {missing_channels}"
            )
        raw_probability = np.asarray(
            result["raw_p_print"], dtype=np.float32
        )
        calibrated_probability = np.asarray(
            result["p_print"], dtype=np.float32
        )
        yolo_confidence = np.asarray(
            session["confidence"], dtype=np.float32
        )
        diagnostic_arrays = {
            "raw_p_print": raw_probability,
            "p_print": calibrated_probability,
            "provisional": provisional,
            "final": final,
            "yolo_confidence": yolo_confidence,
            **{name: np.asarray(channels[name]) for name in channel_order},
        }
        mismatched = {
            name: list(value.shape)
            for name, value in diagnostic_arrays.items()
            if value.shape != observed.shape
        }
        if mismatched:
            raise RuntimeError(
                "A5 diagnostic arrays do not share the page shape: "
                f"{mismatched}"
            )
        eligible_diagnostic = (
            np.isfinite(raw_probability)
            & np.isfinite(calibrated_probability)
        )
        eligible_y, eligible_x = np.where(eligible_diagnostic)
        all_csv_path = directory / "a5_live_pixel_diagnostics.csv"
        final_csv_path = directory / "a5_final_print_diagnostics.csv"
        fieldnames = [
            "y", "x", "raw_p_print", "calibrated_p_print",
            "provisional_decision", "final_decision",
            "yolo_confidence", "obs_source_byte",
            "norm_source_byte", "obs_cnn_centre",
            "norm_cnn_centre", "mask_cnn_centre",
            "depth_cnn_centre", "horizontal_cnn_centre",
            "vertical_cnn_centre", "compact_cnn_centre",
            "preserve_cnn_centre", "remove_cnn_centre",
            "ambiguous_cnn_centre", "edge_cnn_centre",
        ]
        final_print_rows = 0
        with all_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as all_handle, final_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as final_handle:
            all_writer = csv.DictWriter(all_handle, fieldnames=fieldnames)
            final_writer = csv.DictWriter(final_handle, fieldnames=fieldnames)
            all_writer.writeheader()
            final_writer.writeheader()
            for y, x in zip(eligible_y, eligible_x):
                y, x = int(y), int(x)
                provisional_value = str(provisional[y, x])
                final_value = str(final[y, x])
                row = {
                    "y": y,
                    "x": x,
                    "raw_p_print": float(raw_probability[y, x]),
                    "calibrated_p_print": float(
                        calibrated_probability[y, x]
                    ),
                    "provisional_decision": provisional_value,
                    "final_decision": final_value,
                    "yolo_confidence": float(yolo_confidence[y, x]),
                    "obs_source_byte": float(channels["obs"][y, x]),
                    "norm_source_byte": float(channels["norm"][y, x]),
                    "obs_cnn_centre": float(channels["obs"][y, x]) / 255.0,
                    "norm_cnn_centre": float(channels["norm"][y, x]) / 255.0,
                    "mask_cnn_centre": float(channels["mask"][y, x]),
                    "depth_cnn_centre": float(channels["depth"][y, x]),
                    "horizontal_cnn_centre": float(
                        channels["horizontal"][y, x]
                    ),
                    "vertical_cnn_centre": float(
                        channels["vertical"][y, x]
                    ),
                    "compact_cnn_centre": float(
                        channels["compact"][y, x]
                    ),
                    "preserve_cnn_centre": float(
                        channels["preserve"][y, x]
                    ),
                    "remove_cnn_centre": float(
                        channels["remove"][y, x]
                    ),
                    "ambiguous_cnn_centre": float(
                        channels["ambiguous"][y, x]
                    ),
                    "edge_cnn_centre": float(channels["edge"][y, x]),
                }
                all_writer.writerow(row)
                if final_value == "PRINT":
                    final_writer.writerow(row)
                    final_print_rows += 1

        diagnostic_manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "page_shape": list(observed.shape),
            "channel_order": channel_order,
            "patch_size": 33,
            "padding": "reflect",
            "normalisation": {
                "obs": "source value divided by 255",
                "norm": "source value divided by 255",
                "other_channels": "native float32 value",
            },
            "eligible_rows": int(eligible_diagnostic.sum()),
            "final_print_rows": int(final_print_rows),
            "files": {
                "all_eligible_pixels": all_csv_path.name,
                "final_print_pixels": final_csv_path.name,
            },
            "full_page_numpy_arrays_exported": False,
        }
        (directory / "a5_diagnostic_export_manifest.json").write_text(
            json.dumps(diagnostic_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # Whole-page operational audit
        # --------------------------------------------------------

        audit = {
            "created_utc": (
                datetime.now(timezone.utc).isoformat()
            ),

            "session_id": sid,
            "app_version": APP_VERSION,
            "correction_contract": CORRECTION_CONTRACT,

            # ----------------------------------------------------
            # V7.8 identity
            # ----------------------------------------------------

            "v7_8_policy_id": V78_POLICY_ID,
            "v7_8_baseline": "geometry_t80",

            # ----------------------------------------------------
            # Notebook 87 A1 identity
            # ----------------------------------------------------

            "a1_policy_id": result.get(
                "a1_policy_id",
                A1_POLICY_ID,
            ),

            "v89_operator_sha256": result.get(
                "v89_operator_sha256",
            ),

            "a1_source": (
                "frozen/v8_9/v8_9_operator.py"
            ),

            "a1_assembly": (
                "V8.9 corrected with every V7.8 "
                "AMBIGUOUS coordinate restored to "
                "the exact observed byte"
            ),

                        # ----------------------------------------------------
            # Operational publication identity
            # ----------------------------------------------------

            "operational_output_stage": (
                operational_stage
            ),

            "operational_output_file": (
                "operational_corrected.png"
            ),

            "a5_shadow_mode": bool(
                A5_SHADOW_MODE
            ),

            "a5_applied_to_operational_output": bool(
                A5_APPLY_TO_OPERATIONAL_OUTPUT
            ),

            "operational_changed_from_a1_pixels": int(
                np.sum(
                    operational != a1
                )
            ),

            "operational_outside_mask_drift": int(
                np.sum(
                    operational[outside_mask]
                    != observed[outside_mask]
                )
            ),

            # ----------------------------------------------------
            # A5-v1 identity
            # ----------------------------------------------------

            "a5_action": (
                "COPY_EXACT_OBSERVED_BYTE"
            ),

            "a5_fallback": (
                "Notebook87 A1"
            ),

            "a5_policy_id": result.get(
                "a5_policy_id",
                A5_POLICY_ID,
            ),

            "a5_bundle_id": result.get(
                "bundle_id"
            ),

            "a5_model_sha256": result.get(
                "model_sha256"
            ),

            "a5_calibrator_sha256": result.get(
                "calibrator_sha256"
            ),

            # ----------------------------------------------------
            # Population counts
            # ----------------------------------------------------

            "prepared_mask_pixels": int(
                mask.sum()
            ),

            "preserve_pixels": int(
                preserve.sum()
            ),

            "remove_pixels": int(
                remove.sum()
            ),

            "ambiguous_pixels": int(
                ambiguous.sum()
            ),

            "a1_changed_from_v8_9_pixels": int(
                a1_delta_from_v89.sum()
            ),

            "a5_eligible_pixels": int(
                eligible_a5_v1.sum()
            ),

            "a5_final_print_pixels": int(
                print_mask.sum()
            ),

            "a5_effectively_changed_from_a1_pixels": int(
                a5_changed_from_a1.sum()
            ),

            "a5_changed_from_v7_8_pixels": int(
                a5_changed_from_v78.sum()
            ),

            "a5_final_annotation_pixels": int(
                np.sum(
                    final[eligible_a5_v1]
                    == "ANNOTATION"
                )
            ),

            "a5_final_uncertain_pixels": int(
                np.sum(
                    final[eligible_a5_v1]
                    == "UNCERTAIN"
                )
            ),

            # ----------------------------------------------------
            # V7.8 integrity
            # ----------------------------------------------------

            "v7_8_outside_mask_drift": int(
                np.sum(
                    geometry_t80[outside_mask]
                    != observed[outside_mask]
                )
            ),

            # ----------------------------------------------------
            # V8.9 integrity
            # ----------------------------------------------------

            "v8_9_outside_mask_drift": int(
                np.sum(
                    v8_9_corrected[outside_mask]
                    != observed[outside_mask]
                )
            ),

            # ----------------------------------------------------
            # Notebook 87 A1 integrity
            # ----------------------------------------------------

            "a1_outside_mask_drift": int(
                np.sum(
                    a1[outside_mask]
                    != observed[outside_mask]
                )
            ),

            "a1_ambiguous_not_equal_observed": int(
                np.sum(
                    a1[ambiguous]
                    != observed[ambiguous]
                )
            ),

            "a1_non_ambiguous_v89_regression": int(
                np.sum(
                    a1[~ambiguous]
                    != v8_9_corrected[~ambiguous]
                )
            ),

            "a1_unexpected_delta_from_v89": int(
                np.sum(
                    a1_delta_from_v89
                    & ~expected_a1_delta_from_v89
                )
            ),

            "a1_missing_effective_ambiguous_restoration": int(
                np.sum(
                    expected_a1_delta_from_v89
                    & ~a1_delta_from_v89
                )
            ),

            # ----------------------------------------------------
            # A5-v1 integrity
            # ----------------------------------------------------

            "a5_outside_mask_drift": int(
                np.sum(
                    a5[outside_mask]
                    != observed[outside_mask]
                )
            ),

            "a5_non_print_a1_regression": int(
                np.sum(
                    a5[non_print]
                    != a1[non_print]
                )
            ),

            "a5_print_not_equal_observed": int(
                np.sum(
                    a5[print_mask]
                    != observed[print_mask]
                )
            ),

            "a5_changed_byte_not_equal_observed": int(
                np.sum(
                    a5_changed_from_a1
                    & (
                        a5 != observed
                    )
                )
            ),

            "a5_unexpected_changed_byte": int(
                np.sum(
                    a5_changed_from_a1
                    & ~expected_a5_changed_from_a1
                )
            ),

            "a5_missing_effective_print_change": int(
                np.sum(
                    expected_a5_changed_from_a1
                    & ~a5_changed_from_a1
                )
            ),

            "print_outside_eligibility": int(
                np.sum(
                    print_mask
                    & ~eligible_a5_v1
                )
            ),

            "print_outside_authoritative_mask": int(
                np.sum(
                    print_mask
                    & ~mask
                )
            ),

            # ----------------------------------------------------
            # Runtime provenance
            # ----------------------------------------------------

            "telea_contributes_to_v7_8_ambiguous": bool(
                ambiguous.any()
            ),

            "clean_image_used_at_runtime": False,

            "historical_notebook87_a1_operational": True,

            "historical_notebook91_drives_output": False,
        }

        zero_fields = [
            "v7_8_outside_mask_drift",
            "v8_9_outside_mask_drift",

            "a1_outside_mask_drift",
            "a1_ambiguous_not_equal_observed",
            "a1_non_ambiguous_v89_regression",
            "a1_unexpected_delta_from_v89",
            "a1_missing_effective_ambiguous_restoration",

            "a5_outside_mask_drift",
            "a5_non_print_a1_regression",
            "a5_print_not_equal_observed",
            "a5_changed_byte_not_equal_observed",
            "a5_unexpected_changed_byte",
            "a5_missing_effective_print_change",

            "operational_outside_mask_drift",

            "print_outside_eligibility",
            "print_outside_authoritative_mask",
        ]

        failures = {
            key: audit[key]
            for key in zero_fields
            if audit[key] != 0
        }

        if (
            not A5_APPLY_TO_OPERATIONAL_OUTPUT
            and audit[
                "operational_changed_from_a1_pixels"
            ] != 0
        ):
            failures[
                "operational_changed_from_a1_pixels"
            ] = audit[
                "operational_changed_from_a1_pixels"
            ]

        if failures:
            raise RuntimeError(
                "Operational correction contract failed: "
                + json.dumps(
                    failures,
                    sort_keys=True,
                )
            )

        (
            directory
            / "operational_correction_audit.json"
        ).write_text(
            json.dumps(
                audit,
                indent=2,
            )
        )

    except Exception as exc:
        failure = {
            "utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "session_id": sid,
            "contract": CORRECTION_CONTRACT,

            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        (
            directory
            / "correction_failure.json"
        ).write_text(
            json.dumps(
                failure,
                indent=2,
            )
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Operational correction failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    return {
        "ok": True,

        "corrected_url": (
            f"/api/session/{sid}/result"
        ),

        "operational_stage": (
            audit[
                "operational_output_stage"
            ]
        ),

        "a5_shadow_mode": (
            audit[
                "a5_shadow_mode"
            ]
        ),

        "a5_applied_to_operational_output": (
            audit[
                "a5_applied_to_operational_output"
            ]
        ),

        "a5_shadow_url": (
            f"/api/session/{sid}/a5-shadow"
        ),

        "original_url": (
            f"/api/session/{sid}/image"
        ),

        "diagnostics": {
            "prepared_mask_pixels": audit[
                "prepared_mask_pixels"
            ],

            "preserve_pixels": audit[
                "preserve_pixels"
            ],

            "remove_pixels": audit[
                "remove_pixels"
            ],

            "ambiguous_pixels": audit[
                "ambiguous_pixels"
            ],

            "a5_eligible_pixels": audit[
                "a5_eligible_pixels"
            ],

            "a5_print_pixels": audit[
                "a5_final_print_pixels"
            ],

            "a1_changed_from_v8_9": audit[
                "a1_changed_from_v8_9_pixels"
            ],

            "a5_changed_from_a1": audit[
                "a5_effectively_changed_from_a1_pixels"
            ],

            "a5_changed_from_v7_8": audit[
                "a5_changed_from_v7_8_pixels"
            ],

            "outside_mask_drift": audit[
                "a5_outside_mask_drift"
            ],

            "non_print_a1_regression": audit[
                "a5_non_print_a1_regression"
            ],
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

@app.get("/api/session/{sid}/a5-shadow")
def a5_shadow_result(sid: str):
    return png(
        read_session_png(
            sid,
            "a5_corrected.png",
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
        "a5_shadow": "a5_corrected.png",
        "a5": "a5_corrected.png",
        "decisions": "a5_decision_overlay.png",
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
        ("a5_shadow_status", value.a5_shadow_status),
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

        "a5_shadow_status": (
            value.a5_shadow_status
            or "not_reviewed"
        ),

        "a5_failure_stage": (
            value.a5_failure_stage
        ),

        "a5_failure_type": (
            value.a5_failure_type
        ),

        "note": value.note,

        "operational_output_stage": (
            "A5_V1"
            if A5_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_applied_to_operational_output": bool(
            A5_APPLY_TO_OPERATIONAL_OUTPUT
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
        final_print_rows = int(diagnostic_manifest["final_print_rows"])
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
    a5_operator = get_correction_operator()

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
            "A5_V1"
            if A5_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_shadow_mode": bool(A5_SHADOW_MODE),

        "a5_applied_to_operational_output": bool(
            A5_APPLY_TO_OPERATIONAL_OUTPUT
        ),

        "a5_shadow_proposal_file": (
            "a5_corrected.png"
        ),

        # --------------------------------------------------------
        # A5-v1
        # --------------------------------------------------------

        "a5_policy_id": getattr(
            a5_operator,
            "a5_policy_id",
            A5_POLICY_ID,
        ),

        "a5_bundle_id": getattr(
            a5_operator,
            "bundle_id",
            None,
        ),

        "a5_print_action": (
            "COPY_EXACT_OBSERVED_BYTE"
        ),

        "a5_non_print_fallback": (
            "Notebook87 A1"
        ),

        "a5_raw_probability_exported": False,

        "a5_calibrated_probability_exported": False,

        "a5_channel_diagnostic_csv_exported": True,

        "a5_channel_order": (
            channel_order
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

        "a5_shadow_status": review.get(
            "a5_shadow_status"
        ),

        "failure_stage": review.get(
            "a5_failure_stage"
        ),

        "failure_type": review.get(
            "a5_failure_type"
        ),

        "notes": review.get(
            "note",
            "",
        ),

        "operational_output_stage": (
            "A5_V1"
            if A5_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_applied_to_operational_output": bool(
            A5_APPLY_TO_OPERATIONAL_OUTPUT
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
            "A5_V1"
            if A5_APPLY_TO_OPERATIONAL_OUTPUT
            else "NOTEBOOK87_A1"
        ),

        "a5_shadow_mode": bool(A5_SHADOW_MODE),

        "a5_applied_to_operational_output": bool(
            A5_APPLY_TO_OPERATIONAL_OUTPUT
        ),

        "training_evidence": (
            training_evidence
        ),

        "a5_eligible_diagnostic_rows": int(
            eligible_rows
        ),

        "a5_final_print_diagnostic_rows": int(
            final_print_rows
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
    a5_operator = get_correction_operator()

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
            if not A5_APPLY_TO_OPERATIONAL_OUTPUT
            else "A5_V1"
        ),

        "a5_shadow_mode": bool(
            A5_SHADOW_MODE
        ),

        "a5_applied_to_operational_output": bool(
            A5_APPLY_TO_OPERATIONAL_OUTPUT
        ),

        # --------------------------------------------------------
        # A5-v1
        # --------------------------------------------------------

        "a5_policy_id": getattr(
            a5_operator,
            "a5_policy_id",
            A5_POLICY_ID,
        ),

        "bundle_id": getattr(
            a5_operator,
            "bundle_id",
            None,
        ),

        "a5_print_action": (
            "COPY_EXACT_OBSERVED_BYTE"
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
#         operator=A5Operator,
#         output_root=Path,
#     )
#
# If hf045_batch.py has not yet been changed to this interface,
# update that file before enabling batch processing.
# ---------------------------------------------------------------------------

@app.post(
    "/batch-v89-a1-a5",
    response_class=FileResponse,
)
async def batch_v89_a1_a5(
    file: UploadFile = File(...),
):
    """
    Process one HF-04.5.1 ZIP through:

        frozen V7.8
        -> frozen V8.9
        -> Notebook 87 A1
        -> A5-v1 observed-PRINT processing

    The returned ZIP contains page artefacts, audit records
    and the batch manifest.
    """

    filename = Path(
        file.filename or ""
    ).name

    if not filename.lower().endswith(
        ".zip"
    ):
        await file.close()

        raise HTTPException(
            status_code=400,
            detail="Upload one .zip file",
        )

    try:
        upload = await file.read()

        if not upload:
            raise HTTPException(
                status_code=400,
                detail="Uploaded ZIP is empty",
            )

        output = run_batch(
            upload=upload,

            a1_operator=get_a1_operator(),

            operator=get_correction_operator(),

            output_root=BATCH_OUTPUT_ROOT,
        )

        output = Path(output)

        if not output.is_file():
            raise BatchFailure(
                "Batch completed without creating "
                "its output ZIP"
            )

        if output.stat().st_size == 0:
            raise BatchFailure(
                "Batch output ZIP is empty"
            )

        return FileResponse(
            path=str(output),

            filename=(
                "hf045_v89_a1_a5_results.zip"
            ),

            media_type="application/zip",

            headers={
                "Cache-Control": "no-store",
            },
        )

    except HTTPException:
        raise

    except BatchFailure as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "HF-04.5.1 batch processing failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    finally:
        await file.close()


# ---------------------------------------------------------------------------
# Legacy batch URL compatibility
# ---------------------------------------------------------------------------

@app.post(
    "/batch-v78-a5",
    response_class=FileResponse,
    include_in_schema=False,
)
async def batch_v78_a5_compatibility(
    file: UploadFile = File(...),
):
    return await batch_v89_a1_a5(
        file
    )