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
from .a5_operator import A5Operator
from .hf045_batch import BatchFailure, run_batch


# ---------------------------------------------------------------------------
# Application identity and paths
# ---------------------------------------------------------------------------

APP_VERSION = "4.5.1"
CORRECTION_CONTRACT = "frozen-v8.9-notebook87-a1-plus-a5-v1"
A1_POLICY_ID = "Notebook87-A1-from-frozen-V8.9"
V78_POLICY_ID = "V7.8-geometry-t80"
A5_POLICY_ID = "A5-v1-print-p0980-margin0960-run2"

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "frozen" / "yolo_detector" / "best.pt"
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
            device=get_device(),
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
    write_png(directory / "yolo_draft_mask.png", draft)
    np.save(
        directory / "yolo_confidence.npy",
        confidence.astype(np.float16),
    )
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
        # Persist V7.8 evidence
        # --------------------------------------------------------

        write_png(
            directory / "prepared_dilation2_mask.png",
            mask,
        )

        write_png(
            directory / "telea_diagnostic.png",
            result["telea"],
        )

        write_png(
            directory / "paper_estimate.png",
            result["paper"],
        )

        write_png(
            directory / "route_preserve.png",
            preserve,
        )

        write_png(
            directory / "route_remove.png",
            remove,
        )

        write_png(
            directory / "route_ambiguous.png",
            ambiguous,
        )

        write_png(
            directory / "v7_8_corrected.png",
            geometry_t80,
        )

        # --------------------------------------------------------
        # Persist V8.9 / Notebook 87 A1 evidence
        # --------------------------------------------------------

        write_png(
            directory / "v8_9_corrected.png",
            v8_9_corrected,
        )

        write_png(
            directory / "a1_corrected.png",
            a1,
        )

        write_png(
            directory / "a1_delta_from_v8_9.png",
            a1_delta_from_v89,
        )

        # --------------------------------------------------------
        # Persist A5-v1 evidence
        # --------------------------------------------------------

        write_png(
            directory / "a5_eligible.png",
            eligible_a5_v1,
        )

        # Legacy compatibility alias.
        write_png(
            directory / "a5_eligible_ambiguous.png",
            eligible_a5_v1,
        )

        write_png(
            directory / "a5_print_mask.png",
            print_mask,
        )

        write_png(
            directory / "a5_corrected.png",
            a5,
        )

        write_png(
            directory / "a5_delta_from_a1.png",
            a5_changed_from_a1,
        )

        write_png(
            directory / "a5_delta_from_v7_8.png",
            a5_changed_from_v78,
        )

        write_png(
            directory / "a5_decision_overlay.png",
            decision_overlay,
        )

        np.save(
            directory / "a5_print_probability.npy",
            p_print,
        )

        np.save(
            directory / "a5_provisional_decision.npy",
            provisional,
            allow_pickle=True,
        )

        np.save(
            directory / "a5_final_decision.npy",
            final,
            allow_pickle=True,
        )

        # --------------------------------------------------------
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

            "print_outside_eligibility",
            "print_outside_authoritative_mask",
        ]

        failures = {
            key: audit[key]
            for key in zero_fields
            if audit[key] != 0
        }

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
            "a5_corrected.png",
        )
    )


@app.get("/api/session/{sid}/diagnostic/{name}")
def diagnostic(
    sid: str,
    name: str,
):
    names = {
        "mask": "prepared_dilation2_mask.png",

        "telea": "telea_diagnostic.png",
        "paper": "paper_estimate.png",

        "preserve": "route_preserve.png",
        "remove": "route_remove.png",
        "ambiguous": "route_ambiguous.png",

        "v78": "v7_8_corrected.png",

        "v89": "v8_9_corrected.png",

        "a1": "a1_corrected.png",
        "a1_delta": "a1_delta_from_v8_9.png",

        "eligible_a5_v1": "a5_eligible.png",
        "print": "a5_print_mask.png",

        "delta": "a5_delta_from_a1.png",
        "a5_delta_a1": "a5_delta_from_a1.png",
        "a5_delta_v78": "a5_delta_from_v7_8.png",

        "a5": "a5_corrected.png",

        "decisions": "a5_decision_overlay.png",

        # --------------------------------------------------------
        # Temporary compatibility aliases.
        # --------------------------------------------------------

        "clean": "a5_print_mask.png",

        "ring2": "a5_eligible_ambiguous.png",

        "safe_remove": "route_remove.png",

        "m5": "a5_corrected.png",

        # Legacy frontend label only.
        # This now shows the actual frozen V8.9 page.
        "v88": "v8_9_corrected.png",
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
        "status": value.status,
        "note": value.note,
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

    write_png(
        directory
        / "reviewed_authoritative_seed_mask.png",
        session[
            "working"
        ],
    )

    # ------------------------------------------------------------
    # Export an inspectable A5 pixel-level diagnostic table.
    #
    # state.npz remains the authoritative full-array record.
    # These CSV files contain one row per eligible A5 coordinate,
    # with raw/calibrated probabilities, decisions and the central
    # values of the exact 11-channel CNN input schema.
    # ------------------------------------------------------------

    state_path = (
        directory
        / "state.npz"
    )

    raw_probability_path = (
        directory
        / "a5_raw_print_probability.npy"
    )

    calibrated_probability_path = (
        directory
        / "a5_print_probability.npy"
    )

    provisional_path = (
        directory
        / "a5_provisional_decision.npy"
    )

    final_path = (
        directory
        / "a5_final_decision.npy"
    )

    yolo_path = (
        directory
        / "yolo_confidence.npy"
    )

    diagnostic_inputs = [
        state_path,
        raw_probability_path,
        calibrated_probability_path,
        provisional_path,
        final_path,
        yolo_path,
    ]

    missing_diagnostic_inputs = [
        path.name
        for path in diagnostic_inputs
        if not path.is_file()
    ]

    if missing_diagnostic_inputs:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "A5 diagnostic export inputs "
                    "are incomplete."
                ),

                "missing": (
                    missing_diagnostic_inputs
                ),
            },
        )

    channel_order = [
        "obs",
        "norm",
        "mask",
        "depth",
        "horizontal",
        "vertical",
        "compact",
        "preserve",
        "remove",
        "ambiguous",
        "edge",
    ]

    channel_aliases = {
        "obs": [
            "obs",
            "observed",
        ],

        "norm": [
            "norm",
            "normalised",
            "normalized",
        ],

        "mask": [
            "mask",
            "authoritative_mask",
            "prepared_mask",
        ],

        "depth": [
            "depth",
            "mask_depth",
        ],

        "horizontal": [
            "horizontal",
            "g80_horizontal",
        ],

        "vertical": [
            "vertical",
            "g80_vertical",
        ],

        "compact": [
            "compact",
            "g80_compact",
        ],

        "preserve": [
            "preserve",
            "g80_preserve",
        ],

        "remove": [
            "remove",
            "g80_remove",
        ],

        "ambiguous": [
            "ambiguous",
            "g80_ambiguous",
        ],

        "edge": [
            "edge",
            "edges",
            "observed_edge",
        ],
    }

    def state_array(
        state,
        logical_name: str,
    ) -> np.ndarray:
        accepted_names = channel_aliases[
            logical_name
        ]

        for name in accepted_names:
            if name in state.files:
                value = np.asarray(
                    state[
                        name
                    ]
                )

                if value.ndim != 2:
                    raise RuntimeError(
                        "A5 state array "
                        f"{name!r} must be 2-D; "
                        f"found {value.shape}"
                    )

                return value

        raise RuntimeError(
            "state.npz omitted the required "
            f"A5 channel {logical_name!r}; "
            f"accepted names={accepted_names}, "
            f"available={state.files}"
        )

    def decision_text(
        value,
    ) -> str:
        if isinstance(
            value,
            bytes,
        ):
            return value.decode(
                "utf-8"
            )

        return str(
            value
        )

    try:
        with np.load(
            state_path,
            allow_pickle=False,
        ) as state:
            channels = {
                name: np.ascontiguousarray(
                    state_array(
                        state,
                        name,
                    )
                )
                for name in channel_order
            }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load the numeric A5 "
                f"channel arrays from state.npz: {exc}"
            ),
        ) from exc

    raw_probability = np.asarray(
        np.load(
            raw_probability_path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    calibrated_probability = np.asarray(
        np.load(
            calibrated_probability_path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    # These two arrays are trusted local HF outputs. Older versions
    # stored decision strings using dtype=object, which requires
    # allow_pickle=True.
    provisional = np.asarray(
        np.load(
            provisional_path,
            allow_pickle=True,
        )
    )

    final = np.asarray(
        np.load(
            final_path,
            allow_pickle=True,
        )
    )

    yolo_confidence = np.asarray(
        np.load(
            yolo_path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    shape = raw_probability.shape

    if len(
        shape
    ) != 2:
        raise HTTPException(
            status_code=500,
            detail=(
                "The A5 raw probability map must "
                f"be 2-D; found {shape}."
            ),
        )

    arrays_to_validate = {
        "calibrated_probability": (
            calibrated_probability
        ),

        "provisional_decision": (
            provisional
        ),

        "final_decision": (
            final
        ),

        "yolo_confidence": (
            yolo_confidence
        ),

        **channels,
    }

    mismatched_shapes = {
        name: list(
            value.shape
        )
        for name, value in arrays_to_validate.items()
        if value.shape != shape
    }

    if mismatched_shapes:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "A5 diagnostic arrays do not "
                    "share one page shape."
                ),

                "expected_shape": list(
                    shape
                ),

                "mismatched_shapes": (
                    mismatched_shapes
                ),
            },
        )

    eligible = (
        np.isfinite(
            raw_probability
        )
        & np.isfinite(
            calibrated_probability
        )
    )

    eligible_y, eligible_x = np.where(
        eligible
    )

    all_csv_path = (
        directory
        / "a5_live_pixel_diagnostics.csv"
    )

    final_print_csv_path = (
        directory
        / "a5_final_print_diagnostics.csv"
    )

    fieldnames = [
        "y",
        "x",

        "raw_p_print",
        "calibrated_p_print",

        "provisional_decision",
        "final_decision",

        "yolo_confidence",

        "obs_source_byte",
        "norm_source_byte",

        "obs_cnn_centre",
        "norm_cnn_centre",

        "mask_cnn_centre",
        "depth_cnn_centre",
        "horizontal_cnn_centre",
        "vertical_cnn_centre",
        "compact_cnn_centre",
        "preserve_cnn_centre",
        "remove_cnn_centre",
        "ambiguous_cnn_centre",
        "edge_cnn_centre",
    ]

    all_temporary = all_csv_path.with_name(
        all_csv_path.name
        + ".tmp"
    )

    final_temporary = (
        final_print_csv_path.with_name(
            final_print_csv_path.name
            + ".tmp"
        )
    )

    try:
        with (
            all_temporary.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as all_handle,
            final_temporary.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as final_handle,
        ):
            all_writer = csv.DictWriter(
                all_handle,
                fieldnames=fieldnames,
            )

            final_writer = csv.DictWriter(
                final_handle,
                fieldnames=fieldnames,
            )

            all_writer.writeheader()
            final_writer.writeheader()

            final_print_rows = 0

            for y, x in zip(
                eligible_y,
                eligible_x,
            ):
                y = int(
                    y
                )

                x = int(
                    x
                )

                provisional_value = decision_text(
                    provisional[
                        y,
                        x,
                    ]
                )

                final_value = decision_text(
                    final[
                        y,
                        x,
                    ]
                )

                row = {
                    "y": y,
                    "x": x,

                    "raw_p_print": float(
                        raw_probability[
                            y,
                            x,
                        ]
                    ),

                    "calibrated_p_print": float(
                        calibrated_probability[
                            y,
                            x,
                        ]
                    ),

                    "provisional_decision": (
                        provisional_value
                    ),

                    "final_decision": (
                        final_value
                    ),

                    "yolo_confidence": float(
                        yolo_confidence[
                            y,
                            x,
                        ]
                    ),

                    "obs_source_byte": float(
                        channels[
                            "obs"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "norm_source_byte": float(
                        channels[
                            "norm"
                        ][
                            y,
                            x,
                        ]
                    ),

                    # Values on the exact scale supplied to the CNN.
                    "obs_cnn_centre": float(
                        channels[
                            "obs"
                        ][
                            y,
                            x,
                        ]
                    )
                    / 255.0,

                    "norm_cnn_centre": float(
                        channels[
                            "norm"
                        ][
                            y,
                            x,
                        ]
                    )
                    / 255.0,

                    "mask_cnn_centre": float(
                        channels[
                            "mask"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "depth_cnn_centre": float(
                        channels[
                            "depth"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "horizontal_cnn_centre": float(
                        channels[
                            "horizontal"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "vertical_cnn_centre": float(
                        channels[
                            "vertical"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "compact_cnn_centre": float(
                        channels[
                            "compact"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "preserve_cnn_centre": float(
                        channels[
                            "preserve"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "remove_cnn_centre": float(
                        channels[
                            "remove"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "ambiguous_cnn_centre": float(
                        channels[
                            "ambiguous"
                        ][
                            y,
                            x,
                        ]
                    ),

                    "edge_cnn_centre": float(
                        channels[
                            "edge"
                        ][
                            y,
                            x,
                        ]
                    ),
                }

                all_writer.writerow(
                    row
                )

                if final_value.upper() == "PRINT":
                    final_writer.writerow(
                        row
                    )

                    final_print_rows += 1

        all_temporary.replace(
            all_csv_path
        )

        final_temporary.replace(
            final_print_csv_path
        )

    except Exception as exc:
        all_temporary.unlink(
            missing_ok=True
        )

        final_temporary.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to write A5 pixel "
                f"diagnostic CSV files: {exc}"
            ),
        ) from exc

    diagnostic_manifest = {
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "page_shape": list(
            shape
        ),

        "channel_order": (
            channel_order
        ),

        "patch_size": 33,

        "padding": "reflect",

        "normalisation": {
            "obs": "source value divided by 255",
            "norm": "source value divided by 255",
            "other_channels": "native float32 value",
        },

        "eligible_rows": int(
            eligible.sum()
        ),

        "final_print_rows": int(
            final_print_rows
        ),

        "files": {
            "all_eligible_pixels": (
                all_csv_path.name
            ),

            "final_print_pixels": (
                final_print_csv_path.name
            ),

            "full_channel_arrays": (
                state_path.name
            ),

            "raw_probability_map": (
                raw_probability_path.name
            ),

            "calibrated_probability_map": (
                calibrated_probability_path.name
            ),
        },
    }

    (
        directory
        / "a5_diagnostic_export_manifest.json"
    ).write_text(
        json.dumps(
            diagnostic_manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    required = [
        "annotated_page.png",
        "reviewed_authoritative_seed_mask.png",
        "whole_page_review.json",

        "prepared_dilation2_mask.png",

        "route_preserve.png",
        "route_remove.png",
        "route_ambiguous.png",

        "paper_estimate.png",
        "telea_diagnostic.png",

        "v7_8_corrected.png",

        "v8_9_corrected.png",

        "a1_corrected.png",
        "a1_delta_from_v8_9.png",

        "a5_eligible.png",
        "a5_print_mask.png",

        "a5_corrected.png",

        "a5_delta_from_a1.png",
        "a5_delta_from_v7_8.png",

        "a5_decision_overlay.png",

        "a5_raw_print_probability.npy",
        "a5_print_probability.npy",
        "a5_provisional_decision.npy",
        "a5_final_decision.npy",

        "yolo_confidence.npy",
        "state.npz",

        "a5_live_pixel_diagnostics.csv",
        "a5_final_print_diagnostics.csv",
        "a5_diagnostic_export_manifest.json",

        "operational_correction_audit.json",
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

        "telea_separate_file_is_diagnostic": True,

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

        "a5_raw_probability_exported": True,

        "a5_calibrated_probability_exported": True,

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

    for path in sorted(
        directory.iterdir()
    ):
        if (
            path.is_file()
            and path != output
            and path.name
            != "package_manifest.json"
        ):
            manifest[
                "files"
            ].append(
                {
                    "name": path.name,

                    "bytes": (
                        path.stat().st_size
                    ),
                }
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

    with zipfile.ZipFile(
        output,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(
            directory.iterdir()
        ):
            if (
                path.is_file()
                and path != output
            ):
                archive.write(
                    path,
                    path.name,
                )

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

        "a5_eligible_diagnostic_rows": int(
            eligible.sum()
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