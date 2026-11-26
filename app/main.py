from pathlib import Path
from datetime import datetime, timezone
import io
import json
import shutil
import zipfile

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import store, yolo
from .a5_operator import A5Operator
from .hf045_batch import BatchFailure, run_batch


# ---------------------------------------------------------------------------
# Application identity and paths
# ---------------------------------------------------------------------------

APP_VERSION = "4.5.0"
CORRECTION_CONTRACT = "frozen-v7.8-plus-a5-v1-observed-print"
V78_POLICY_ID = "V7.8-geometry-t80"
A5_POLICY_ID = "A5-v1-print-p0980-margin0960-run2"

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "frozen" / "yolo_detector" / "best.pt"
BATCH_OUTPUT_ROOT = ROOT / "batch_runs"
STARTUP_EVIDENCE_DIR = ROOT / "startup_evidence"

BATCH_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

_model = None
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


def read_session_png(sid: str, filename: str) -> np.ndarray:
    path = store.folder(sid) / filename
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session artefact not found: {filename}",
        )

    return image


# ---------------------------------------------------------------------------
# Operational result validation
# ---------------------------------------------------------------------------

def validate_operational_result(
    observed: np.ndarray,
    result: dict,
) -> dict:
    """
    Validate the frozen V7.8 + A5-v1 observed-PRINT contract.

    Frozen V7.8 baseline:
        PRESERVE  -> observed
        REMOVE    -> paper
        AMBIGUOUS -> Telea

    Frozen A5-v1:
        eligible population:
            authoritative mask
            AND observed < 200
            AND NOT V7.8 PRESERVE

        PRINT      -> exact observed byte
        ANNOTATION -> exact V7.8 baseline
        UNCERTAIN  -> exact V7.8 baseline

        outside authoritative mask -> exact observed byte
    """

    required = {
        "mask",
        "telea",
        "paper",
        "routes",
        "geometry_t80",
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
            "Correction operator omitted required fields: "
            f"{missing}"
        )

    shape = observed.shape

    mask = require_bool_mask(
        result["mask"],
        "mask",
        shape,
    )

    telea = require_image(
        result["telea"],
        "telea",
        shape,
    )

    paper = require_image(
        result["paper"],
        "paper",
        shape,
    )

    baseline = require_image(
        result["geometry_t80"],
        "geometry_t80",
        shape,
    )

    a5 = require_image(
        result["a5"],
        "a5",
        shape,
    )

    eligible_a5_v1 = require_bool_mask(
        result["eligible_a5_v1"],
        "eligible_a5_v1",
        shape,
    )

    print_mask = require_bool_mask(
        result["print_mask"],
        "print_mask",
        shape,
    )

    routes = result["routes"]

    if not isinstance(routes, dict):
        raise RuntimeError(
            "result['routes'] must be a dictionary"
        )

    route_names = {
        "preserve",
        "remove",
        "ambiguous",
    }

    missing_routes = sorted(
        route_names - set(routes)
    )

    if missing_routes:
        raise RuntimeError(
            "Correction operator omitted routes: "
            f"{missing_routes}"
        )

    preserve = require_bool_mask(
        routes["preserve"],
        "route preserve",
        shape,
    )

    remove = require_bool_mask(
        routes["remove"],
        "route remove",
        shape,
    )

    ambiguous = require_bool_mask(
        routes["ambiguous"],
        "route ambiguous",
        shape,
    )

    # ------------------------------------------------------------
    # Frozen V7.8 route integrity
    # ------------------------------------------------------------

    if np.any(
        preserve & remove
    ):
        raise RuntimeError(
            "PRESERVE and REMOVE routes overlap"
        )

    if np.any(
        preserve & ambiguous
    ):
        raise RuntimeError(
            "PRESERVE and AMBIGUOUS routes overlap"
        )

    if np.any(
        remove & ambiguous
    ):
        raise RuntimeError(
            "REMOVE and AMBIGUOUS routes overlap"
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
        missing_route = int(
            np.sum(
                mask & ~routed
            )
        )

        escaped_route = int(
            np.sum(
                routed & ~mask
            )
        )

        raise RuntimeError(
            "V7.8 routes do not exactly partition "
            "the prepared mask: "
            f"missing={missing_route}, "
            f"outside={escaped_route}"
        )

    # ------------------------------------------------------------
    # Frozen V7.8 output contract
    # ------------------------------------------------------------

    if np.any(
        baseline[preserve]
        != observed[preserve]
    ):
        raise RuntimeError(
            "V7.8 PRESERVE does not reproduce observed bytes"
        )

    if np.any(
        baseline[remove]
        != paper[remove]
    ):
        raise RuntimeError(
            "V7.8 REMOVE does not reproduce paper bytes"
        )

    if np.any(
        baseline[ambiguous]
        != telea[ambiguous]
    ):
        raise RuntimeError(
            "V7.8 AMBIGUOUS does not reproduce Telea bytes"
        )

    if np.any(
        baseline[~mask]
        != observed[~mask]
    ):
        raise RuntimeError(
            "V7.8 changed pixels outside the prepared mask"
        )

    # ------------------------------------------------------------
    # Frozen A5-v1 candidate population
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
        missing_eligible = int(
            np.sum(
                expected_eligible
                & ~eligible_a5_v1
            )
        )

        unexpected_eligible = int(
            np.sum(
                eligible_a5_v1
                & ~expected_eligible
            )
        )

        raise RuntimeError(
            "A5-v1 eligible population mismatch: "
            f"missing={missing_eligible}, "
            f"unexpected={unexpected_eligible}"
        )

    if np.any(
        print_mask & ~eligible_a5_v1
    ):
        raise RuntimeError(
            "A5-v1 PRINT exists outside the "
            "eligible population"
        )

    # ------------------------------------------------------------
    # Frozen A5-v1 output contract
    # ------------------------------------------------------------

    if np.any(
        a5[print_mask]
        != observed[print_mask]
    ):
        raise RuntimeError(
            "A5-v1 PRINT output does not equal "
            "the exact observed byte"
        )

    if np.any(
        a5[~print_mask]
        != baseline[~print_mask]
    ):
        raise RuntimeError(
            "A5-v1 non-PRINT output differs "
            "from frozen V7.8"
        )

    if np.any(
        a5[~mask]
        != observed[~mask]
    ):
        raise RuntimeError(
            "A5-v1 changed pixels outside "
            "the authoritative mask"
        )

    changed = (
        a5 != baseline
    )

    expected_changed = (
        print_mask
        & (observed != baseline)
    )

    if not np.array_equal(
        changed,
        expected_changed,
    ):
        unexpected_changes = int(
            np.sum(
                changed
                & ~expected_changed
            )
        )

        missing_changes = int(
            np.sum(
                expected_changed
                & ~changed
            )
        )

        raise RuntimeError(
            "A5-v1 page delta does not equal "
            "effective PRINT restoration: "
            f"unexpected={unexpected_changes}, "
            f"missing={missing_changes}"
        )

    if np.any(
        changed
        & (a5 != observed)
    ):
        raise RuntimeError(
            "A5-v1 changed byte does not "
            "equal observed"
        )

    # ------------------------------------------------------------
    # Probability and decision arrays
    # ------------------------------------------------------------

    p_print = np.asarray(
        result["p_print"]
    )

    if p_print.shape != shape:
        raise RuntimeError(
            "p_print shape mismatch: "
            f"expected {shape}, "
            f"found {p_print.shape}"
        )

    if not np.isfinite(
        p_print[eligible_a5_v1]
    ).all():
        raise RuntimeError(
            "p_print contains non-finite values "
            "at eligible coordinates"
        )

    if np.any(
        p_print[eligible_a5_v1] < 0.0
    ) or np.any(
        p_print[eligible_a5_v1] > 1.0
    ):
        raise RuntimeError(
            "p_print lies outside 0..1"
        )

    provisional = np.asarray(
        result["provisional"]
    )

    final = np.asarray(
        result["final"]
    )

    if provisional.shape != shape:
        raise RuntimeError(
            "Provisional decision array "
            f"shape mismatch: {provisional.shape}"
        )

    if final.shape != shape:
        raise RuntimeError(
            "Final decision array "
            f"shape mismatch: {final.shape}"
        )

    valid_classes = {
        "PRINT",
        "ANNOTATION",
        "UNCERTAIN",
        "INELIGIBLE",
    }

    provisional_classes = set(
        provisional.astype(str).ravel()
    )

    final_classes = set(
        final.astype(str).ravel()
    )

    unexpected_provisional = (
        provisional_classes
        - valid_classes
    )

    unexpected_final = (
        final_classes
        - valid_classes
    )

    if unexpected_provisional:
        raise RuntimeError(
            "Unexpected provisional A5-v1 classes: "
            f"{sorted(unexpected_provisional)}"
        )

    if unexpected_final:
        raise RuntimeError(
            "Unexpected final A5-v1 classes: "
            f"{sorted(unexpected_final)}"
        )

    # Every eligible coordinate must have a real decision.
    if np.any(
        provisional[eligible_a5_v1]
        == "INELIGIBLE"
    ):
        raise RuntimeError(
            "Eligible A5-v1 coordinate has no "
            "provisional decision"
        )

    if np.any(
        final[eligible_a5_v1]
        == "INELIGIBLE"
    ):
        raise RuntimeError(
            "Eligible A5-v1 coordinate has no "
            "final decision"
        )

    # Every non-eligible coordinate must remain outside the
    # classifier population.
    if np.any(
        provisional[~eligible_a5_v1]
        != "INELIGIBLE"
    ):
        raise RuntimeError(
            "A5-v1 provisional decision exists "
            "outside eligibility"
        )

    if np.any(
        final[~eligible_a5_v1]
        != "INELIGIBLE"
    ):
        raise RuntimeError(
            "A5-v1 final decision exists "
            "outside eligibility"
        )

    expected_print_mask = (
        final == "PRINT"
    )

    if not np.array_equal(
        print_mask,
        expected_print_mask,
    ):
        missing_print = int(
            np.sum(
                expected_print_mask
                & ~print_mask
            )
        )

        unexpected_print = int(
            np.sum(
                print_mask
                & ~expected_print_mask
            )
        )

        raise RuntimeError(
            "A5-v1 PRINT mask does not equal "
            "final PRINT decisions: "
            f"missing={missing_print}, "
            f"unexpected={unexpected_print}"
        )

    # Run filtering may demote PRINT to UNCERTAIN.
    # It must not manufacture a new PRINT decision.
    if np.any(
        (final == "PRINT")
        & (provisional != "PRINT")
    ):
        raise RuntimeError(
            "A5-v1 run filter created a new PRINT decision"
        )

    if np.any(
        (provisional == "ANNOTATION")
        & (final != "ANNOTATION")
    ):
        raise RuntimeError(
            "A5-v1 run filter altered "
            "an ANNOTATION decision"
        )

    # ------------------------------------------------------------
    # Normalised result returned to main.py
    # ------------------------------------------------------------

    result["mask"] = mask
    result["telea"] = telea
    result["paper"] = paper

    result["routes"] = {
        **routes,
        "preserve": preserve,
        "remove": remove,
        "ambiguous": ambiguous,
    }

    result["geometry_t80"] = baseline
    result["a1"] = baseline
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
        result = get_correction_operator().run(
            observed=observed,
            reviewed_seed=reviewed_seed,
            page=sid,
        )

        result = validate_operational_result(
            observed=observed,
            result=result,
        )

        mask = result["mask"]
        routes = result["routes"]

        baseline = result["geometry_t80"]
        a5 = result["a5"]

        print_mask = result["print_mask"]
        eligible = result["eligible_a5_v1"]

        p_print = result["p_print"]
        provisional = result["provisional"]
        final = result["final"]

        preserve = routes["preserve"]
        remove = routes["remove"]
        ambiguous = routes["ambiguous"]

        outside_mask = ~mask
        non_print = ~print_mask

        changed = (
            a5 != baseline
        )

        expected_changed = (
            print_mask
            & (observed != baseline)
        )

        # --------------------------------------------------------
        # Persist canonical V7.8 evidence
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
            baseline,
        )

        # Compatibility alias for the existing UI/package.
        write_png(
            directory / "a1_corrected.png",
            baseline,
        )

        # --------------------------------------------------------
        # Persist frozen A5-v1 evidence
        # --------------------------------------------------------

        write_png(
            directory / "a5_eligible.png",
            eligible,
        )

        # Compatibility alias for any existing frontend route.
        write_png(
            directory / "a5_eligible_ambiguous.png",
            eligible,
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
            directory / "a5_delta_from_v7_8.png",
            changed,
        )

        # Compatibility alias for the existing frontend/package.
        write_png(
            directory / "a5_delta_from_a1.png",
            changed,
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
        # Whole-page integrity audit
        # --------------------------------------------------------

        audit = {
            "created_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "session_id": sid,
            "app_version": APP_VERSION,
            "correction_contract": CORRECTION_CONTRACT,

            "v7_8_baseline": "geometry_t80",
            "a5_action": "COPY_EXACT_OBSERVED_BYTE",
            "a5_fallback": "geometry_t80",

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

            "a5_eligible_pixels": int(
                eligible.sum()
            ),
            "a5_final_print_pixels": int(
                print_mask.sum()
            ),
            "a5_effectively_changed_pixels": int(
                changed.sum()
            ),

            "a5_final_annotation_pixels": int(
                np.sum(
                    final[eligible]
                    == "ANNOTATION"
                )
            ),
            "a5_final_uncertain_pixels": int(
                np.sum(
                    final[eligible]
                    == "UNCERTAIN"
                )
            ),

            # Required zero-value integrity fields.
            "v7_8_outside_mask_drift": int(
                np.sum(
                    baseline[outside_mask]
                    != observed[outside_mask]
                )
            ),

            "a5_outside_mask_drift": int(
                np.sum(
                    a5[outside_mask]
                    != observed[outside_mask]
                )
            ),

            "a5_non_print_v7_8_regression": int(
                np.sum(
                    a5[non_print]
                    != baseline[non_print]
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
                    changed
                    & (a5 != observed)
                )
            ),

            "a5_unexpected_changed_byte": int(
                np.sum(
                    changed
                    & ~expected_changed
                )
            ),

            "a5_missing_effective_print_change": int(
                np.sum(
                    expected_changed
                    & ~changed
                )
            ),

            "print_outside_eligibility": int(
                np.sum(
                    print_mask
                    & ~eligible
                )
            ),

            "print_outside_authoritative_mask": int(
                np.sum(
                    print_mask
                    & ~mask
                )
            ),

            # Telea is diagnostic as a separate published output,
            # but it contributes operationally to geometry_t80 at
            # V7.8 AMBIGUOUS coordinates.
            "telea_contributes_to_v7_8_ambiguous": bool(
                ambiguous.any()
            ),

            "clean_image_used_at_runtime": False,
            "historical_notebook91_drives_output": False,
        }

        zero_fields = [
            "v7_8_outside_mask_drift",
            "a5_outside_mask_drift",
            "a5_non_print_v7_8_regression",
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
            "a5_changed_from_v7_8": audit[
                "a5_effectively_changed_pixels"
            ],
            "outside_mask_drift": audit[
                "a5_outside_mask_drift"
            ],
            "non_print_v7_8_regression": audit[
                "a5_non_print_v7_8_regression"
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
def diagnostic(sid: str, name: str):
    names = {
        "mask": "prepared_dilation2_mask.png",
        "telea": "telea_diagnostic.png",
        "paper": "paper_estimate.png",
        "preserve": "route_preserve.png",
        "remove": "route_remove.png",
        "ambiguous": "route_ambiguous.png",
        "a1": "a1_corrected.png",
        "eligible_a5_v1": "a5_eligible_ambiguous.png",
        "clean": "a5_print_mask.png",
        "delta": "a5_delta_from_a1.png",
        "a5": "a5_corrected.png",

        # Temporary compatibility aliases for the current front end.
        "ring2": "a5_eligible_ambiguous.png",
        "safe_remove": "route_remove.png",
        "m5": "a5_corrected.png",
        "v78": "a1_corrected.png",
        "v88": "a5_corrected.png",
    }

    filename = names.get(name)

    if filename is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown diagnostic",
        )

    return png(read_session_png(sid, filename))


# ---------------------------------------------------------------------------
# Review and packaging
# ---------------------------------------------------------------------------

@app.post("/api/session/{sid}/approval")
def approval(sid: str, value: Approval):
    allowed = {
        "approved",
        "return_to_mask",
        "flag_failure",
    }

    if value.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Invalid status",
        )

    directory = store.folder(sid)

    (
        directory / "whole_page_review.json"
    ).write_text(
        json.dumps(
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "status": value.status,
                "note": value.note,
            },
            indent=2,
        )
    )

    return {
        "ok": True,
        "status": value.status,
    }


@app.post("/api/session/{sid}/finish")
def finish(sid: str):
    directory = store.folder(sid)
    session = store.load(sid)

    write_png(
        directory / "reviewed_authoritative_seed_mask.png",
        session["working"],
    )

    required = [
        "annotated_page.png",
        "reviewed_authoritative_seed_mask.png",
        "prepared_dilation2_mask.png",
        "route_preserve.png",
        "route_remove.png",
        "route_ambiguous.png",
        "paper_estimate.png",
        "telea_diagnostic.png",
        "a1_corrected.png",
        "a5_eligible_ambiguous.png",
        "a5_print_mask.png",
        "a5_corrected.png",
        "a5_delta_from_a1.png",
        "operational_correction_audit.json",
    ]

    missing = [
        filename
        for filename in required
        if not (directory / filename).exists()
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

    output = directory / "hf_04_5_training_and_audit.zip"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": "HF-04.5",
        "correction_contract": CORRECTION_CONTRACT,
        "v78_policy_id": V78_POLICY_ID,
        "a5_policy_id": A5_POLICY_ID,
        "training_pair": [
            "annotated_page.png",
            "reviewed_authoritative_seed_mask.png",
        ],
        "raw_yolo_is_training_truth": False,
        "telea_is_diagnostic_only": True,
        "historical_notebook91_is_operational": False,
        "files": [],
    }

    for path in sorted(directory.iterdir()):
        if path.is_file() and path != output:
            manifest["files"].append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                }
            )

    (
        directory / "package_manifest.json"
    ).write_text(json.dumps(manifest, indent=2))

    with zipfile.ZipFile(
        output,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path != output:
                archive.write(path, path.name)

    return {
        "download_url": f"/api/session/{sid}/download",
    }


@app.get("/api/session/{sid}/download")
def download(sid: str):
    path = (
        store.folder(sid)
        / "hf_04_5_training_and_audit.zip"
    )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Finish package has not been created",
        )

    return FileResponse(
        path,
        filename="hf_04_5_training_and_audit.zip",
    )


# ---------------------------------------------------------------------------
# Health and startup evidence
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    operator = get_correction_operator()

    return {
        "ok": True,
        "version": APP_VERSION,
        "correction_contract": CORRECTION_CONTRACT,
        "v78_policy_id": getattr(
            operator,
            "v78_policy_id",
            V78_POLICY_ID,
        ),
        "a5_policy_id": getattr(
            operator,
            "a5_policy_id",
            A5_POLICY_ID,
        ),
        "bundle_id": getattr(
            operator,
            "bundle_id",
            None,
        ),
        "device": get_device(),
        "telea_operational": False,
        "historical_notebook91_operational": False,
    }


@app.get("/startup-evidence")
def download_startup_evidence():
    root = STARTUP_EVIDENCE_DIR

    verdict = root / "startup_verdict.json"
    a1_replay = (
        root
        / "operational_a1_replay"
        / "replay_result.json"
    )
    a5_replay = (
        root
        / "safe_cleanup_a5_replay"
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
    runtime = root / "runtime_software_identities.json"

    required = [
        verdict,
        a1_replay,
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
                    "Startup commissioning evidence incomplete"
                ),
                "missing": missing,
            },
        )

    value = json.loads(verdict.read_text())

    if value.get("passed") is not True:
        raise HTTPException(
            status_code=503,
            detail=(
                "Startup commissioning verdict is not PASS"
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
        filename="hf_04_5_startup_evidence.zip",
    )


@app.get("/startup-evidence-status")
def startup_evidence_status():
    root = STARTUP_EVIDENCE_DIR

    return {
        "root": str(root.resolve()),
        "root_exists": root.exists(),
        "files": (
            sorted(
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file()
            )
            if root.exists()
            else []
        ),
    }


# ---------------------------------------------------------------------------
# HF-04.5 batch
# ---------------------------------------------------------------------------

@app.post("/batch-a1-a5")
async def batch_a1_a5(file: UploadFile = File(...)):
    """
    HF-04.5 remains fail-closed until hf045_batch.run_batch has been
    updated to consume A5Operator and the operational A1/A5
    result contract.

    Do not send the historical Notebook 91 A5Operator into this route.
    """

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Upload one .zip file",
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "HF-04.5 batch is paused until hf045_batch.run_batch "
            "is updated and regression-tested for operational A1 "
            "plus AMBIGUOUS-only safe-cleanup A5."
        ),
    )