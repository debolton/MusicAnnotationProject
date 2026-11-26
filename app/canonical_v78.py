import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frozen V7.8 constants
# ---------------------------------------------------------------------------

MASK_DILATION = 2
CROP_MARGIN = 48
GEOMETRY_THRESHOLD = 80
BAND_WIDTH = 4


# ---------------------------------------------------------------------------
# General validation and immutability helpers
# ---------------------------------------------------------------------------

def require_grayscale_uint8(
    value: np.ndarray,
    name: str,
) -> np.ndarray:
    value = np.asarray(value)

    if value.ndim != 2:
        raise RuntimeError(
            f"{name} must be a two-dimensional greyscale array; "
            f"found shape {value.shape}"
        )

    if value.dtype != np.uint8:
        if not np.issubdtype(value.dtype, np.number):
            raise RuntimeError(
                f"{name} must contain numeric values; "
                f"found dtype {value.dtype}"
            )

        if not np.isfinite(value).all():
            raise RuntimeError(
                f"{name} contains non-finite values"
            )

        if value.min() < 0 or value.max() > 255:
            raise RuntimeError(
                f"{name} contains values outside 0..255"
            )

        value = value.astype(np.uint8)

    return np.ascontiguousarray(value)


def require_same_shape(
    reference: np.ndarray,
    value: np.ndarray,
    name: str,
) -> None:
    if value.shape != reference.shape:
        raise RuntimeError(
            f"{name} shape mismatch: "
            f"expected {reference.shape}, found {value.shape}"
        )


def binary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise RuntimeError(
            f"Mask must be two-dimensional; found {mask.shape}"
        )

    if (
        np.issubdtype(mask.dtype, np.number)
        and not np.isfinite(mask).all()
    ):
        raise RuntimeError("Mask contains non-finite values")

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def immutable(
    original: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    original = require_grayscale_uint8(
        original,
        "original",
    )
    candidate = require_grayscale_uint8(
        candidate,
        "candidate",
    )
    mask = binary(mask)

    require_same_shape(original, candidate, "candidate")
    require_same_shape(original, mask, "mask")

    active = mask > 0

    output = original.copy()
    output[active] = candidate[active]

    if not np.array_equal(
        output[~active],
        original[~active],
    ):
        raise RuntimeError("Outside-mask drift")

    return output


# ---------------------------------------------------------------------------
# Authoritative mask preparation
# ---------------------------------------------------------------------------

def prepare_mask(
    seed: np.ndarray,
    dilation: int = MASK_DILATION,
) -> np.ndarray:
    if not isinstance(dilation, int) or dilation < 0:
        raise ValueError(
            "dilation must be a non-negative integer"
        )

    mask = binary(seed)

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * dilation + 1,
                2 * dilation + 1,
            ),
        )

        mask = cv2.dilate(
            mask,
            kernel,
        )

    return binary(mask)


# ---------------------------------------------------------------------------
# Diagnostic Telea reconstruction
# ---------------------------------------------------------------------------

def padded_components(
    mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    mask = binary(mask)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            8,
        )
    )

    del labels

    height, width = mask.shape
    boxes: list[tuple[int, int, int, int]] = []

    for component_id in range(1, count):
        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )
        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )
        component_width = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )
        component_height = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < 8:
            continue

        boxes.append(
            (
                max(0, x - CROP_MARGIN),
                max(0, y - CROP_MARGIN),
                min(
                    width,
                    x + component_width + CROP_MARGIN,
                ),
                min(
                    height,
                    y + component_height + CROP_MARGIN,
                ),
            )
        )

    changed = True

    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []

        while boxes:
            ax0, ay0, ax1, ay1 = boxes.pop()
            remaining = []

            for bx0, by0, bx1, by1 in boxes:
                disjoint = (
                    ax1 < bx0
                    or bx1 < ax0
                    or ay1 < by0
                    or by1 < ay0
                )

                if disjoint:
                    remaining.append(
                        (bx0, by0, bx1, by1)
                    )
                    continue

                ax0 = min(ax0, bx0)
                ay0 = min(ay0, by0)
                ax1 = max(ax1, bx1)
                ay1 = max(ay1, by1)
                changed = True

            boxes = remaining
            merged.append(
                (ax0, ay0, ax1, ay1)
            )

        boxes = merged

    return boxes


def telea_reconstruct(
    observed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    observed = require_grayscale_uint8(
        observed,
        "observed",
    )
    mask = binary(mask)

    require_same_shape(observed, mask, "mask")

    result = observed.copy()

    for x0, y0, x1, y1 in padded_components(mask):
        observed_crop = observed[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        candidate = cv2.inpaint(
            observed_crop,
            mask_crop,
            3,
            cv2.INPAINT_TELEA,
        )

        result_crop = result[y0:y1, x0:x1]
        active = mask_crop > 0
        result_crop[active] = candidate[active]

    result = immutable(
        observed,
        result,
        mask,
    )

    if not np.array_equal(
        result[mask == 0],
        observed[mask == 0],
    ):
        raise RuntimeError(
            "Telea reconstruction changed pixels "
            "outside the prepared mask"
        )

    return result


# ---------------------------------------------------------------------------
# Frozen V7.8 observable calculations
# ---------------------------------------------------------------------------

def background_surface(
    observed: np.ndarray,
    mask: np.ndarray,
    sigma: float = 45.0,
) -> np.ndarray:
    observed = require_grayscale_uint8(
        observed,
        "observed",
    )
    mask = binary(mask)

    require_same_shape(observed, mask, "mask")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    visible = (
        (mask == 0)
        & (observed >= 200)
    )

    values = (
        observed.astype(np.float32)
        * visible.astype(np.float32)
    )
    weights = visible.astype(np.float32)

    numerator = cv2.GaussianBlur(
        values,
        (0, 0),
        sigma,
    )
    denominator = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigma,
    )

    surface = numerator / np.maximum(
        denominator,
        1e-4,
    )

    fallback = cv2.GaussianBlur(
        observed,
        (0, 0),
        sigma,
    )

    result = np.where(
        denominator > 0.02,
        surface,
        fallback,
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def outside_dark(
    observed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return (
        (observed < 190)
        & (mask == 0)
    )


def support_maps(
    observed: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = require_grayscale_uint8(
        observed,
        "observed",
    )
    mask = binary(mask)

    require_same_shape(observed, mask, "mask")

    dark = outside_dark(
        observed,
        mask,
    )

    horizontal = (
        cv2.morphologyEx(
            dark.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            np.ones(
                (1, 35),
                np.uint8,
            ),
        )
        > 0
    )

    vertical = (
        cv2.morphologyEx(
            dark.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            np.ones(
                (15, 1),
                np.uint8,
            ),
        )
        > 0
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            dark.astype(np.uint8),
            8,
        )
    )

    compact = np.zeros_like(
        dark,
        dtype=bool,
    )

    for component_id in range(1, count):
        width = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )
        height = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if (
            3 <= area <= 650
            and width <= 40
            and height <= 40
        ):
            compact[labels == component_id] = True

    kernel = np.ones(
        (
            2 * BAND_WIDTH + 1,
            2 * BAND_WIDTH + 1,
        ),
        np.uint8,
    )

    horizontal = (
        cv2.dilate(
            horizontal.astype(np.uint8),
            kernel,
        )
        > 0
    )
    vertical = (
        cv2.dilate(
            vertical.astype(np.uint8),
            kernel,
        )
        > 0
    )
    compact = (
        cv2.dilate(
            compact.astype(np.uint8),
            kernel,
        )
        > 0
    )

    return horizontal, vertical, compact


# ---------------------------------------------------------------------------
# Frozen V7.8 routing
# ---------------------------------------------------------------------------

def calculate_routes(
    observed: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Calculate the frozen V7.8 route and support arrays.

    This function does not decide what pixel value operational A1
    places into AMBIGUOUS coordinates.
    """

    observed = require_grayscale_uint8(
        observed,
        "observed",
    )
    mask = binary(mask)

    require_same_shape(observed, mask, "mask")

    paper = background_surface(
        observed,
        mask,
    )

    normalised = cv2.divide(
        observed,
        np.maximum(paper, 1),
        scale=255,
    )

    horizontal, vertical, compact = support_maps(
        observed,
        mask,
    )

    active = mask > 0

    preserve = (
        active
        & (horizontal | vertical | compact)
        & (observed < 200)
    )

    remove = (
        active
        & ~preserve
        & (observed >= GEOMETRY_THRESHOLD)
        & (normalised >= GEOMETRY_THRESHOLD)
    )

    ambiguous = (
        active
        & ~preserve
        & ~remove
    )

    if np.any(preserve & remove):
        raise RuntimeError(
            "PRESERVE and REMOVE routes overlap"
        )

    if np.any(preserve & ambiguous):
        raise RuntimeError(
            "PRESERVE and AMBIGUOUS routes overlap"
        )

    if np.any(remove & ambiguous):
        raise RuntimeError(
            "REMOVE and AMBIGUOUS routes overlap"
        )

    routed = preserve | remove | ambiguous

    if not np.array_equal(routed, active):
        missing = int(
            np.sum(active & ~routed)
        )
        outside = int(
            np.sum(routed & ~active)
        )

        raise RuntimeError(
            "Routing does not partition the prepared mask: "
            f"missing={missing}, outside={outside}"
        )

    return {
        "preserve": preserve,
        "remove": remove,
        "ambiguous": ambiguous,
        "horizontal": horizontal,
        "vertical": vertical,
        "compact": compact,
        "paper": paper,
        "normalised": normalised,
    }


# ---------------------------------------------------------------------------
# Historical V7.8 comparator
# ---------------------------------------------------------------------------

def geometry_t80(
    observed: np.ndarray,
    mask: np.ndarray,
    telea: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Historical frozen V7.8 geometry_t80.

    This function is retained for historical replay and diagnostics.

    Historical semantics:
        PRESERVE  -> observed
        REMOVE    -> paper
        AMBIGUOUS -> Telea
    """

    observed = require_grayscale_uint8(
        observed,
        "observed",
    )
    mask = binary(mask)
    telea = require_grayscale_uint8(
        telea,
        "telea",
    )

    require_same_shape(observed, mask, "mask")
    require_same_shape(observed, telea, "telea")

    routes = calculate_routes(
        observed,
        mask,
    )

    historical = observed.copy()
    historical[routes["remove"]] = (
        routes["paper"][routes["remove"]]
    )
    historical[routes["ambiguous"]] = (
        telea[routes["ambiguous"]]
    )
    historical[routes["preserve"]] = (
        observed[routes["preserve"]]
    )

    historical = immutable(
        observed,
        historical,
        mask,
    )

    return historical, routes


# ---------------------------------------------------------------------------
# Corrected operational A1
# ---------------------------------------------------------------------------

def assemble_operational_a1(
    observed: np.ndarray,
    mask: np.ndarray,
    routes: dict,
) -> np.ndarray:
    """
    Corrected operational A1.

    Operational semantics:
        PRESERVE  -> observed
        REMOVE    -> paper
        AMBIGUOUS -> observed

    Telea does not contribute any operational A1 bytes.
    """

    observed = require_grayscale_uint8(
        observed,
        "observed",
    )
    mask = binary(mask)

    require_same_shape(observed, mask, "mask")

    required = {
        "preserve",
        "remove",
        "ambiguous",
        "paper",
    }

    missing = sorted(required - set(routes))
    if missing:
        raise RuntimeError(
            f"Routes missing required arrays: {missing}"
        )

    preserve = np.asarray(
        routes["preserve"],
        dtype=bool,
    )
    remove = np.asarray(
        routes["remove"],
        dtype=bool,
    )
    ambiguous = np.asarray(
        routes["ambiguous"],
        dtype=bool,
    )
    paper = require_grayscale_uint8(
        routes["paper"],
        "paper",
    )

    for name, value in (
        ("preserve", preserve),
        ("remove", remove),
        ("ambiguous", ambiguous),
        ("paper", paper),
    ):
        require_same_shape(
            observed,
            value,
            name,
        )

    active = mask > 0
    routed = preserve | remove | ambiguous

    if not np.array_equal(routed, active):
        raise RuntimeError(
            "Operational A1 routes do not exactly partition "
            "the prepared mask"
        )

    a1 = observed.copy()

    # PRESERVE remains exactly observed.
    a1[preserve] = observed[preserve]

    # REMOVE becomes the frozen paper estimate.
    a1[remove] = paper[remove]

    # AMBIGUOUS deliberately remains exactly observed.
    a1[ambiguous] = observed[ambiguous]

    a1 = immutable(
        observed,
        a1,
        mask,
    )

    if np.any(
        a1[preserve] != observed[preserve]
    ):
        raise RuntimeError(
            "A1 PRESERVE does not equal observed"
        )

    if np.any(
        a1[remove] != paper[remove]
    ):
        raise RuntimeError(
            "A1 REMOVE does not equal paper"
        )

    if np.any(
        a1[ambiguous] != observed[ambiguous]
    ):
        raise RuntimeError(
            "A1 AMBIGUOUS does not equal observed"
        )

    if np.any(
        a1[~active] != observed[~active]
    ):
        raise RuntimeError(
            "A1 changed pixels outside the prepared mask"
        )

    return a1


# ---------------------------------------------------------------------------
# Observable channel construction
# ---------------------------------------------------------------------------

def build_channels(
    observed: np.ndarray,
    mask: np.ndarray,
    routes: dict,
) -> dict:
    """
    Return the 11 observable channels used by the V8.2 model lineage.

    obs and norm deliberately remain uint8 0..255 here.
    The model runtime is responsible for applying the frozen /255
    normalisation during tensor construction.
    """

    observed = require_grayscale_uint8(
        observed,
        "observed",
    )
    mask = binary(mask)

    required = {
        "normalised",
        "horizontal",
        "vertical",
        "compact",
        "preserve",
        "remove",
        "ambiguous",
    }

    missing = sorted(required - set(routes))
    if missing:
        raise RuntimeError(
            f"Cannot build channels; routes missing: {missing}"
        )

    channels = {
        "obs": observed,
        "norm": require_grayscale_uint8(
            routes["normalised"],
            "normalised",
        ),
        "mask": mask > 0,
        "depth": cv2.distanceTransform(
            (mask > 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        ).astype(np.float32),
        "horizontal": np.asarray(
            routes["horizontal"],
            dtype=bool,
        ),
        "vertical": np.asarray(
            routes["vertical"],
            dtype=bool,
        ),
        "compact": np.asarray(
            routes["compact"],
            dtype=bool,
        ),
        "preserve": np.asarray(
            routes["preserve"],
            dtype=bool,
        ),
        "remove": np.asarray(
            routes["remove"],
            dtype=bool,
        ),
        "ambiguous": np.asarray(
            routes["ambiguous"],
            dtype=bool,
        ),
        "edge": (
            cv2.Canny(
                observed,
                70,
                170,
            )
            > 0
        ),
    }

    expected_order = (
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
    )

    if tuple(channels) != expected_order:
        raise RuntimeError(
            "Channel order changed unexpectedly: "
            f"{tuple(channels)}"
        )

    for name, value in channels.items():
        require_same_shape(
            observed,
            np.asarray(value),
            f"channel {name}",
        )

    return channels


# ---------------------------------------------------------------------------
# Main page builder
# ---------------------------------------------------------------------------

def build_v78(
    observed: np.ndarray,
    reviewed_seed: np.ndarray,
) -> dict:
    """
    Build the historical V7.8 diagnostics and corrected operational A1.

    Returned semantics:
        geometry_t80:
            historical V7.8 comparator using Telea in AMBIGUOUS

        a1:
            corrected operational baseline:
            PRESERVE  -> observed
            REMOVE    -> paper
            AMBIGUOUS -> observed

        eligible:
            proper operational A5 population, AMBIGUOUS only
    """

    observed = require_grayscale_uint8(
        observed,
        "observed",
    )

    seed = binary(reviewed_seed)

    require_same_shape(
        observed,
        seed,
        "reviewed_seed",
    )

    mask = prepare_mask(
        seed,
        MASK_DILATION,
    )

    telea = telea_reconstruct(
        observed,
        mask,
    )

    historical_geometry_t80, routes = geometry_t80(
        observed,
        mask,
        telea,
    )

    a1 = assemble_operational_a1(
        observed,
        mask,
        routes,
    )

    channels = build_channels(
        observed,
        mask,
        routes,
    )

    # Proper operational A5 is permitted to inspect AMBIGUOUS only.
    # Any optional dark-pixel restriction belongs in the frozen A5
    # policy/bundle and must not silently broaden this population.
    eligible = routes["ambiguous"].copy()

    if np.any(
        eligible
        & ~routes["ambiguous"]
    ):
        raise RuntimeError(
            "A5 eligibility includes non-AMBIGUOUS pixels"
        )

    if np.any(
        eligible
        & ~(mask > 0)
    ):
        raise RuntimeError(
            "A5 eligibility escapes the prepared mask"
        )

    ys, xs = np.where(eligible)

    if not np.array_equal(
        a1[mask == 0],
        observed[mask == 0],
    ):
        raise RuntimeError(
            "Operational A1 outside-mask drift"
        )

    return {
        # Reviewed, undilated HITL authority.
        "seed": seed,

        # Prepared dilation-2 intervention authority.
        "mask": mask,
        "prepared_mask": mask,

        # Comparator/diagnostic only.
        "telea": telea,
        "geometry_t80": historical_geometry_t80,
        "historical_geometry_t80": (
            historical_geometry_t80
        ),

        # Corrected operational baseline.
        "a1": a1,

        # Frozen route, support, paper and normalisation arrays.
        "routes": routes,
        "paper": routes["paper"],
        "normalised": routes["normalised"],

        # Ordered observable representation.
        "channels": channels,

        # Proper safe-cleanup A5 population.
        "eligible": eligible,
        "eligible_a5": eligible,

        # Page coordinates in deterministic NumPy order.
        "ys": ys.astype(np.int64),
        "xs": xs.astype(np.int64),
    }