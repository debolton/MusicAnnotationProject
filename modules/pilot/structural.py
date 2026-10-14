# modules/pilot/structural.py

import cv2
import numpy as np


# --------------------------------------------------
# Configuration defaults
# --------------------------------------------------

MASK_DILATION = 2
CROP_MARGIN = 48
GEOMETRY_THRESHOLD = 80
BAND_WIDTH = 4
CONTEXT_RING_RADIUS = 18


# --------------------------------------------------
# Shared helper
# --------------------------------------------------

def immutable(
    original,
    candidate,
    mask
):

    out = original.copy()

    out[mask > 0] = candidate[
        mask > 0
    ]

    assert np.array_equal(
        out[mask == 0],
        original[mask == 0]
    )

    return out


# --------------------------------------------------
# Binary mask handling
# --------------------------------------------------

def binary(mask):

    return np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)


def prepare_mask(
    seed,
    dilation=MASK_DILATION
):

    mask = binary(seed)

    if dilation > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * dilation + 1,
                2 * dilation + 1
            )
        )

        mask = cv2.dilate(
            mask,
            kernel
        )

    return mask


# --------------------------------------------------
# Connected component crop merging
# --------------------------------------------------

def padded_components(mask):

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (mask > 0).astype(
                np.uint8
            ),
            8
        )
    )

    height, width = mask.shape

    boxes = []

    for index in range(1, count):

        x, y, bw, bh, area = (
            stats[index]
        )

        if area >= 8:

            boxes.append(

                (
                    max(
                        0,
                        x - CROP_MARGIN
                    ),

                    max(
                        0,
                        y - CROP_MARGIN
                    ),

                    min(
                        width,
                        x + bw + CROP_MARGIN
                    ),

                    min(
                        height,
                        y + bh + CROP_MARGIN
                    ),
                )

            )

    changed = True

    while changed:

        changed = False

        merged = []

        while boxes:

            (
                ax0,
                ay0,
                ax1,
                ay1
            ) = boxes.pop()

            remaining = []

            for (
                bx0,
                by0,
                bx1,
                by1
            ) in boxes:

                overlap = not (

                    ax1 < bx0
                    or bx1 < ax0
                    or ay1 < by0
                    or by1 < ay0

                )

                if overlap:

                    ax0 = min(
                        ax0,
                        bx0
                    )

                    ay0 = min(
                        ay0,
                        by0
                    )

                    ax1 = max(
                        ax1,
                        bx1
                    )

                    ay1 = max(
                        ay1,
                        by1
                    )

                    changed = True

                else:

                    remaining.append(

                        (
                            bx0,
                            by0,
                            bx1,
                            by1
                        )

                    )

            boxes = remaining

            merged.append(

                (
                    ax0,
                    ay0,
                    ax1,
                    ay1
                )

            )

        boxes = merged

    return boxes


# --------------------------------------------------
# Telea reconstruction
# --------------------------------------------------

def telea_reconstruct(
    observed,
    mask
):

    result = observed.copy()

    for (
        x0,
        y0,
        x1,
        y1
    ) in padded_components(mask):

        image = observed[
            y0:y1,
            x0:x1
        ]

        local_mask = mask[
            y0:y1,
            x0:x1
        ]

        candidate = cv2.inpaint(
            image,
            local_mask,
            3,
            cv2.INPAINT_TELEA
        )

        local = result[
            y0:y1,
            x0:x1
        ]

        local[
            local_mask > 0
        ] = candidate[
            local_mask > 0
        ]

    return immutable(
        observed,
        result,
        mask
    )


# --------------------------------------------------
# Background estimation
# --------------------------------------------------

def background_surface(
    observed,
    mask,
    sigma=45.0
):

    visible = (
        (mask == 0)
        &
        (observed >= 200)
    )

    values = (
        observed.astype(
            np.float32
        )
        * visible
    )

    weights = visible.astype(
        np.float32
    )

    numerator = cv2.GaussianBlur(
        values,
        (0, 0),
        sigma
    )

    denominator = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigma
    )

    surface = (
        numerator
        /
        np.maximum(
            denominator,
            1e-4
        )
    )

    fallback = cv2.GaussianBlur(
        observed,
        (0, 0),
        sigma
    )

    return np.clip(

        np.where(
            denominator > 0.02,
            surface,
            fallback
        ),

        0,
        255

    ).astype(np.uint8)


def outside_dark(
    observed,
    mask
):

    return (
        (observed < 190)
        &
        (mask == 0)
    )


# --------------------------------------------------
# Structural support maps
# --------------------------------------------------

def support_maps(
    observed,
    mask
):

    dark = outside_dark(
        observed,
        mask
    )

    horizontal = (

        cv2.morphologyEx(
            dark.astype(
                np.uint8
            ) * 255,
            cv2.MORPH_OPEN,
            np.ones(
                (1, 35),
                np.uint8
            )
        ) > 0

    )

    vertical = (

        cv2.morphologyEx(
            dark.astype(
                np.uint8
            ) * 255,
            cv2.MORPH_OPEN,
            np.ones(
                (15, 1),
                np.uint8
            )
        ) > 0

    )

    (
        count,
        labels,
        stats,
        _
    ) = cv2.connectedComponentsWithStats(
        dark.astype(
            np.uint8
        ),
        8
    )

    compact = np.zeros_like(
        dark
    )

    for index in range(
        1,
        count
    ):

        (
            x,
            y,
            width,
            height,
            area
        ) = stats[index]

        if (
            3 <= area <= 650
            and width <= 40
            and height <= 40
        ):

            compact[
                labels == index
            ] = True

    kernel = np.ones(

        (
            2 * BAND_WIDTH + 1,
            2 * BAND_WIDTH + 1
        ),

        np.uint8

    )

    return (

        cv2.dilate(
            horizontal.astype(
                np.uint8
            ),
            kernel
        ) > 0,

        cv2.dilate(
            vertical.astype(
                np.uint8
            ),
            kernel
        ) > 0,

        cv2.dilate(
            compact.astype(
                np.uint8
            ),
            kernel
        ) > 0,

    )


# --------------------------------------------------
# Frozen geometry_t80
# --------------------------------------------------

def geometry_t80(
    observed,
    mask,
    telea
):

    paper = background_surface(
        observed,
        mask
    )

    normalised = cv2.divide(
        observed,
        np.maximum(
            paper,
            1
        ),
        scale=255
    )

    (
        horizontal,
        vertical,
        compact
    ) = support_maps(
        observed,
        mask
    )

    m = mask > 0

    preserve = (

        m
        &
        (
            horizontal
            |
            vertical
            |
            compact
        )
        &
        (observed < 200)

    )

    remove = (

        m
        &
        ~preserve
        &
        (
            observed
            >= GEOMETRY_THRESHOLD
        )
        &
        (
            normalised
            >= GEOMETRY_THRESHOLD
        )

    )

    ambiguous = (

        m
        &
        ~preserve
        &
        ~remove

    )

    result = observed.copy()

    result[
        remove
    ] = paper[
        remove
    ]

    result[
        ambiguous
    ] = telea[
        ambiguous
    ]

    result[
        preserve
    ] = observed[
        preserve
    ]

    return (

        immutable(
            observed,
            result,
            mask
        ),

        {
            "preserve":
                preserve,

            "remove":
                remove,

            "ambiguous":
                ambiguous,

            "horizontal":
                horizontal,

            "vertical":
                vertical,

            "compact":
                compact,

            "paper":
                paper,

            "normalised":
                normalised,
        }

    )


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def surrounding_features(
    observed,
    component,
    routes
):

    ys, xs = np.where(
        component
    )

    x0 = xs.min()
    x1 = xs.max() + 1

    y0 = ys.min()
    y1 = ys.max() + 1

    area = int(
        component.sum()
    )

    width = x1 - x0
    height = y1 - y0

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2
                * CONTEXT_RING_RADIUS
                + 1,

                2
                * CONTEXT_RING_RADIUS
                + 1
            )
        )
    )

    ring = (

        cv2.dilate(
            component.astype(
                np.uint8
            ),
            kernel
        ) > 0

    ) & ~component

    values = observed[
        ring
    ]

    edges = (

        cv2.Canny(
            observed,
            70,
            170
        ) > 0

    )

    depth = (
        cv2.distanceTransform(
            component.astype(
                np.uint8
            ),
            cv2.DIST_L2,
            5
        )
    )

    def fraction(name):

        return float(

            np.sum(
                routes[name]
                & component
            )

            / area

        )

    return {

        "w":
            width,

        "h":
            height,

        "area":
            area,

        "aspect_ratio":
            max(
                width,
                height
            )
            /
            max(
                min(
                    width,
                    height
                ),
                1
            ),

        "fill_ratio":
            area
            /
            max(
                width * height,
                1
            ),

        "touches_page_border":
            bool(
                x0 == 0
                or y0 == 0
                or x1
                == observed.shape[1]
                or y1
                == observed.shape[0]
            ),

        "context_pixels":
            int(ring.sum()),

        "context_edge_density":
            float(
                edges[ring].mean()
            )
            if ring.any()
            else 0.0,

        "context_dark_density":
            float(
                (
                    values < 190
                ).mean()
            )
            if values.size
            else 0.0,

        "context_mean_intensity":
            float(
                values.mean()
            )
            if values.size
            else 255.0,

        "context_intensity_std":
            float(
                values.std()
            )
            if values.size
            else 0.0,

        "maximum_mask_depth":
            float(
                depth[
                    component
                ].max()
            ),

        "horizontal_fraction":
            fraction(
                "horizontal"
            ),

        "vertical_fraction":
            fraction(
                "vertical"
            ),

        "compact_fraction":
            fraction(
                "compact"
            ),

        "preserve_fraction":
            fraction(
                "preserve"
            ),

        "remove_fraction":
            fraction(
                "remove"
            ),

        "ambiguous_fraction":
            fraction(
                "ambiguous"
            ),

        "reconstruction_dependency_fraction":
            fraction(
                "ambiguous"
            ),

        "component_pixels":
            area,
    }
