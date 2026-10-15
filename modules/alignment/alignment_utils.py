# modules/alignment/alignment_utils.py

import cv2
import json
import hashlib
import numpy as np

from pathlib import Path


EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# File helpers
# --------------------------------------------------

def image_files(folder):

    folder = Path(folder)

    if not folder.exists():
        return []

    return sorted(

        p

        for p in folder.iterdir()

        if (
            p.is_file()
            and p.suffix.lower()
            in EXTENSIONS
        )

    )


def page_names(folder):

    return sorted(

        {
            p.stem

            for p

            in image_files(folder)
        }

    )


def find_image(
    folder,
    page_name,
):

    matches = [

        path

        for path

        in image_files(folder)

        if (
            path.stem.lower()
            ==
            page_name.lower()
        )

    ]

    if len(matches) > 1:

        raise RuntimeError(

            f"Multiple files "
            f"found for "
            f"{page_name} "
            f"in {folder}: "
            f"{matches}"

        )

    return (

        matches[0]

        if matches

        else None

    )


# --------------------------------------------------
# Hash helpers
# --------------------------------------------------

def md5_file(path):

    h = hashlib.md5()

    with open(path, "rb") as f:

        for block in iter(

            lambda:
            f.read(1024 * 1024),

            b""

        ):

            h.update(block)

    return h.hexdigest()


def object_hash(value):

    return hashlib.md5(

        json.dumps(
            value,
            sort_keys=True
        ).encode()

    ).hexdigest()


# --------------------------------------------------
# JSON helpers
# --------------------------------------------------

def read_json(
    path,
    default=None,
):

    try:

        path = Path(path)

        if path.exists():

            return json.loads(
                path.read_text()
            )

    except Exception:

        pass

    return default


def write_json_atomic(
    path,
    value,
):

    path = Path(path)

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(

        json.dumps(
            value,
            indent=2
        )

    )

    temp.replace(path)


# --------------------------------------------------
# File cleanup
# --------------------------------------------------

def remove_if_exists(path):

    path = Path(path)

    if path.exists():

        path.unlink()


# --------------------------------------------------
# Image helpers
# --------------------------------------------------

def create_overlay(
    base,
    mask,
    opacity=0.60,
    colour=(0, 0, 255),
):

    if base.ndim == 2:

        base_colour = (
            cv2.cvtColor(
                base,
                cv2.COLOR_GRAY2BGR
            )
        )

    else:

        base_colour = (
            base.copy()
        )

    overlay = (
        base_colour.copy()
    )

    overlay[
        mask > 0
    ] = colour

    return cv2.addWeighted(

        base_colour,
        1 - opacity,

        overlay,
        opacity,

        0

    )


# --------------------------------------------------
# Preview helper
# --------------------------------------------------

def show_images(
    paths,
    columns=2,
    limit=24,
):

    import matplotlib.pyplot as plt

    paths = list(paths)[:limit]

    if not paths:

        print(
            "No preview images found."
        )

        return

    rows = int(
        np.ceil(
            len(paths) / columns
        )
    )

    plt.figure(

        figsize=(
            columns * 7,
            rows * 6,
        )

    )

    for index, path in enumerate(paths):

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        plt.subplot(
            rows,
            columns,
            index + 1,
        )

        plt.imshow(

            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        )

        plt.title(
            Path(path).name
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()
