# modules/validator.py

import cv2
import re
import numpy as np


class DatasetValidator:

    def __init__(self, cfg):

        self.cfg = cfg

    def normalise_mask(
        self,
        mask,
        threshold=0,
        min_area=30
    ):

        binary = np.where(
            mask > threshold,
            255,
            0
        ).astype(np.uint8)

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                8
            )
        )

        clean = (
            np.zeros_like(binary)
        )

        for i in range(1, count):

            if (
                stats[
                    i,
                    cv2.CC_STAT_AREA
                ]
                >= min_area
            ):
                clean[
                    labels == i
                ] = 255

        return clean

    def validate_pages(self):

        valid = {}

        for mask_path in (
            self.cfg.corrected_mask_dir
            .glob("*")
        ):

            page = mask_path.stem

            if not re.fullmatch(
                r"Page\d+",
                page,
                re.I
            ):
                continue

            image_path = (
                self.cfg.aligned_dir /
                f"{page}.png"
            )

            if not image_path.exists():
                continue

            aligned = cv2.imread(
                str(image_path),
                0
            )

            mask = cv2.imread(
                str(mask_path),
                0
            )

            if (
                aligned is None
                or mask is None
            ):
                continue

            if aligned.shape != mask.shape:
                continue

            valid[page] = {
                "aligned":
                    image_path,
                "mask":
                    mask_path
            }

        return valid
