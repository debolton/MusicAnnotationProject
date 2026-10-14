import cv2
import base64
import json
import numpy as np

from pathlib import Path
from datetime import (
    datetime,
    timezone
)


class ReviewSaver:

    def __init__(
        self,
        aligned_dir,
        corrected_dir,
        audit_dir,
        preview_dir,
        queue_manager
    ):

        self.aligned_dir = aligned_dir
        self.corrected_dir = corrected_dir
        self.audit_dir = audit_dir
        self.preview_dir = preview_dir

        self.queue = queue_manager

    def decode_mask(
        self,
        value,
        shape
    ):

        if not value:

            return np.zeros(
                shape,
                np.uint8
            )

        payload = (
            value.split(",", 1)[1]
            if "," in value
            else value
        )

        raw = base64.b64decode(
            payload
        )

        array = np.frombuffer(
            raw,
            np.uint8
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_GRAYSCALE
        )

        if image.shape != shape:

            image = cv2.resize(
                image,
                (
                    shape[1],
                    shape[0]
                ),
                interpolation=
                cv2.INTER_NEAREST
            )

        return np.where(
            image > 0,
            255,
            0
        ).astype(np.uint8)

    def save_review(
        self,
        source,
        draft,
        additions,
        erasures,
        notes,
        source_md5
    ):

        output_name = (
            f"Page"
            f"{self.queue.next_page_number():03d}"
        )

        final = (
            (
                (draft > 0)
                | (additions > 0)
            )
            & ~(erasures > 0)
        ).astype(np.uint8) * 255

        image_output = (
            self.aligned_dir /
            f"{output_name}.png"
        )

        mask_output = (
            self.corrected_dir /
            f"{output_name}.png"
        )

        cv2.imwrite(
            str(image_output),
            source
        )

        cv2.imwrite(
            str(mask_output),
            final
        )

        audit = {
            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "page":
                output_name,

            "source_md5":
                source_md5,

            "notes":
                notes
        }

        (
            self.audit_dir /
            f"{output_name}.json"
        ).write_text(
            json.dumps(
                audit,
                indent=2
            )
        )

        return output_name
