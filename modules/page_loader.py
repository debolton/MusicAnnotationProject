import cv2
import base64
import json
import numpy as np


class PageLoader:

    def __init__(
        self,
        queue,
        source_dir,
        draft_dir,
        confidence_dir
    ):

        self.queue = queue
        self.source_dir = source_dir
        self.draft_dir = draft_dir
        self.confidence_dir = confidence_dir

    def data_url(
        self,
        array,
        extension=".png"
    ):

        ok, encoded = cv2.imencode(
            extension,
            array
        )

        if not ok:
            raise RuntimeError(
                "Encoding failed"
            )

        return (
            "data:image/png;base64,"
            + base64.b64encode(
                encoded.tobytes()
            ).decode()
        )

    def confidence_to_png(
        self,
        confidence
    ):

        return np.clip(
            confidence * 255,
            0,
            255
        ).astype(np.uint8)

    def package_page(
        self,
        page_name
    ):

        source = self.queue[page_name]

        image = cv2.imread(
            str(source)
        )

        draft = cv2.imread(
            str(
                self.draft_dir /
                f"{source.stem}_draft.png"
            ),
            0
        )

        confidence_file = (
            self.confidence_dir /
            f"{source.stem}_confidence.npy"
        )

        confidence = np.load(
            confidence_file
        )

        return {

            "name":
                page_name,

            "source_path":
                str(source),

            "width":
                image.shape[1],

            "height":
                image.shape[0],

            "image":
                self.data_url(image),

            "draft":
                self.data_url(draft),

            "confidence":
                self.data_url(
                    self.confidence_to_png(
                        confidence
                    )
                )
        }
