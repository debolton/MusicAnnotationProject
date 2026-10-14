# modules/confidence_maps.py

import cv2
import numpy as np


class ConfidenceMapGenerator:

    def __init__(
        self,
        predictor,
        tile_size=640,
        stride=320
    ):

        self.predictor = predictor

        self.tile_size = tile_size

        self.stride = stride

    def positions(self, n):

        if n <= self.tile_size:
            return [0]

        values = list(
            range(
                0,
                n - self.tile_size + 1,
                self.stride
            )
        )

        if (
            values[-1]
            != n - self.tile_size
        ):
            values.append(
                n - self.tile_size
            )

        return values

    def build(
        self,
        gray,
        confidence=0.2,
        image_size=640
    ):

        colour = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        h, w = gray.shape

        padded_h = max(
            h,
            self.tile_size
        )

        padded_w = max(
            w,
            self.tile_size
        )

        canvas = np.full(
            (
                padded_h,
                padded_w,
                3
            ),
            255,
            np.uint8
        )

        canvas[:h, :w] = colour

        confidence_map = np.zeros(
            (
                padded_h,
                padded_w
            ),
            np.float32
        )

        for y in self.positions(
            padded_h
        ):

            for x in self.positions(
                padded_w
            ):

                result = (
                    self.predictor
                    .predict_tile(
                        canvas[
                            y:y+self.tile_size,
                            x:x+self.tile_size
                        ],
                        confidence,
                        image_size
                    )
                )

                if result.masks is None:
                    continue

                masks = (
                    result.masks
                    .data
                    .cpu()
                    .numpy()
                )

                scores = (
                    result.boxes
                    .conf
                    .cpu()
                    .numpy()
                )

                for mask, score in zip(
                    masks,
                    scores
                ):

                    mask = cv2.resize(
                        mask,
                        (
                            self.tile_size,
                            self.tile_size
                        ),
                        interpolation=
                        cv2.INTER_NEAREST
                    ) > 0.5

                    region = confidence_map[
                        y:y+self.tile_size,
                        x:x+self.tile_size
                    ]

                    region[
                        mask &
                        (score > region)
                    ] = score

        return confidence_map[
            :h,
            :w
        ]
