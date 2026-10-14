# modules/gst_evaluator.py

import csv
import json
import cv2
import numpy as np

from pathlib import Path


class GSTEvaluator:

    def __init__(
        self,
        model,
        release,
        dataset,
    ):

        self.model = model
        self.release = release
        self.dataset = dataset

        cfg = release.config

        self.tile_size = int(
            cfg["gold_standard"]["tile_size"]
        )

        self.tile_stride = int(
            cfg["gold_standard"]["tile_stride"]
        )

        self.confidence = float(
            release.training[
                "prediction_confidence"
            ]
        )

        self.image_size = int(
            release.training[
                "image_size"
            ]
        )

        self.background_blur_sigma = 75

        self.clahe_clip_limit = 2.0

        self.clahe_grid_size = (
            8,
            8,
        )

        self.results = []
        self.instance_results = []

        self.root = (
            release.predictions_dir /
            "gst_normalisation_comparison"
        )

        self.normalised_dir = (
            self.root /
            "normalised_pages"
        )

        self.prediction_dir = (
            self.root /
            "predictions"
        )

        self.comparison_dir = (
            self.root /
            "comparisons"
        )

    # ------------------------------------------
    # Initialisation
    # ------------------------------------------

    def prepare_folders(self):

        for folder in [

            self.root,

            self.normalised_dir,

            self.prediction_dir,

            self.comparison_dir,

        ]:

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

    # ------------------------------------------
    # Image processing
    # ------------------------------------------

    def normalise_background(
        self,
        image
    ):

        background = cv2.GaussianBlur(
            image,
            (0, 0),
            self.background_blur_sigma
        )

        background = np.maximum(
            background,
            1
        )

        return cv2.divide(
            image,
            background,
            scale=255
        )

    def apply_clahe(
        self,
        image
    ):

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
