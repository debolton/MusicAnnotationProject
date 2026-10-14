# modules/dataset_release.py

import json
import hashlib
import re

from pathlib import Path
from datetime import datetime

import torch


class DatasetRelease:

    REQUIRED_PUBLICATION_SCHEMA = 3

    def __init__(
        self,
        config_path,
        requested_dataset_version=None
    ):

        self.config_path = Path(config_path)

        self.requested_dataset_version = (
            requested_dataset_version
        )

        self.config = None
        self.release = None
        self.manifest = None

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def md5(path):

        h = hashlib.md5()

        with open(path, "rb") as f:

            for block in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):
                h.update(block)

        return h.hexdigest()

    @staticmethod
    def load_json(path, label):

        path = Path(path)

        if not path.exists():

            raise RuntimeError(
                f"Missing {label}: {path}"
            )

        return json.loads(
            path.read_text()
        )

    @staticmethod
    def page_prefix_from_tile(stem):

        match = re.match(
            r"^(.*)_y\d+_x\d+$",
            stem
        )

        return (
            match.group(1)
            if match
            else None
        )

    # --------------------------------------------------
    # Load release
    # --------------------------------------------------

    def load(self):

        if not self.config_path.exists():

            raise RuntimeError(
                f"Missing config file: "
                f"{self.config_path}"
            )

        self.config = json.loads(
            self.config_path.read_text()
        )

        self.root = Path(
            self.config["project_root"]
        )

        self.dataset_versions = (
            self.root /
            "dataset_versions"
        )

        self.current_dataset = (
            self.dataset_versions /
            "current_dataset.json"
        )

        self.gold = (
            self.root /
            "gold_standard_testing"
        )

        self.runs = (
            self.root /
            "training_runs"
        )

        self.predictions = (
            self.root /
            "predictions"
        )

        self.aligned = (
            self.root /
            "aligned"
        )

        if (
            self.requested_dataset_version
            is None
        ):

            pointer = self.load_json(
                self.current_dataset,
                "current dataset pointer"
            )

            self.dataset_version_name = (
                pointer["version_name"]
            )

            self.release_path = (
                self.root /
                pointer["release_file"]
            )

        else:

            self.dataset_version_name = (
                self.requested_dataset_version
            )

            self.release_path = (
                self.dataset_versions /
                f"{self.dataset_version_name}.json"
            )

        self.release = self.load_json(
            self.release_path,
            "dataset release"
        )

        self.manifest = (
            self.release[
                "manifest_snapshot"
            ]
        )

        self.released_config = (
            self.release[
                "config_snapshot"
            ]
        )

        self.training = (
            self.released_config[
                "training"
            ]
        )

        self.fixed_validation = set(
            self.released_config.get(
                "validation_pages",
                []
            )
        )

        self.train_pages = set(
            self.manifest[
                "training_pages"
            ]
        )

        self.validation_pages = set(
            self.manifest[
                "validation_pages"
            ]
        )

        self.run_name = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + self.dataset_version_name
            + "_"
            + Path(
                self.training["model"]
            ).stem
        )

        self.run_dir = (
            self.runs /
            self.run_name
        )

        self.predictions_dir = (
            self.predictions /
            self.run_name
        )

        return self

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_schema(self):

        actual = int(
            self.release.get(
                "publication_schema",
                0
            )
        )

        if (
            actual
            != self.REQUIRED_PUBLICATION_SCHEMA
        ):

            raise RuntimeError(
                f"Unsupported dataset "
                f"publication schema "
                f"{actual}. "
                f"Expected "
                f"{self.REQUIRED_PUBLICATION_SCHEMA}"
            )

        self.publication_schema = actual

    def validate_version(self):

        version = self.release.get(
            "version_name"
        )

        if (
            version
            != self.dataset_version_name
        ):

            raise RuntimeError(
                "Dataset pointer/version mismatch"
            )

    def validate_page_sets(self):

        missing = (
            self.fixed_validation
            - self.validation_pages
        )

        if missing:

            raise RuntimeError(
                "Fixed validation pages "
                f"missing: {sorted(missing)}"
            )

        overlap = (
            self.train_pages
            & self.validation_pages
        )

        if overlap:

            raise RuntimeError(
                "Train/validation overlap: "
                f"{sorted(overlap)}"
            )

    def validate_release_pages(self):

        release_train = {

            item["page"]

            for item
            in self.release[
                "page_objects"
            ]

            if item["split"]
            == "train"
        }

        release_val = {

            item["page"]

            for item
            in self.release[
                "page_objects"
            ]

            if item["split"]
            == "val"
        }

        if (
            release_train
            != self.train_pages
        ):

            raise RuntimeError(
                "Release train pages do "
                "not match manifest"
            )

        if (
            release_val
            != self.validation_pages
        ):

            raise RuntimeError(
                "Release validation pages "
                "do not match manifest"
            )

    def validate_gpu(self):

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA GPU not available"
            )

        self.gpu_name = (
            torch.cuda.get_device_name(
                0
            )
        )

    # --------------------------------------------------
    # Master validation
    # --------------------------------------------------

    def validate(self):

        self.validate_schema()

        self.validate_version()

        self.validate_page_sets()

        self.validate_release_pages()

        self.validate_gpu()

        return True

    # --------------------------------------------------
    # Summary helper
    # --------------------------------------------------

    def summary(self):

        return {

            "dataset_version":
                self.dataset_version_name,

            "release_hash":
                self.release[
                    "release_hash"
                ],

            "publication_schema":
                self.publication_schema,

            "training_pages":
                len(
                    self.train_pages
                ),

            "validation_pages":
                len(
                    self.validation_pages
                ),

            "training_tiles":
                int(
                    self.release[
                        "training_tile_count"
                    ]
                ),

            "validation_tiles":
                int(
                    self.release[
                        "validation_tile_count"
                    ]
                ),

            "gpu":
                self.gpu_name,

            "run_name":
                self.run_name,
        }
