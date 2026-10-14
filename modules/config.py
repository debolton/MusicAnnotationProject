# modules/config.py

import json
from pathlib import Path


class ProjectConfig:

    def __init__(self, config_path):

        self.config_path = Path(config_path)

        self.config = json.loads(
            self.config_path.read_text()
        )

        self.root = Path(
            self.config["project_root"]
        )

        self.tile_size = int(
            self.config["tile_size"]
        )

        self.tile_stride = int(
            self.config["tile_stride"]
        )

        self.labels = self.config["labels"]

        self.aligned_dir = (
            self.root / "aligned"
        )

        self.corrected_mask_dir = (
            self.root / "masks" / "corrected"
        )

        self.training_mask_dir = (
            self.root / "masks" / "training"
        )

        self.dataset_dir = (
            self.root / "dataset"
        )

        self.state_dir = (
            self.root / "pipeline_state"
        )

        self.hitl_split_path = (
            self.state_dir /
            "hitl_split_assignments.json"
        )
