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

        # additional paths

        self.hitl_dir = (
            self.root /
            "human_in_the_middle"
        )

        self.draft_mask_dir = (
            self.hitl_dir /
            "draft_masks"
        )

        self.confidence_dir = (
            self.hitl_dir /
            "confidence_maps"
        )

        self.draft_audit_dir = (
            self.hitl_dir /
            "draft_audit"
        )

        self.review_audit_dir = (
            self.hitl_dir /
            "audit"
        )

        self.source_dir = (
            self.root /
            "source" /
            "annotated_only"
        )

        self.model_path = (
            self.root /
            "training_runs" /
            "latest" /
            "weights" /
            "best.pt"
        )
