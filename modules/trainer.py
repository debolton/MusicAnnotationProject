from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

from modules.dataset_release import DatasetRelease
from modules.dataset_stager import DatasetStager
from modules.gst_evaluator import GSTEvaluator
from modules.run_exporter import RunExporter


class ModelTrainer:

    def __init__(
        self,
        config_path,
        requested_dataset_version=None,
    ):

        self.release = DatasetRelease(
            config_path=config_path,
            requested_dataset_version=requested_dataset_version,
        )

    def train(self):

        self.release.load()

        self.release.validate()

        stager = DatasetStager(
            self.release
        )

        staging = stager.stage()

        yaml_file = staging["yaml"]

        training = self.release.training

        model = YOLO(
            training["model"]
        )

        model.train(
            data=str(yaml_file),
            epochs=training["epochs"],
            patience=training["patience"],
            imgsz=training["image_size"],
            batch=training["batch_size"],
            device=0,
            workers=2,
            project=str(
                self.release.runs_dir
            ),
            name=self.release.run_name,
            exist_ok=False,
            seed=training["seed"],
            plots=True,
            cache="disk",
        )

        best_model = (
            self.release.run_dir
            / "weights"
            / "best.pt"
        )

        if not best_model.exists():

            raise RuntimeError(
                "best.pt missing after training"
            )

        trained = YOLO(
            str(best_model)
        )

        metrics = trained.val(
            data=str(yaml_file),
            split="val",
            imgsz=training["image_size"],
            device=0,
            project=str(
                self.release.predictions_dir
            ),
            name="metrics",
            exist_ok=True,
        )

        metric_dict = {

            key: float(value)

            for key, value
            in metrics.results_dict.items()
        }

        trained.predict(
            source=str(
                staging["dataset"]
                / "images"
                / "val"
            ),
            save=True,
            conf=training[
                "prediction_confidence"
            ],
            imgsz=training[
                "image_size"
            ],
            device=0,
            project=str(
                self.release.predictions_dir
            ),
            name="validation_tiles",
            exist_ok=True,
            retina_masks=True,
        )

        gst = GSTEvaluator(
            model=trained,
            release=self.release,
            dataset=staging,
        )

        gst_results = gst.run()

        exporter = RunExporter(
            self.release
        )

        exporter.export_summary(
            metric_dict,
            gst_results,
            str(best_model),
        )

        exporter.copy_outputs()

        return {

            "run_name":
                self.release.run_name,

            "best_model":
                str(best_model),

            "metrics":
                metric_dict,
        }

    @staticmethod
    def ensure_gpu():

        if not torch.cuda.is_available():

            raise RuntimeError(
                "GPU is required"
            )

        return torch.cuda.get_device_name(
            0
        )
