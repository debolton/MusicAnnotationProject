# modules/trainer.py

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

    # --------------------------------------------------
    # Main training workflow
    # --------------------------------------------------

    def train(self):

        #
        # Load and validate selected dataset release
        #

        self.release.load()

        self.release.validate()

        print("Dataset release validated")

        #
        # Build local training dataset
        #

        stager = DatasetStager(
            self.release
        )

        staging = stager.stage()

        yaml_file = staging["yaml"]

        dataset_path = staging["dataset"]

        print(
            "Dataset staged:",
            dataset_path
        )

        #
        # Train model
        #

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
                self.release.runs
            ),
            name=self.release.run_name,
            exist_ok=False,
            seed=training["seed"],
            plots=True,
            cache="disk",
        )

        #
        # Locate best model
        #

        best_model = (
            self.release.run_dir
            / "weights"
            / "best.pt"
        )

        if not best_model.exists():

            raise RuntimeError(
                "best.pt missing after training"
            )

        print(
            "Best model:",
            best_model
        )

        trained = YOLO(
            str(best_model)
        )

        #
        # Validation metrics
        #

        metrics = trained.val(
            data=str(yaml_file),
            split="val",
            imgsz=training["image_size"],
            device=0,
            project=str(
                self.release.predictions
            ),
            name=self.release.run_name,
            exist_ok=True,
        )

        metric_dict = {

            key: float(value)

            for key, value
            in metrics.results_dict.items()

        }

        print(
            "Validation complete"
        )

        #
        # Predict validation tiles
        #

        trained.predict(
            source=str(
                dataset_path
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
                self.release.predictions
            ),
            name=self.release.run_name
            + "_validation_tiles",
            exist_ok=True,
            retina_masks=True,
        )

        print(
            "Validation tile predictions complete"
        )

        #
        # Full-page validation predictions
        #

        for page in sorted(
            self.release.validation_pages
        ):

            hits = [

                p

                for p in (
                    self.release.aligned
                ).glob("*")

                if p.stem.lower()
                == page.lower()

            ]

            if not hits:

                print(
                    "Missing validation page:",
                    page
                )

                continue

            trained.predict(
                source=str(hits[0]),
                save=True,
                conf=training[
                    "prediction_confidence"
                ],
                imgsz=1280,
                device=0,
                project=str(
                    self.release.predictions
                ),
                name=self.release.run_name
                + "_validation_pages",
                exist_ok=True,
                retina_masks=True,
            )

        print(
            "Validation page predictions complete"
        )

        #
        # GST evaluation
        #

        gst = GSTEvaluator(
            model=trained,
            release=self.release,
            dataset=staging,
        )

        gst_results = gst.run()

        print(
            "GST comparison complete"
        )

        #
        # Export summaries
        #

        exporter = RunExporter(
            self.release
        )

        summary = exporter.export_summary(
            metric_dict=metric_dict,
            gst_results=gst_results,
            best_model=str(best_model),
            dataset_path=staging["dataset"],
            manifest_path=staging["manifest"],
        )

        exporter.copy_release_manifest(
            staging["release"]
        )

        exporter.copy_outputs()

        #
        # Final response object
        #

        return {

            "run_name":
                self.release.run_name,

            "dataset_version":
                self.release.dataset_version_name,

            "best_model":
                str(best_model),

            "summary":
                summary,

            "metrics":
                metric_dict,
        }

    # --------------------------------------------------
    # Optional helper methods
    # --------------------------------------------------

    def validate_only(self):

        self.release.load()

        self.release.validate()

        return self.release.summary()

    def stage_only(self):

        self.release.load()

        self.release.validate()

        stager = DatasetStager(
            self.release
        )

        return stager.stage()
