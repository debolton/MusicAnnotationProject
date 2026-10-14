# modules/run_exporter.py

import json
import shutil

from pathlib import Path


class RunExporter:

    def __init__(self, release):

        self.release = release

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def image_files(self, folder):

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
        }

        folder = Path(folder)

        if not folder.exists():
            return []

        return sorted(

            p

            for p in folder.iterdir()

            if (
                p.is_file()
                and p.suffix.lower()
                in extensions
            )

        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def build_summary(
        self,
        metric_dict,
        gst_results,
        best_model,
        dataset_path,
        manifest_path,
    ):

        release = self.release.release

        summary = {

            "run_name":
                self.release.run_name,

            "dataset_version":
                self.release.dataset_version_name,

            "publication_schema":
                self.release.publication_schema,

            "dataset_release_hash":
                release["release_hash"],

            "dataset_release_file":
                str(
                    self.release.release_path
                ),

            "model":
                self.release.training[
                    "model"
                ],

            "training_pages":
                sorted(
                    self.release.train_pages
                ),

            "validation_pages":
                sorted(
                    self.release.validation_pages
                ),

            "training_page_count":
                len(
                    self.release.train_pages
                ),

            "validation_page_count":
                len(
                    self.release.validation_pages
                ),

            "training_tile_count":
                len(
                    self.image_files(
                        dataset_path
                        / "images"
                        / "train"
                    )
                ),

            "validation_tile_count":
                len(
                    self.image_files(
                        dataset_path
                        / "images"
                        / "val"
                    )
                ),

            "metrics":
                metric_dict,

            "manifest_md5":
                self.release.md5(
                    manifest_path
                ),

            "best_model":
                str(best_model),

            "gst_normalisation_enabled":
                True,

            "gst_results":
                gst_results.get(
                    "results",
                    []
                ),
        }

        return summary

    # --------------------------------------------------
    # Export Summary JSON
    # --------------------------------------------------

    def export_summary(
        self,
        metric_dict,
        gst_results,
        best_model,
        dataset_path=None,
        manifest_path=None,
    ):

        if dataset_path is None:
            dataset_path = (
                Path("/content")
                / "MusicAnnotationProject_local"
                / "dataset"
            )

        if manifest_path is None:
            manifest_path = (
                Path("/content")
                / "MusicAnnotationProject_local"
                / "dataset_manifest_v3.json"
            )

        summary = self.build_summary(
            metric_dict=metric_dict,
            gst_results=gst_results,
            best_model=best_model,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
        )

        run_summary_path = (
            self.release.run_dir
            / "run_summary.json"
        )

        run_summary_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        run_summary_path.write_text(
            json.dumps(
                summary,
                indent=2
            )
        )

        print(
            "Run summary:",
            run_summary_path
        )

        return summary

    # --------------------------------------------------
    # Copy release metadata
    # --------------------------------------------------

    def copy_release_manifest(
        self,
        local_release_file,
    ):

        target = (
            self.release.run_dir
            / "dataset_release.json"
        )

        shutil.copy2(
            local_release_file,
            target,
        )

        return target

    # --------------------------------------------------
    # Copy outputs from local staging
    # --------------------------------------------------

    def copy_outputs(self):

        drive_runs = (
            self.release.root
            / "training_runs"
        )

        drive_predictions = (
            self.release.root
            / "predictions"
        )

        source_run_dir = (
            self.release.run_dir
        )

        source_prediction_dir = (
            self.release.predictions_dir
        )

        destination_run_dir = (
            drive_runs
            / self.release.run_name
        )

        destination_prediction_dir = (
            drive_predictions
            / self.release.run_name
        )

        if destination_run_dir.exists():

            shutil.rmtree(
                destination_run_dir
            )

        if destination_prediction_dir.exists():

            shutil.rmtree(
                destination_prediction_dir
            )

        shutil.copytree(
            source_run_dir,
            destination_run_dir
        )

        shutil.copytree(
            source_prediction_dir,
            destination_prediction_dir
        )

        print(
            "Training run copied:",
            destination_run_dir
        )

        print(
            "Predictions copied:",
            destination_prediction_dir
        )

        best_model = (
            destination_run_dir
            / "weights"
            / "best.pt"
        )

        if best_model.exists():

            print(
                "Best model:",
                best_model
            )

        return {

            "run_dir":
                destination_run_dir,

            "prediction_dir":
                destination_prediction_dir,

            "best_model":
                best_model,
        }
