# app.py

import json
import traceback

import gradio as gr

from modules.hitl_generator import HITLDraftGenerator
from modules.dataset_builder import DatasetBuilder
from modules.trainer import ModelTrainer
from modules.pilot.pilot_reviewer import PilotReviewer


CONFIG_PATH = "project_config.json"


# ==================================================
# Helpers
# ==================================================

def safe_call(fn):

    try:

        return fn()

    except Exception as e:

        traceback.print_exc()

        return (
            "ERROR\n\n"
            f"{type(e).__name__}\n\n"
            f"{str(e)}"
        )


# ==================================================
# Draft generation
# ==================================================

def generate_drafts():

    def _run():

        generator = (
            HITLDraftGenerator(
                CONFIG_PATH
            )
        )

        return generator.generate()

    return safe_call(_run)


# ==================================================
# Dataset build
# ==================================================

def build_dataset():

    def _run():

        builder = DatasetBuilder(
            CONFIG_PATH
        )

        return builder.build_dataset()

    return safe_call(_run)


# ==================================================
# Validation
# ==================================================

def validate_release():

    def _run():

        trainer = ModelTrainer(
            CONFIG_PATH
        )

        summary = (
            trainer.validate_only()
        )

        return json.dumps(
            summary,
            indent=2
        )

    return safe_call(_run)


# ==================================================
# Training
# ==================================================

def train_model():

    def _run():

        trainer = ModelTrainer(
            CONFIG_PATH
        )

        result = trainer.train()

        return json.dumps(
            {
                "run_name":
                    result["run_name"],

                "dataset_version":
                    result[
                        "dataset_version"
                    ],

                "best_model":
                    result[
                        "best_model"
                    ],

                "metrics":
                    result[
                        "metrics"
                    ],
            },
            indent=2,
        )

    return safe_call(_run)


# ==================================================
# Main app
# ==================================================

with gr.Blocks(
    title="Music Annotation Project",
    fill_width=True,
) as app:

    gr.Markdown(
        """
# 🎼 Music Annotation Project

End-to-end workflow:

1. Generate YOLO Draft Masks
2. Review Drafts
3. Build Dataset Release
4. Validate Dataset Release
5. Train New Model
6. Run Pilot Reviewer
"""
    )

    # ------------------------------------------------
    # Draft Generation
    # ------------------------------------------------

    with gr.Tab("Generate Drafts"):

        gr.Markdown(
            """
Generate:

- Draft masks
- Confidence maps
- Draft audits

for all unreviewed pages.
"""
        )

        draft_btn = gr.Button(
            "Generate Drafts",
            variant="primary"
        )

        draft_output = gr.Textbox(
            lines=12,
            label="Output"
        )

        draft_btn.click(
            generate_drafts,
            outputs=draft_output,
        )

    # ------------------------------------------------
    # Dataset Builder
    # ------------------------------------------------

    with gr.Tab("Build Dataset"):

        gr.Markdown(
            """
Create immutable dataset release.

Builds:

- page objects
- manifests
- dataset versions
"""
        )

        build_btn = gr.Button(
            "Build Dataset",
            variant="primary"
        )

        build_output = gr.Textbox(
            lines=15,
            label="Output"
        )

        build_btn.click(
            build_dataset,
            outputs=build_output,
        )

    # ------------------------------------------------
    # Validate
    # ------------------------------------------------

    with gr.Tab("Validate Release"):

        validate_btn = gr.Button(
            "Run Safety Gate"
        )

        validate_output = gr.Code(
            language="json"
        )

        validate_btn.click(
            validate_release,
            outputs=validate_output,
        )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    with gr.Tab("Training"):

        gr.Markdown(
            """
Workflow:

- Load release
- Stage dataset
- Train YOLO
- Validation
- GST comparison
- Export results
"""
        )

        train_btn = gr.Button(
            "Train Model",
            variant="primary"
        )

        train_output = gr.Code(
            language="json"
        )

        train_btn.click(
            train_model,
            outputs=train_output,
        )

    # ------------------------------------------------
    # Pilot Reviewer
    # ------------------------------------------------

    with gr.Tab("Pilot Reviewer"):

        gr.Markdown(
            """
Loads the frozen V7.7F policy,
runs the regression gate,
builds proposals and launches
the supervised reviewer.
"""
        )

        try:

            reviewer = (
                PilotReviewer(
                    CONFIG_PATH
                )
            )

            reviewer.build()

        except Exception as e:

            gr.Error(
                f"Pilot reviewer "
                f"failed to load: {e}"
            )

    # ------------------------------------------------
    # System Status
    # ------------------------------------------------

    with gr.Tab("System Status"):

        gr.Markdown(
            """
### Refactored Modules

✅ Draft Generator

✅ Dataset Builder

✅ Dataset Releases

✅ Dataset Staging

✅ Training

✅ GST Evaluation

✅ Run Export

✅ Pilot Reviewer

✅ Frozen Policy Bundle

✅ Regression Gate

✅ Decision Preservation
"""
        )


# ==================================================
# Launch
# ==================================================

if __name__ == "__main__":

    app.launch()
