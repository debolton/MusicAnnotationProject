# app.py

import json
from pathlib import Path

import gradio as gr

from modules.hitl_generator import HITLDraftGenerator
from modules.dataset_builder import DatasetBuilder
from modules.trainer import ModelTrainer


CONFIG_FILE = "project_config.json"


# =====================================================
# Draft Generation
# =====================================================

def generate_drafts():

    try:

        generator = HITLDraftGenerator(
            CONFIG_FILE
        )

        result = generator.generate()

        return result

    except Exception as ex:

        return f"ERROR\n\n{ex}"


# =====================================================
# Dataset Builder
# =====================================================

def build_dataset():

    try:

        builder = DatasetBuilder(
            CONFIG_FILE
        )

        result = builder.build_dataset()

        return result

    except Exception as ex:

        return f"ERROR\n\n{ex}"


# =====================================================
# Training
# =====================================================

def train_model():

    try:

        trainer = ModelTrainer(
            CONFIG_FILE
        )

        result = trainer.train()

        metrics = result["metrics"]

        return f"""
Training Complete

Run:
{result["run_name"]}

Dataset:
{result["dataset_version"]}

Best Model:
{result["best_model"]}

Metrics:
{json.dumps(metrics, indent=2)}
"""

    except Exception as ex:

        return f"ERROR\n\n{ex}"


# =====================================================
# Dataset Validation
# =====================================================

def validate_dataset():

    try:

        trainer = ModelTrainer(
            CONFIG_FILE
        )

        summary = (
            trainer.validate_only()
        )

        return json.dumps(
            summary,
            indent=2
        )

    except Exception as ex:

        return f"ERROR\n\n{ex}"


# =====================================================
# Gradio UI
# =====================================================

with gr.Blocks(
    title="Music Annotation Project"
) as app:

    gr.Markdown(
        """
# 🎼 Music Annotation Project

End-to-end workflow:

1. Generate Draft Masks
2. Review Drafts
3. Build Dataset Release
4. Train Model
"""
    )

    # -------------------------------------------------
    # Generate Drafts
    # -------------------------------------------------

    with gr.Tab(
        "Generate Drafts"
    ):

        gr.Markdown(
            """
Run YOLO on unreviewed source
pages and generate:

- draft masks
- confidence maps
- draft audits
"""
        )

        draft_btn = gr.Button(
            "Generate Draft Masks",
            variant="primary"
        )

        draft_status = gr.Textbox(
            label="Status",
            lines=10
        )

        draft_btn.click(
            fn=generate_drafts,
            outputs=draft_status
        )

    # -------------------------------------------------
    # Review Drafts
    # -------------------------------------------------

    with gr.Tab(
        "Review Drafts"
    ):

        gr.Markdown(
            """
The canvas reviewer currently
runs from review_app.py.

Once migrated:

- load page
- add annotations
- erase false positives
- save reviewed mask
"""
        )

        gr.Info(
            "Reviewer integration pending."
        )

    # -------------------------------------------------
    # Build Dataset
    # -------------------------------------------------

    with gr.Tab(
        "Build Dataset"
    ):

        gr.Markdown(
            """
Builds the immutable
schema-3 dataset release.
"""
        )

        build_btn = gr.Button(
            "Build Dataset",
            variant="primary"
        )

        build_status = gr.Textbox(
            label="Build Log",
            lines=15
        )

        build_btn.click(
            fn=build_dataset,
            outputs=build_status
        )

    # -------------------------------------------------
    # Validate Release
    # -------------------------------------------------

    with gr.Tab(
        "Validate Release"
    ):

        validate_btn = gr.Button(
            "Run Safety Gate"
        )

        validation_output = gr.Code(
            language="json"
        )

        validate_btn.click(
            fn=validate_dataset,
            outputs=validation_output
        )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    with gr.Tab(
        "Train Model"
    ):

        gr.Markdown(
            """
Stages dataset release,
trains YOLO,
validates,
runs GST comparison,
and exports outputs.
"""
        )

        train_btn = gr.Button(
            "Train Model",
            variant="primary"
        )

        training_output = gr.Textbox(
            label="Training Status",
            lines=20
        )

        train_btn.click(
            fn=train_model,
            outputs=training_output
        )

    # -------------------------------------------------
    # About
    # -------------------------------------------------

    with gr.Tab(
        "Project Status"
    ):

        status_text = """
Current architecture

✅ HITL Draft Generation

✅ Dataset Builder

✅ Dataset Releases

✅ Trainer

✅ GST Evaluator

✅ Run Exporter

🚧 Canvas Reviewer Integration
"""

        gr.Markdown(status_text)


# =====================================================
# Launch
# =====================================================

if __name__ == "__main__":

    app.launch()
