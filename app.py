import gradio as gr

from modules.dataset_builder import (
    DatasetBuilder
)


def rebuild_dataset():

    builder = DatasetBuilder(
        "project_config.json"
    )

    return builder.build_dataset()


with gr.Blocks() as app:

    gr.Markdown(
        "# Sheet Music AI"
    )

    btn = gr.Button(
        "Build Dataset"
    )

    output = gr.Textbox()

    btn.click(
        rebuild_dataset,
        outputs=output
    )

app.launch()
