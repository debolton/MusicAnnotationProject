import gradio as gr

from modules.dataset_builder import (
    DatasetBuilder
)


def rebuild_dataset():

    builder = DatasetBuilder(
        "project_config.json"
    )

    return builder.build_dataset()

from modules.hitl_generator import (
    HITLDraftGenerator
)

def generate_hitl():

    generator = (
        HITLDraftGenerator(
            "project_config.json"
        )
    )

    return generator.generate()



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

with gr.Tab("Generate Drafts"):

    generate_btn = gr.Button(
        "Generate Draft Masks"
    )

    output = gr.Textbox()

    generate_btn.click(
        generate_hitl,
        outputs=output
    )


app.launch()
