# app.py
import json
import traceback
from pathlib import Path
import gradio as gr
from modules.alignment.alignment_pipeline import AlignmentPipeline
from modules.hitl_generator import HITLDraftGenerator
from modules.dataset_builder import DatasetBuilder
from modules.trainer import ModelTrainer

CONFIG_PATH = 'project_config.json'

def load_config():
    p = Path(CONFIG_PATH)
    if not p.exists():
        raise RuntimeError(f'Missing configuration file: {p}')
    return json.loads(p.read_text())


def safe_run(fn):
    try:
        return fn()
    except Exception:
        return traceback.format_exc()


def run_alignment():
    return safe_run(lambda: json.dumps(AlignmentPipeline(load_config()).run(), indent=2, default=str))


def generate_drafts():
    return safe_run(lambda: str(HITLDraftGenerator(CONFIG_PATH).generate()))


def build_dataset():
    return safe_run(lambda: str(DatasetBuilder(CONFIG_PATH).build_dataset()))


def validate_release():
    return safe_run(lambda: json.dumps(ModelTrainer(CONFIG_PATH).validate_only(), indent=2, default=str))


def train_model():
    return safe_run(lambda: json.dumps(ModelTrainer(CONFIG_PATH).train(), indent=2, default=str))

with gr.Blocks(title='Music Annotation Project') as app:
    gr.Markdown('# ð¼ Music Annotation Project

End-to-end workflow for alignment, draft generation, dataset publication and YOLO training.')

    with gr.Tab('1ï¸â£ Align Pages'):
        btn = gr.Button('Run Alignment', variant='primary')
        out = gr.Code(label='Alignment Results')
        btn.click(run_alignment, outputs=out)

    with gr.Tab('2ï¸â£ Generate Drafts'):
        btn = gr.Button('Generate Draft Masks', variant='primary')
        out = gr.Textbox(lines=15)
        btn.click(generate_drafts, outputs=out)

    with gr.Tab('3ï¸â£ Build Dataset'):
        btn = gr.Button('Build Dataset', variant='primary')
        out = gr.Textbox(lines=20)
        btn.click(build_dataset, outputs=out)

    with gr.Tab('4ï¸â£ Validate Release'):
        btn = gr.Button('Validate Release')
        out = gr.Code()
        btn.click(validate_release, outputs=out)

    with gr.Tab('5ï¸â£ Train Model'):
        btn = gr.Button('Train Model', variant='primary')
        out = gr.Code()
        btn.click(train_model, outputs=out)

    with gr.Tab('â¹ï¸ Project Status'):
        gr.Markdown('Pilot reviewer runs separately via pilot_app.py')

if __name__ == '__main__':
    app.launch()
