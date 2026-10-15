from fastapi import FastAPI

from modules.alignment.alignment_pipeline import (
    AlignmentPipeline
)

from modules.hitl_generator import (
    HITLDraftGenerator
)

from modules.dataset_builder import (
    DatasetBuilder
)

from modules.trainer import (
    ModelTrainer
)

import json

app = FastAPI()

CONFIG_FILE = "project_config.json"


def load_config():

    with open(CONFIG_FILE) as f:
        return json.load(f)


@app.post("/align")
def align():

    return (
        AlignmentPipeline(
            load_config()
        ).run()
    )


@app.post("/drafts")
def drafts():

    return {

        "result":
        HITLDraftGenerator(
            CONFIG_FILE
        ).generate()

    }


@app.post("/dataset")
def dataset():

    return {

        "result":
        DatasetBuilder(
            CONFIG_FILE
        ).build_dataset()

    }


@app.post("/train")
def train():

    return ModelTrainer(
        CONFIG_FILE
    ).train()
