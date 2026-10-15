from fastapi import FastAPI
import json

from modules.alignment.alignment_pipeline import (
    AlignmentPipeline
)

from modules.dataset_builder import (
    DatasetBuilder
)

from modules.trainer import (
    ModelTrainer
)

app = FastAPI()

CONFIG_PATH = "project_config.json"


def load_config():

    with open(CONFIG_PATH) as f:
        return json.load(f)


@app.get("/")
def root():

    return {
        "status": "running"
    }


@app.post("/align")
def align():

    result = (
        AlignmentPipeline(
            load_config()
        ).run()
    )

    return result


@app.post("/dataset")
def dataset():

    return {

        "result":
        DatasetBuilder(
            CONFIG_PATH
        ).build_dataset()

    }


@app.post("/train")
def train():

    return ModelTrainer(
        CONFIG_PATH
    ).train()
