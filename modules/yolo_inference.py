# modules/yolo_inference.py

import torch

from ultralytics import YOLO


class YOLOPredictor:

    def __init__(self, model_path):

        if not model_path.exists():

            raise RuntimeError(
                f"Model not found: {model_path}"
            )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "GPU required"
            )

        self.model = YOLO(
            str(model_path)
        )

    def predict_tile(
        self,
        tile,
        confidence=0.2,
        image_size=640
    ):

        return self.model.predict(
            tile,
            conf=confidence,
            imgsz=image_size,
            device=0,
            verbose=False,
            retina_masks=True
        )[0]
