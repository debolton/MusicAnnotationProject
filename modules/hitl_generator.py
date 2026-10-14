# modules/hitl_generator.py

import cv2
import json
import numpy as np

from datetime import (
    datetime,
    timezone
)

from .config import ProjectConfig
from .utils import (
    md5,
    image_files
)

from .audit import (
    AuditManager
)

from .yolo_inference import (
    YOLOPredictor
)

from .confidence_maps import (
    ConfidenceMapGenerator
)


class HITLDraftGenerator:

    def __init__(
        self,
        config_path
    ):

        self.cfg = ProjectConfig(
            config_path
        )

        self.predictor = (
            YOLOPredictor(
                self.cfg.model_path
            )
        )

        self.generator = (
            ConfidenceMapGenerator(
                self.predictor
            )
        )

    def generate(self):

        reviewed = AuditManager(
            self.cfg.review_audit_dir
        ).reviewed_sources()

        model_hash = md5(
            self.cfg.model_path
        )

        generated = 0

        for source in image_files(
            self.cfg.source_dir
        ):

            source_hash = md5(source)

            if source_hash in reviewed:
                continue

            gray = cv2.imread(
                str(source),
                0
            )

            if gray is None:
                continue

            confidence = (
                self.generator.build(
                    gray
                )
            )

            binary = (
                confidence >= 0.20
            ).astype(
                np.uint8
            ) * 255

            confidence_file = (
                self.cfg.confidence_dir /
                f"{source.stem}_confidence.npy"
            )

            mask_file = (
                self.cfg.draft_mask_dir /
                f"{source.stem}_draft.png"
            )

            np.save(
                confidence_file,
                confidence.astype(
                    np.float16
                )
            )

            cv2.imwrite(
                str(mask_file),
                binary
            )

            audit = {

                "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

                "source_md5":
                source_hash,

                "model_md5":
                model_hash,

                "confidence_map":
                str(
                    confidence_file
                ),

                "draft_mask":
                str(
                    mask_file
                )
            }

            (
                self.cfg
                .draft_audit_dir /
                f"{source.stem}.json"
            ).write_text(
                json.dumps(
                    audit,
                    indent=2
                )
            )

            generated += 1

        return (
            f"Generated "
            f"{generated} "
            f"draft masks"
        )
