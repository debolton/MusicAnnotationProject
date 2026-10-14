# modules/dataset_builder.py

from .config import ProjectConfig
from .hitl_splitter import HitlSplitter
from .validator import DatasetValidator
from .publisher import DatasetPublisher


class DatasetBuilder:

    def __init__(self, config_path):

        self.cfg = ProjectConfig(
            config_path
        )

        self.validator = (
            DatasetValidator(
                self.cfg
            )
        )

        self.publisher = (
            DatasetPublisher(
                self.cfg
            )
        )

    def build_dataset(self):

        splitter = HitlSplitter(
            self.cfg.root /
            "human_in_the_middle" /
            "audit",

            self.cfg.hitl_split_path
        )

        split_map = splitter.build()

        valid_pages = (
            self.validator
            .validate_pages()
        )

        # move:
        # positions()
        # training_mask()
        # tile creation
        # label generation
        # state handling
        # publication

        return (
            f"Processed "
            f"{len(valid_pages)} pages"
        )
