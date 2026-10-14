# modules/publisher.py

import tarfile
from pathlib import Path

from .utils import (
    object_hash,
    write_json_atomic
)


class DatasetPublisher:

    def __init__(self, cfg):

        self.cfg = cfg

    def publish_page_tar(
        self,
        page,
        metadata,
        files,
        target
    ):

        target.mkdir(
            parents=True,
            exist_ok=True
        )

        item_hash = (
            object_hash(metadata)
        )

        tar_path = (
            target /
            f"{item_hash}.tar"
        )

        with tarfile.open(
            tar_path,
            "w"
        ) as archive:

            for file_list in files.values():

                for file in file_list:

                    archive.add(
                        file,
                        recursive=False
                    )

        write_json_atomic(
            target /
            f"{item_hash}.json",
            metadata
        )

        return item_hash
