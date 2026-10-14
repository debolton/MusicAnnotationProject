# modules/hitl_splitter.py

from datetime import (
    datetime,
    timezone
)

from .utils import (
    read_json,
    write_json_atomic
)


class HitlSplitter:

    def __init__(
        self,
        audit_dir,
        split_path
    ):

        self.audit_dir = audit_dir
        self.split_path = split_path

    def build(self):

        pages = sorted(
            p.stem
            for p in self.audit_dir.glob(
                "Page*.json"
            )
        )

        assignments = read_json(
            self.split_path,
            {
                "version": 1,
                "assignments": {}
            }
        )

        split_map = (
            assignments["assignments"]
        )

        assigned = [
            p for p in pages
            if p in split_map
        ]

        new_pages = [
            p for p in pages
            if p not in split_map
        ]

        idx = len(assigned)

        for page in new_pages:

            idx += 1

            split_map[page] = (
                "val"
                if idx % 5 == 0
                else "train"
            )

        write_json_atomic(
            self.split_path,
            {
                "version": 2,
                "updated_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "assignments":
                    split_map
            }
        )

        return split_map
