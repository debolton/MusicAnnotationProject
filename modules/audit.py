# modules/audit.py

import json


class AuditManager:

    def __init__(self, review_dir):

        self.review_dir = review_dir

    def reviewed_sources(self):

        results = set()

        for path in self.review_dir.glob(
            "Page*.json"
        ):

            try:

                audit = json.loads(
                    path.read_text()
                )

                md5 = audit.get(
                    "source_md5"
                )

                if md5:
                    results.add(md5)

            except Exception:
                pass

        return results
