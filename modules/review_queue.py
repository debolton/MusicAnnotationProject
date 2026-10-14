from pathlib import Path
import json
import hashlib
import re

SUPPORTED = {
    ".jpg", ".jpeg", ".png",
    ".tif", ".tiff", ".bmp"
}


class ReviewQueue:

    def __init__(
        self,
        source_dir,
        draft_mask_dir,
        audit_dir,
        aligned_dir,
        corrected_dir
    ):
        self.source_dir = source_dir
        self.draft_mask_dir = draft_mask_dir
        self.audit_dir = audit_dir
        self.aligned_dir = aligned_dir
        self.corrected_dir = corrected_dir

    def files(self, folder):

        return sorted(
            p for p in Path(folder).iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED
        )

    def md5(self, path):

        h = hashlib.md5()

        with open(path, "rb") as f:
            for block in iter(
                lambda: f.read(1048576),
                b""
            ):
                h.update(block)

        return h.hexdigest()

    def reviewed_hashes(self):

        result = set()

        for path in self.audit_dir.glob(
            "Page*.json"
        ):

            try:

                value = json.loads(
                    path.read_text()
                ).get("source_md5")

                if value:
                    result.add(value)

            except Exception:
                pass

        return result

    def next_page_number(self):

        values = []

        for folder in [
            self.aligned_dir,
            self.corrected_dir
        ]:

            for path in self.files(folder):

                match = re.fullmatch(
                    r"Page(\d+)",
                    path.stem,
                    re.I
                )

                if match:
                    values.append(
                        int(match.group(1))
                    )

        return max(
            values,
            default=0
        ) + 1

    def build_queue(self):

        done = self.reviewed_hashes()

        return [

            p

            for p in self.files(
                self.source_dir
            )

            if (
                self.draft_mask_dir /
                f"{p.stem}_draft.png"
            ).exists()

            and self.md5(p) not in done
        ]
