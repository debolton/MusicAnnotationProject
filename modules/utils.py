# modules/utils.py

import json
import hashlib
from pathlib import Path


SUPPORTED = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}


def image_files(folder):

    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
    )


def object_hash(value):

    return hashlib.md5(
        json.dumps(
            value,
            sort_keys=True
        ).encode()
    ).hexdigest()


def md5(path):

    h = hashlib.md5()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1048576),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path, default):

    try:

        if Path(path).exists():
            return json.loads(
                Path(path).read_text()
            )

    except Exception:
        pass

    return default


def write_json_atomic(path, value):

    path = Path(path)

    temp = Path(
        str(path) + ".tmp"
    )

    temp.write_text(
        json.dumps(
            value,
            indent=2
        )
    )

    temp.replace(path)
