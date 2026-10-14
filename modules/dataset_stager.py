# modules/dataset_stager.py

import json
import shutil
import tarfile

from pathlib import Path


class DatasetStager:

    def __init__(self, release):

        self.release = release

        self.root = release.root

        self.local_root = Path(
            "/content/MusicAnnotationProject_local"
        )

        self.local_dataset = (
            self.local_root /
            "dataset"
        )

        self.local_runs = (
            self.local_root /
            "training_runs"
        )

        self.local_predictions = (
            self.local_root /
            "predictions"
        )

        self.local_gold = (
            self.local_root /
            "gold_standard_testing"
        )

        self.local_aligned = (
            self.local_root /
            "aligned"
        )

        self.local_yaml = (
            self.local_root /
            "dataset_v4_local.yaml"
        )

        self.local_manifest = (
            self.local_root /
            "dataset_manifest_v3.json"
        )

        self.local_release = (
            self.local_root /
            "dataset_release.json"
        )

    # --------------------------------------------------
    # Generic helpers
    # --------------------------------------------------

    def copy_tree_contents(
        self,
        source,
        destination
    ):

        source = Path(source)
        destination = Path(destination)

        if not source.exists():

            raise RuntimeError(
                f"Required source missing: "
                f"{source}"
            )

        for path in source.rglob("*"):

            if (
                path.is_file()
                and not path.name.endswith(
                    ".cache"
                )
            ):

                target = (
                    destination /
                    path.relative_to(
                        source
                    )
                )

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    path,
                    target,
                )

    def image_files(self, folder):

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        }

        folder = Path(folder)

        if not folder.exists():
            return []

        return sorted(
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in extensions
        )

    # --------------------------------------------------
    # Prepare local structure
    # --------------------------------------------------

    def initialise_local_workspace(self):

        if self.local_root.exists():

            shutil.rmtree(
                self.local_root
            )

        self.local_root.mkdir(
            parents=True
        )

        self.local_runs.mkdir()
        self.local_predictions.mkdir()
        self.local_gold.mkdir()
        self.local_aligned.mkdir()

    # --------------------------------------------------
    # Extract page objects
    # --------------------------------------------------

    def extract_release(self):

        page_objects = (
            self.release.release[
                "page_objects"
            ]
        )

        for index, item in enumerate(
            page_objects,
            start=1
        ):

            object_path = (
                self.root /
                item["relative_path"]
            )

            if not object_path.exists():

                raise RuntimeError(
                    f"Missing page object: "
                    f"{object_path}"
                )

            if (
                object_path.stem
                != item["object_hash"]
            ):

                raise RuntimeError(
                    f"Object name/hash "
                    f"mismatch: "
                    f"{object_path}"
                )

            marker_path = (
                object_path
                .with_suffix(".json")
            )

            if not marker_path.exists():

                raise RuntimeError(
                    f"Missing marker: "
                    f"{marker_path}"
                )

            marker = json.loads(
                marker_path.read_text()
            )

            if (
                int(
                    marker.get(
                        "publication_schema",
                        0
                    )
                )
                != 3
            ):

                raise RuntimeError(
                    f"{item['page']}: "
                    "marker schema mismatch"
                )

            if (
                marker.get(
                    "object_hash"
                )
                != item["object_hash"]
            ):

                raise RuntimeError(
                    f"{item['page']}: "
                    "object hash mismatch"
                )

            if (
                int(
                    marker.get(
                        "tile_count",
                        -1
                    )
                )
                != int(
                    item["tile_count"]
                )
            ):

                raise RuntimeError(
                    f"{item['page']}: "
                    "tile count mismatch"
                )

            with tarfile.open(
                object_path,
                "r"
            ) as archive:

                members = [

                    m

                    for m
                    in archive.getmembers()

                    if m.isfile()

                ]

                expected = (
                    int(
                        item["tile_count"]
                    ) * 3
                )

                if (
                    len(members)
                    != expected
                ):

                    raise RuntimeError(
                        f"{item['page']}: "
                        "unexpected TAR "
                        "member count"
                    )

                for member in members:

                    member_path = Path(
                        member.name
                    )

                    if (
                        member_path
                        .is_absolute()
                    ):

                        raise RuntimeError(
                            "Unsafe tar path"
                        )

                    if (
                        ".."
                        in member_path.parts
                    ):

                        raise RuntimeError(
                            "Unsafe tar path"
                        )

                archive.extractall(
                    self.local_root
                )

            if (
                index % 25 == 0
                or index
                == len(page_objects)
            ):

                print(
                    "Extracted pages:",
                    index,
                    "/",
                    len(page_objects)
                )

    # --------------------------------------------------
    # Local YAML
    # --------------------------------------------------

    def write_yaml(self):

        yaml_text = (
            f"path: {self.local_dataset}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: annotation\n"
        )

        self.local_yaml.write_text(
            yaml_text
        )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def page_prefix(self, stem):

        parts = stem.split("_y")

        if len(parts) != 2:
            return None

        return parts[0]

    def validate_split(
        self,
        split,
        expected_pages,
        expected_tiles
    ):

        image_dir = (
            self.local_dataset /
            "images" /
            split
        )

        label_dir = (
            self.local_dataset /
            "labels" /
            split
        )

        mask_dir = (
            self.local_dataset /
            "mask_tiles" /
            split
        )

        image_map = {
            p.stem: p
            for p in self.image_files(
                image_dir
            )
        }

        label_map = {
            p.stem: p
            for p in label_dir.glob(
                "*.txt"
            )
        }

        mask_map = {
            p.stem: p
            for p in self.image_files(
                mask_dir
            )
        }

        if (
            set(image_map)
            != set(label_map)
            or
            set(image_map)
            != set(mask_map)
        ):

            raise RuntimeError(
                f"{split}: "
                "image/label/mask "
                "mismatch"
            )

        if (
            len(image_map)
            != expected_tiles
        ):

            raise RuntimeError(
                f"{split}: "
                f"expected "
                f"{expected_tiles} "
                f"tiles, found "
                f"{len(image_map)}"
            )

        prefixes = {

            self.page_prefix(stem)

            for stem in image_map

        }

        if prefixes != expected_pages:

            raise RuntimeError(
                f"{split}: "
                "manifest/page "
                "mismatch"
            )

        print(
            f"{split}:",
            len(image_map),
            "tiles across",
            len(prefixes),
            "pages"
        )

        return prefixes

    # --------------------------------------------------
    # Main stage operation
    # --------------------------------------------------

    def stage(self):

        self.initialise_local_workspace()

        self.extract_release()

        self.copy_tree_contents(
            self.release.gold,
            self.local_gold
        )

        self.copy_tree_contents(
            self.release.aligned,
            self.local_aligned
        )

        self.local_manifest.write_text(
            json.dumps(
                self.release.manifest,
                indent=2
            )
        )

        self.local_release.write_text(
            json.dumps(
                self.release.release,
                indent=2
            )
        )

        self.write_yaml()

        train_prefixes = (
            self.validate_split(
                "train",
                self.release.train_pages,
                int(
                    self.release.release[
                        "training_tile_count"
                    ]
                )
            )
        )

        val_prefixes = (
            self.validate_split(
                "val",
                self.release.validation_pages,
                int(
                    self.release.release[
                        "validation_tile_count"
                    ]
                )
            )
        )

        if (
            train_prefixes
            & val_prefixes
        ):

            raise RuntimeError(
                "Page prefixes "
                "appear in both "
                "splits"
            )

        return {
            "root":
                self.local_root,
            "dataset":
                self.local_dataset,
            "yaml":
                self.local_yaml,
            "gold":
                self.local_gold,
            "aligned":
                self.local_aligned,
            "manifest":
                self.local_manifest,
            "release":
                self.local_release,
        }
