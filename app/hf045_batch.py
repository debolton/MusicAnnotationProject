from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import stat
import zipfile

import cv2
import numpy as np


SUPPORTED = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

MAX_UPLOAD_BYTES = 1_000_000_000
MAX_FILES = 5000
MAX_UNCOMPRESSED_BYTES = 4_000_000_000


class BatchFailure(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def _safe_members(
    archive: zipfile.ZipFile,
):
    members = archive.infolist()

    if len(members) > MAX_FILES:
        raise BatchFailure(
            "ZIP contains too many entries: "
            f"{len(members)}"
        )

    total_uncompressed = sum(
        int(item.file_size)
        for item in members
    )

    if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
        raise BatchFailure(
            "ZIP uncompressed size exceeds the batch limit"
        )

    seen_paths = set()

    for item in members:
        raw_name = item.filename

        if not raw_name:
            raise BatchFailure(
                "ZIP contains an unnamed entry"
            )

        if "\x00" in raw_name:
            raise BatchFailure(
                "ZIP member contains a null character"
            )

        if "\\" in raw_name:
            raise BatchFailure(
                "ZIP member uses backslashes: "
                f"{raw_name}"
            )

        path = PurePosixPath(raw_name)

        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
        ):
            raise BatchFailure(
                "Unsafe ZIP member path: "
                f"{raw_name}"
            )

        first_part = path.parts[0]

        if (
            len(first_part) >= 2
            and first_part[1] == ":"
        ):
            raise BatchFailure(
                "Windows drive-style ZIP member path "
                f"is prohibited: {raw_name}"
            )

        normalised_path = str(path)

        if normalised_path in seen_paths:
            raise BatchFailure(
                "Duplicate ZIP member path: "
                f"{normalised_path}"
            )

        seen_paths.add(normalised_path)

        unix_mode = (
            item.external_attr >> 16
        )

        if stat.S_ISLNK(unix_mode):
            raise BatchFailure(
                "ZIP symbolic link is prohibited: "
                f"{raw_name}"
            )

        if item.flag_bits & 0x1:
            raise BatchFailure(
                "Encrypted ZIP member is prohibited: "
                f"{raw_name}"
            )

        if item.file_size < 0:
            raise BatchFailure(
                "ZIP member has an invalid size: "
                f"{raw_name}"
            )

        if item.is_dir():
            continue

        yield item


def _role(
    path: PurePosixPath,
):
    parts = [
        part.lower()
        for part in path.parts
    ]

    stem = path.stem.lower()

    if path.suffix.lower() not in SUPPORTED:
        return None

    mask_directories = {
        "masks",
        "mask",
        "reviewed_masks",
        "corrected_masks",
        "authoritative_masks",
    }

    mask_suffixes = (
        "_reviewed_authoritative_seed_mask",
        "_authoritative_mask",
        "_reviewed_mask",
        "_corrected_mask",
        "_mask",
    )

    if any(
        part in mask_directories
        for part in parts[:-1]
    ):
        return "mask"

    if stem.endswith(mask_suffixes):
        return "mask"

    observed_directories = {
        "observed",
        "annotated",
        "annotated_pages",
        "images",
        "pages",
        "input",
    }

    if any(
        part in observed_directories
        for part in parts[:-1]
    ):
        return "observed"

    return "observed"


def _pair_key(
    path: PurePosixPath,
    role: str,
) -> str:
    stem = path.stem

    if role == "mask":
        for suffix in (
            "_reviewed_authoritative_seed_mask",
            "_authoritative_mask",
            "_reviewed_mask",
            "_corrected_mask",
            "_mask",
        ):
            if stem.lower().endswith(suffix):
                stem = stem[:-len(suffix)]
                break

    stem = stem.strip()

    if not stem:
        raise BatchFailure(
            "Could not derive a page key from "
            f"{path}"
        )

    if stem in {".", ".."}:
        raise BatchFailure(
            "Unsafe page key derived from "
            f"{path}"
        )

    if any(
        character in stem
        for character in (
            "/",
            "\\",
            "\x00",
        )
    ):
        raise BatchFailure(
            "Unsafe page key derived from "
            f"{path}"
        )

    return stem


def _decode_gray(
    data: bytes,
    name: str,
) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(
            data,
            np.uint8,
        ),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise BatchFailure(
            f"Unreadable image: {name}"
        )

    if image.ndim != 2:
        raise BatchFailure(
            f"Image is not greyscale-compatible: {name}"
        )

    return np.ascontiguousarray(
        image.astype(np.uint8)
    )


def _write_png(
    path: Path,
    array,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    value = np.asarray(array)

    if value.dtype == bool:
        value = (
            value.astype(np.uint8)
            * 255
        )
    elif value.dtype != np.uint8:
        if not np.issubdtype(
            value.dtype,
            np.number,
        ):
            raise BatchFailure(
                "Could not write non-numeric PNG array: "
                f"{path.name}"
            )

        if not np.isfinite(value).all():
            raise BatchFailure(
                "Could not write non-finite PNG array: "
                f"{path.name}"
            )

        value = np.clip(
            value,
            0,
            255,
        ).astype(np.uint8)

    value = np.ascontiguousarray(value)

    if not cv2.imwrite(
        str(path),
        value,
    ):
        raise BatchFailure(
            f"Could not write PNG: {path.name}"
        )


def _require_result_fields(
    result: dict,
    required: set[str],
    page: str,
) -> None:
    if not isinstance(result, dict):
        raise BatchFailure(
            f"{page}: operator result is not a dictionary"
        )

    missing = sorted(
        required - set(result)
    )

    if missing:
        raise BatchFailure(
            f"{page}: operator result omitted required "
            f"fields: {missing}"
        )


def _require_shape(
    value,
    name: str,
    shape: tuple[int, int],
    page: str,
) -> np.ndarray:
    array = np.asarray(value)

    if array.shape != shape:
        raise BatchFailure(
            f"{page}: {name} shape mismatch: "
            f"expected {shape}, found {array.shape}"
        )

    return array


def _require_uint8_page(
    value,
    name: str,
    shape: tuple[int, int],
    page: str,
) -> np.ndarray:
    array = _require_shape(
        value,
        name,
        shape,
        page,
    )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise BatchFailure(
            f"{page}: {name} is not numeric"
        )

    if not np.isfinite(array).all():
        raise BatchFailure(
            f"{page}: {name} contains non-finite values"
        )

    if array.size and (
        array.min() < 0
        or array.max() > 255
    ):
        raise BatchFailure(
            f"{page}: {name} contains values outside 0..255"
        )

    return np.ascontiguousarray(
        array.astype(np.uint8)
    )


def _require_bool_page(
    value,
    name: str,
    shape: tuple[int, int],
    page: str,
) -> np.ndarray:
    array = _require_shape(
        value,
        name,
        shape,
        page,
    )

    if (
        np.issubdtype(array.dtype, np.number)
        and not np.isfinite(array).all()
    ):
        raise BatchFailure(
            f"{page}: {name} contains non-finite values"
        )

    return np.ascontiguousarray(
        array.astype(bool)
    )


def _normalise_seed(
    seed: np.ndarray,
) -> np.ndarray:
    return np.where(
        np.asarray(seed) > 0,
        255,
        0,
    ).astype(np.uint8)


def run_batch(
    upload: bytes,
    operator,
    output_root: Path,
) -> Path:
    if not upload:
        raise BatchFailure(
            "Uploaded ZIP is empty"
        )

    if len(upload) > MAX_UPLOAD_BYTES:
        raise BatchFailure(
            "Uploaded ZIP exceeds the batch limit"
        )

    if operator is None:
        raise BatchFailure(
            "A5 operator is not initialised"
        )

    if not callable(
        getattr(operator, "run", None)
    ):
        raise BatchFailure(
            "A5 operator does not provide run()"
        )

    output_root = Path(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = (
        datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + "_hf045"
    )

    run = (
        output_root
        / run_id
    )

    pages_dir = (
        run
        / "pages"
    )

    run.mkdir(
        parents=True,
        exist_ok=False,
    )

    pages_dir.mkdir()

    try:
        try:
            archive = zipfile.ZipFile(
                io.BytesIO(upload),
                "r",
            )
        except zipfile.BadZipFile as exc:
            raise BatchFailure(
                "Uploaded file is not a valid ZIP archive"
            ) from exc

        with archive:
            records = []

            for item in _safe_members(
                archive
            ):
                path = PurePosixPath(
                    item.filename
                )

                role = _role(path)

                if role is None:
                    continue

                try:
                    data = archive.read(item)
                except Exception as exc:
                    raise BatchFailure(
                        "Could not read ZIP member: "
                        f"{item.filename}"
                    ) from exc

                if len(data) != item.file_size:
                    raise BatchFailure(
                        "ZIP member size mismatch: "
                        f"{item.filename}"
                    )

                records.append(
                    {
                        "path": str(path),
                        "role": role,
                        "key": _pair_key(
                            path,
                            role,
                        ),
                        "data": data,
                        "sha256": _sha256_bytes(
                            data
                        ),
                    }
                )

        if not records:
            raise BatchFailure(
                "ZIP contains no supported image files"
            )

        observed_records = {}
        mask_records = {}

        for record in records:
            target = (
                mask_records
                if record["role"] == "mask"
                else observed_records
            )

            key = record["key"]

            if key in target:
                raise BatchFailure(
                    f"Duplicate {record['role']} key "
                    f"'{key}': "
                    f"{target[key]['path']} and "
                    f"{record['path']}"
                )

            target[key] = record

        keys = sorted(
            set(observed_records)
            | set(mask_records)
        )

        missing_observed = [
            key
            for key in keys
            if key not in observed_records
        ]

        missing_masks = [
            key
            for key in keys
            if key not in mask_records
        ]

        if (
            missing_observed
            or missing_masks
        ):
            raise BatchFailure(
                "Observed/mask pairing is incomplete. "
                f"Missing observed={missing_observed[:20]}, "
                f"missing masks={missing_masks[:20]}"
            )

        summary_rows = []

        for key in keys:
            observed_record = (
                observed_records[key]
            )

            mask_record = (
                mask_records[key]
            )

            observed = _decode_gray(
                observed_record["data"],
                observed_record["path"],
            )

            seed = _decode_gray(
                mask_record["data"],
                mask_record["path"],
            )

            if observed.shape != seed.shape:
                raise BatchFailure(
                    f"{key}: observed/mask shape mismatch "
                    f"{observed.shape} versus {seed.shape}"
                )

            seed = _normalise_seed(seed)

            try:
                result = operator.run(
                    observed=observed,
                    reviewed_seed=seed,
                    page=key,
                    batch=512,
                )
            except Exception as exc:
                raise BatchFailure(
                    f"{key}: operator processing failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            required_fields = {
                "mask",
                "telea",
                "paper",
                "routes",
                "geometry_t80",
                "a1",
                "a5",
                "eligible_a5_v1",
                "p_print",
                "provisional",
                "final",
                "print_mask",
            }

            _require_result_fields(
                result,
                required_fields,
                key,
            )

            shape = observed.shape

            authority = _require_bool_page(
                result["mask"],
                "authoritative mask",
                shape,
                key,
            )

            telea = _require_uint8_page(
                result["telea"],
                "Telea diagnostic",
                shape,
                key,
            )

            paper = _require_uint8_page(
                result["paper"],
                "paper estimate",
                shape,
                key,
            )

            geometry_t80 = _require_uint8_page(
                result["geometry_t80"],
                "frozen V7.8 geometry_t80",
                shape,
                key,
            )

            a5 = _require_uint8_page(
                result["a5"],
                "frozen A5-v1 page",
                shape,
                key,
            )

            a1 = _require_uint8_page(
                result["a1"],
                "Notebook 87 A1 page",
                shape,
                key,
            )

            eligible = _require_bool_page(
                result["eligible_a5_v1"],
                "A5-v1 eligibility",
                shape,
                key,
            )

            print_mask = _require_bool_page(
                result["print_mask"],
                "A5-v1 PRINT mask",
                shape,
                key,
            )

            p_print = _require_shape(
                result["p_print"],
                "A5-v1 PRINT probability",
                shape,
                key,
            )

            provisional = _require_shape(
                result["provisional"],
                "A5-v1 provisional decisions",
                shape,
                key,
            )

            final = _require_shape(
                result["final"],
                "A5-v1 final decisions",
                shape,
                key,
            )

            routes = result["routes"]

            if not isinstance(routes, dict):
                raise BatchFailure(
                    f"{key}: V7.8 routes are not a dictionary"
                )

            missing_routes = sorted(
                {
                    "preserve",
                    "remove",
                    "ambiguous",
                }
                - set(routes)
            )

            if missing_routes:
                raise BatchFailure(
                    f"{key}: V7.8 routes omit "
                    f"{missing_routes}"
                )

            preserve = _require_bool_page(
                routes["preserve"],
                "V7.8 PRESERVE route",
                shape,
                key,
            )

            remove = _require_bool_page(
                routes["remove"],
                "V7.8 REMOVE route",
                shape,
                key,
            )

            ambiguous = _require_bool_page(
                routes["ambiguous"],
                "V7.8 AMBIGUOUS route",
                shape,
                key,
            )

            if np.any(
                preserve & remove
            ):
                raise BatchFailure(
                    f"{key}: PRESERVE and REMOVE overlap"
                )

            if np.any(
                preserve & ambiguous
            ):
                raise BatchFailure(
                    f"{key}: PRESERVE and AMBIGUOUS overlap"
                )

            if np.any(
                remove & ambiguous
            ):
                raise BatchFailure(
                    f"{key}: REMOVE and AMBIGUOUS overlap"
                )

            routed = (
                preserve
                | remove
                | ambiguous
            )

            if not np.array_equal(
                routed,
                authority,
            ):
                missing_route = int(
                    np.sum(
                        authority
                        & ~routed
                    )
                )

                escaped_route = int(
                    np.sum(
                        routed
                        & ~authority
                    )
                )

                raise BatchFailure(
                    f"{key}: V7.8 routes do not exactly "
                    "partition the authoritative mask: "
                    f"missing={missing_route}, "
                    f"outside={escaped_route}"
                )

            if np.any(
                geometry_t80[~authority]
                != observed[~authority]
            ):
                raise BatchFailure(
                    f"{key}: V7.8 has outside-mask drift"
                )

            if np.any(
                print_mask
                & ~eligible
            ):
                raise BatchFailure(
                    f"{key}: PRINT exists outside "
                    "A5-v1 eligibility"
                )

            if np.any(
                print_mask
                & ~authority
            ):
                raise BatchFailure(
                    f"{key}: PRINT exists outside "
                    "the authoritative mask"
                )

            if np.any(
                a5[~print_mask]
                != geometry_t80[~print_mask]
            ):
                raise BatchFailure(
                    f"{key}: A5 differs from V7.8 "
                    "outside PRINT"
                )

            if np.any(
                a5[print_mask]
                != observed[print_mask]
            ):
                raise BatchFailure(
                    f"{key}: A5 PRINT does not equal "
                    "the exact observed byte"
                )

            if np.any(
                a5[~authority]
                != observed[~authority]
            ):
                raise BatchFailure(
                    f"{key}: A5 changed pixels outside "
                    "the authoritative mask"
                )

            a5_changed = (
                a5
                != geometry_t80
            )

            expected_a5_changed = (
                print_mask
                & (
                    observed
                    != geometry_t80
                )
            )

            if not np.array_equal(
                a5_changed,
                expected_a5_changed,
            ):
                unexpected = int(
                    np.sum(
                        a5_changed
                        & ~expected_a5_changed
                    )
                )

                missing = int(
                    np.sum(
                        expected_a5_changed
                        & ~a5_changed
                    )
                )

                raise BatchFailure(
                    f"{key}: A5 delta does not equal "
                    "effective PRINT restoration: "
                    f"unexpected={unexpected}, "
                    f"missing={missing}"
                )

            if np.any(
                a5_changed
                & (
                    a5
                    != observed
                )
            ):
                raise BatchFailure(
                    f"{key}: an A5 changed byte does not "
                    "equal observed"
                )

            if np.any(
                a1[ambiguous]
                != observed[ambiguous]
            ):
                raise BatchFailure(
                    f"{key}: A1 AMBIGUOUS bytes do not "
                    "equal observed"
                )

            if np.any(
                a1[~ambiguous]
                != a5[~ambiguous]
            ):
                raise BatchFailure(
                    f"{key}: A1 differs from A5 outside "
                    "the AMBIGUOUS route"
                )

            if np.any(
                a1[~authority]
                != observed[~authority]
            ):
                raise BatchFailure(
                    f"{key}: A1 changed pixels outside "
                    "the authoritative mask"
                )

            a1_changed_from_a5 = (
                a1
                != a5
            )

            expected_a1_changed_from_a5 = (
                ambiguous
                & (
                    observed
                    != a5
                )
            )

            if not np.array_equal(
                a1_changed_from_a5,
                expected_a1_changed_from_a5,
            ):
                unexpected = int(
                    np.sum(
                        a1_changed_from_a5
                        & ~expected_a1_changed_from_a5
                    )
                )

                missing = int(
                    np.sum(
                        expected_a1_changed_from_a5
                        & ~a1_changed_from_a5
                    )
                )

                raise BatchFailure(
                    f"{key}: A1 delta from A5 does not equal "
                    "effective AMBIGUOUS-to-observed "
                    "restoration: "
                    f"unexpected={unexpected}, "
                    f"missing={missing}"
                )

            a1_changed_from_v78 = (
                a1
                != geometry_t80
            )

            out = (
                pages_dir
                / key
            )

            out.mkdir(
                parents=False,
                exist_ok=False,
            )

            _write_png(
                out / "observed.png",
                observed,
            )

            _write_png(
                out / "reviewed_authoritative_seed_mask.png",
                seed,
            )

            _write_png(
                out / "prepared_dilation2_mask.png",
                authority,
            )

            _write_png(
                out / "telea_diagnostic.png",
                telea,
            )

            _write_png(
                out / "paper_estimate.png",
                paper,
            )

            _write_png(
                out / "route_preserve.png",
                preserve,
            )

            _write_png(
                out / "route_remove.png",
                remove,
            )

            _write_png(
                out / "route_ambiguous.png",
                ambiguous,
            )

            _write_png(
                out / "v7_8_geometry_t80.png",
                geometry_t80,
            )

            _write_png(
                out / "a5_v1_corrected.png",
                a5,
            )

            _write_png(
                out / "a1_notebook87.png",
                a1,
            )

            _write_png(
                out / "a5_eligible.png",
                eligible,
            )

            _write_png(
                out / "a5_print_mask.png",
                print_mask,
            )

            _write_png(
                out / "a5_delta_from_v7_8.png",
                a5_changed,
            )

            _write_png(
                out / "a1_delta_from_a5.png",
                a1_changed_from_a5,
            )

            _write_png(
                out / "a1_delta_from_v7_8.png",
                a1_changed_from_v78,
            )

            np.save(
                out / "a5_print_probability.npy",
                p_print,
            )

            np.save(
                out / "a5_provisional_decision.npy",
                provisional,
                allow_pickle=True,
            )

            np.save(
                out / "a5_final_decision.npy",
                final,
                allow_pickle=True,
            )

            provisional_print = int(
                np.sum(
                    provisional
                    == "PRINT"
                )
            )

            final_print = int(
                np.sum(
                    final
                    == "PRINT"
                )
            )

            run_demotions = int(
                np.sum(
                    (
                        provisional
                        == "PRINT"
                    )
                    & (
                        final
                        != "PRINT"
                    )
                )
            )

            audit = {
                "page": key,
                "observed_source": observed_record["path"],
                "mask_source": mask_record["path"],
                "observed_sha256": observed_record["sha256"],
                "mask_sha256": mask_record["sha256"],
                "shape": list(shape),

                "prepared_mask_pixels": int(
                    authority.sum()
                ),
                "preserve_pixels": int(
                    preserve.sum()
                ),
                "remove_pixels": int(
                    remove.sum()
                ),
                "ambiguous_pixels": int(
                    ambiguous.sum()
                ),

                "a5_eligible_pixels": int(
                    eligible.sum()
                ),
                "a5_provisional_print_pixels": (
                    provisional_print
                ),
                "a5_final_print_pixels": (
                    final_print
                ),
                "a5_print_mask_pixels": int(
                    print_mask.sum()
                ),
                "a5_run_demotions": (
                    run_demotions
                ),
                "a5_changed_from_v7_8_pixels": int(
                    a5_changed.sum()
                ),

                "a1_changed_from_a5_pixels": int(
                    a1_changed_from_a5.sum()
                ),
                "a1_changed_from_v7_8_pixels": int(
                    a1_changed_from_v78.sum()
                ),

                "v7_8_outside_mask_drift": int(
                    np.sum(
                        geometry_t80[~authority]
                        != observed[~authority]
                    )
                ),
                "a5_outside_mask_drift": int(
                    np.sum(
                        a5[~authority]
                        != observed[~authority]
                    )
                ),
                "a5_non_print_v7_8_regression": int(
                    np.sum(
                        a5[~print_mask]
                        != geometry_t80[~print_mask]
                    )
                ),
                "a5_print_not_equal_observed": int(
                    np.sum(
                        a5[print_mask]
                        != observed[print_mask]
                    )
                ),
                "a5_changed_byte_not_equal_observed": int(
                    np.sum(
                        a5_changed
                        & (
                            a5
                            != observed
                        )
                    )
                ),

                "a1_ambiguous_not_equal_observed": int(
                    np.sum(
                        a1[ambiguous]
                        != observed[ambiguous]
                    )
                ),
                "a1_non_ambiguous_regression_from_a5": int(
                    np.sum(
                        a1[~ambiguous]
                        != a5[~ambiguous]
                    )
                ),
                "a1_outside_mask_drift": int(
                    np.sum(
                        a1[~authority]
                        != observed[~authority]
                    )
                ),

                "a5_policy_id": result.get(
                    "a5_policy_id"
                ),
                "a5_bundle_id": result.get(
                    "bundle_id"
                ),
                "a5_model_sha256": result.get(
                    "model_sha256"
                ),
                "a5_calibrator_sha256": result.get(
                    "calibrator_sha256"
                ),
                "a1_assembly": result.get(
                    "a1_assembly",
                    (
                        "A5_V1_CORRECTED_THEN_V78_"
                        "AMBIGUOUS_RESTORED_TO_"
                        "EXACT_OBSERVED"
                    ),
                ),
            }

            zero_fields = [
                "v7_8_outside_mask_drift",
                "a5_outside_mask_drift",
                "a5_non_print_v7_8_regression",
                "a5_print_not_equal_observed",
                "a5_changed_byte_not_equal_observed",
                "a1_ambiguous_not_equal_observed",
                "a1_non_ambiguous_regression_from_a5",
                "a1_outside_mask_drift",
            ]

            failures = {
                field: audit[field]
                for field in zero_fields
                if audit[field] != 0
            }

            if failures:
                raise BatchFailure(
                    f"{key}: operational integrity "
                    f"contract failed: {failures}"
                )

            (
                out
                / "page_audit.json"
            ).write_text(
                json.dumps(
                    audit,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            summary_rows.append(
                audit
            )

        if not summary_rows:
            raise BatchFailure(
                "No pages were processed"
            )

        summary_path = (
            run
            / "batch_summary.csv"
        )

        fieldnames = list(
            summary_rows[0].keys()
        )

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="raise",
            )

            writer.writeheader()
            writer.writerows(
                summary_rows
            )

        manifest = {
            "version": "HF-04.5",
            "created_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "pages": len(summary_rows),

            "policy_id": getattr(
                operator,
                "a5_policy_id",
                None,
            ),
            "bundle_id": getattr(
                operator,
                "bundle_id",
                None,
            ),
            "model_sha256": getattr(
                operator,
                "model_sha256",
                None,
            ),
            "calibrator_sha256": getattr(
                operator,
                "calibrator_sha256",
                None,
            ),

            "class_contract": {
                "model_output_0": "ANNOTATION",
                "model_output_1": "PRINT",
                "truth_code_2": "UNCERTAIN",
                "probability": "p_print",
            },

            "implementation": (
                "A5Operator.run once per page; "
                "frozen V7.8 geometry_t80; "
                "frozen A5-v1 observed-PRINT; "
                "Notebook 87 A1 assembly"
            ),

            "page_lineage": {
                "v7_8_geometry_t80": (
                    "PRESERVE copies observed; "
                    "REMOVE uses paper estimate; "
                    "AMBIGUOUS uses Telea"
                ),
                "a5_v1": (
                    "Frozen V7.8 except final PRINT "
                    "coordinates, which copy exact "
                    "observed bytes"
                ),
                "a1_notebook87": (
                    "A5-v1 with every frozen V7.8 "
                    "AMBIGUOUS coordinate restored to "
                    "the exact observed byte"
                ),
            },

            "input_zip_sha256": _sha256_bytes(
                upload
            ),
            "input_zip_bytes": len(upload),
            "summary_sha256": _sha256_file(
                summary_path
            ),

            "limits": {
                "maximum_upload_bytes": (
                    MAX_UPLOAD_BYTES
                ),
                "maximum_zip_entries": (
                    MAX_FILES
                ),
                "maximum_uncompressed_bytes": (
                    MAX_UNCOMPRESSED_BYTES
                ),
            },

            "fail_closed": True,
            "partial_zip_permitted": False,
        }

        manifest_path = (
            run
            / "batch_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        zip_path = (
            output_root
            / f"{run_id}.zip"
        )

        temporary_zip = (
            output_root
            / f".{run_id}.zip.tmp"
        )

        if temporary_zip.exists():
            temporary_zip.unlink()

        try:
            with zipfile.ZipFile(
                temporary_zip,
                "w",
                zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as output_archive:
                for path in sorted(
                    run.rglob("*")
                ):
                    if not path.is_file():
                        continue

                    output_archive.write(
                        path,
                        path.relative_to(run),
                    )

            if not temporary_zip.is_file():
                raise BatchFailure(
                    "Batch ZIP creation did not produce "
                    "an output file"
                )

            if temporary_zip.stat().st_size == 0:
                raise BatchFailure(
                    "Batch ZIP creation produced an empty file"
                )

            temporary_zip.replace(
                zip_path
            )

        except Exception:
            if temporary_zip.exists():
                temporary_zip.unlink()

            raise

        if not zip_path.is_file():
            raise BatchFailure(
                "Batch completed without creating "
                "the final output ZIP"
            )

        return zip_path

    except BatchFailure:
        # Keep the run directory for forensic examination.
        # No partial ZIP is returned.
        raise

    except Exception as exc:
        # Keep the run directory for forensic examination.
        # Convert unexpected failures into the batch exception type.
        raise BatchFailure(
            "Unexpected HF-04.5 batch failure: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
