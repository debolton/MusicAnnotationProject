from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import shutil
import stat
import zipfile

import cv2
import numpy as np

from .final_evidence_upload import (
    EVIDENCE_SCHEMA_VERSION, commit_staged_object, file_record,
    get_s3_client, training_sample_id, write_json,
)


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
HISTORICAL_DATA_ROLE = "TRAINING_ELIGIBLE"
HISTORICAL_INGESTION_METHOD = "HISTORICAL_YOLO_CORPUS_IMPORT"


class BatchFailure(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(
        np.asarray(value)
    )

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())

    return digest.hexdigest()


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

    seen_paths: set[str] = set()

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

        unix_mode = item.external_attr >> 16

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
) -> str | None:
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

    if value.ndim != 2:
        raise BatchFailure(
            "Could not write non-two-dimensional PNG array: "
            f"{path.name}, shape={value.shape}"
        )

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


def _json_safe(value):
    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        return value.item()

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serialisable"
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
        np.issubdtype(
            array.dtype,
            np.number,
        )
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
    return np.ascontiguousarray(
        np.where(
            np.asarray(seed) > 0,
            255,
            0,
        ).astype(np.uint8)
    )


def _normalise_decision_map(
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

    if np.issubdtype(
        array.dtype,
        np.number,
    ):
        if not np.isfinite(array).all():
            raise BatchFailure(
                f"{page}: {name} contains non-finite values"
            )

        mapping = {
            0: "ANNOTATION",
            1: "PRINT",
            2: "UNCERTAIN",
        }

        integer = array.astype(np.int64)

        unknown = sorted(
            set(integer.ravel().tolist())
            - set(mapping)
        )

        if unknown:
            raise BatchFailure(
                f"{page}: {name} contains unknown "
                f"numeric classes: {unknown}"
            )

        result = np.asarray(
            [
                mapping[int(item)]
                for item in integer.ravel()
            ],
            dtype=object,
        ).reshape(shape)

    else:
        result = np.char.upper(
            array.astype(str)
        ).astype(object)

    allowed = {
        "PRINT",
        "ANNOTATION",
        "UNCERTAIN",
        "INELIGIBLE",
    }

    unexpected = sorted(
        set(result.ravel().tolist())
        - allowed
    )

    if unexpected:
        raise BatchFailure(
            f"{page}: {name} contains unexpected "
            f"classes: {unexpected}"
        )

    return result


def _decision_visual(
    provisional: np.ndarray,
    final: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    visual = np.zeros(
        eligible.shape,
        dtype=np.uint8,
    )

    visual[
        eligible
        & (
            final == "ANNOTATION"
        )
    ] = 85

    visual[
        eligible
        & (
            final == "UNCERTAIN"
        )
    ] = 170

    visual[
        eligible
        & (
            final == "PRINT"
        )
    ] = 255

    demoted = (
        eligible
        & (
            provisional == "PRINT"
        )
        & (
            final != "PRINT"
        )
    )

    visual[demoted] = 210

    return visual


def _write_audit_json(
    path: Path,
    value: dict,
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_safe,
        ),
        encoding="utf-8",
    )



def _commit_historical_detector_pair(*, run, page, observed, seed,
                                     observed_record, mask_record):
    """Store a canonical historic YOLO training pair, content-addressably."""
    get_s3_client()
    root = Path(run) / "_detector_training_staging" / page
    if root.exists():
        shutil.rmtree(root)
    target = root / "detector_training"
    target.mkdir(parents=True, exist_ok=False)
    annotated = target / "annotated.png"
    mask = target / "authoritative_mask.png"
    _write_png(annotated, observed)
    _write_png(mask, seed)
    sample_id = training_sample_id(annotated, mask)
    metadata = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_role": "DETECTOR_TRAINING_CANDIDATE",
        "data_role": HISTORICAL_DATA_ROLE,
        "ingestion_method": HISTORICAL_INGESTION_METHOD,
        "sample_id": sample_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_page_key": page,
        "source_observed_member": observed_record["path"],
        "source_mask_member": mask_record["path"],
        "source_observed_sha256": observed_record["sha256"],
        "source_mask_sha256": mask_record["sha256"],
        "annotated_image": file_record(annotated),
        "authoritative_mask": {**file_record(mask),
            "semantics": "HUMAN_REVIEWED_UNDILATED_ANNOTATION_SEED_MASK",
            "dilation": 0},
        "previously_used_for_yolo_training": True,
        "incumbent_training_exposure": "KNOWN_EXPOSED",
        "raw_yolo_prediction_is_truth": False,
        "training_eligibility": {"detector_training": True,
            "a5_pixel_training": False, "independent_holdout": False,
            "regression_only": False},
    }
    write_json(target / "metadata.json", metadata)
    try:
        receipt = commit_staged_object(staged_directory=target,
            final_prefix="detector_training", object_id=sample_id)
        return {"stored": True, "sample_id": sample_id,
                "data_role": HISTORICAL_DATA_ROLE, **receipt}
    finally:
        shutil.rmtree(root, ignore_errors=True)

def run_batch(
    upload: bytes,
    a1_operator,
    operator,
    output_root: Path,
) -> Path:
    """
    Run the operational correction lineage:

        A1Operator.run()
            -> frozen V7.8
            -> frozen V8.9
            -> Notebook 87 A1

        A5Operator.run_from_a1()
            -> A5 with Notebook 87 A1 as the exact
               non-PRINT fallback

    Packaging is fail-closed. A partial ZIP is never returned.
    The uncompressed run directory is retained if processing fails.
    """

    if not upload:
        raise BatchFailure(
            "Uploaded ZIP is empty"
        )

    if len(upload) > MAX_UPLOAD_BYTES:
        raise BatchFailure(
            "Uploaded ZIP exceeds the batch limit"
        )

    if a1_operator is None:
        raise BatchFailure(
            "A1 operator is not initialised"
        )

    if not callable(
        getattr(
            a1_operator,
            "run",
            None,
        )
    ):
        raise BatchFailure(
            "A1 operator does not provide run()"
        )

    if operator is None:
        raise BatchFailure(
            "A5 operator is not initialised"
        )

    if not callable(
        getattr(
            operator,
            "run_from_a1",
            None,
        )
    ):
        raise BatchFailure(
            "A5 operator does not provide run_from_a1()"
        )

    output_root = Path(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    get_s3_client()

    run_id = (
        datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + "_hf045"
    )

    run = output_root / run_id
    pages_dir = run / "pages"

    run.mkdir(
        parents=True,
        exist_ok=False,
    )

    pages_dir.mkdir()

    input_zip_sha256 = _sha256_bytes(
        upload
    )

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

        observed_records: dict[str, dict] = {}
        mask_records: dict[str, dict] = {}

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

        summary_rows: list[dict] = []
        detector_training_receipts: list[dict] = []

        for key in keys:
            observed_record = observed_records[key]
            mask_record = mask_records[key]

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

            detector_receipt = _commit_historical_detector_pair(
                run=run, page=key, observed=observed, seed=seed,
                observed_record=observed_record, mask_record=mask_record,
            )
            detector_training_receipts.append({"page": key, **detector_receipt})

            try:
                a1_result = a1_operator.run(
                    observed=observed,
                    reviewed_seed=seed,
                    page=key,
                )

                result = operator.run_from_a1(
                    observed=observed,
                    reviewed_seed=seed,
                    a1_result=a1_result,
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
                "normalised",
                "routes",
                "geometry_t80",
                "v8_9_corrected",
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

            normalised = _require_uint8_page(
                result["normalised"],
                "normalised observed page",
                shape,
                key,
            )

            geometry_t80 = _require_uint8_page(
                result["geometry_t80"],
                "frozen V7.8 geometry_t80",
                shape,
                key,
            )

            v8_9_corrected = _require_uint8_page(
                result["v8_9_corrected"],
                "frozen V8.9 corrected page",
                shape,
                key,
            )

            a1 = _require_uint8_page(
                result["a1"],
                "Notebook 87 A1 page",
                shape,
                key,
            )

            a5 = _require_uint8_page(
                result["a5"],
                "frozen A5-v1 page",
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

            if not np.issubdtype(
                p_print.dtype,
                np.number,
            ):
                raise BatchFailure(
                    f"{key}: p_print is not numeric"
                )

            if np.any(
                ~np.isfinite(
                    p_print[eligible]
                )
            ):
                raise BatchFailure(
                    f"{key}: eligible p_print values "
                    "contain non-finite values"
                )

            if (
                np.any(
                    p_print[eligible] < 0.0
                )
                or np.any(
                    p_print[eligible] > 1.0
                )
            ):
                raise BatchFailure(
                    f"{key}: eligible p_print values "
                    "lie outside 0..1"
                )

            provisional = _normalise_decision_map(
                result["provisional"],
                "A5-v1 provisional decisions",
                shape,
                key,
            )

            final = _normalise_decision_map(
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

            expected_eligible = (
                authority
                & (
                    observed < 200
                )
                & ~preserve
            )

            if not np.array_equal(
                eligible,
                expected_eligible,
            ):
                missing_eligible = int(
                    np.sum(
                        expected_eligible
                        & ~eligible
                    )
                )

                unexpected_eligible = int(
                    np.sum(
                        eligible
                        & ~expected_eligible
                    )
                )

                raise BatchFailure(
                    f"{key}: A5 eligibility differs from "
                    "the frozen candidate rule: "
                    f"missing={missing_eligible}, "
                    f"unexpected={unexpected_eligible}"
                )

            expected_print_mask = (
                eligible
                & (
                    final == "PRINT"
                )
            )

            if not np.array_equal(
                print_mask,
                expected_print_mask,
            ):
                missing_print = int(
                    np.sum(
                        expected_print_mask
                        & ~print_mask
                    )
                )

                unexpected_print = int(
                    np.sum(
                        print_mask
                        & ~expected_print_mask
                    )
                )

                raise BatchFailure(
                    f"{key}: PRINT mask differs from final "
                    "PRINT decisions: "
                    f"missing={missing_print}, "
                    f"unexpected={unexpected_print}"
                )

            if np.any(
                provisional[~eligible]
                != "INELIGIBLE"
            ):
                raise BatchFailure(
                    f"{key}: provisional decisions exist "
                    "outside A5 eligibility"
                )

            if np.any(
                final[~eligible]
                != "INELIGIBLE"
            ):
                raise BatchFailure(
                    f"{key}: final decisions exist "
                    "outside A5 eligibility"
                )

            if np.any(
                final[eligible]
                == "INELIGIBLE"
            ):
                raise BatchFailure(
                    f"{key}: eligible coordinates contain "
                    "INELIGIBLE final decisions"
                )

            if np.any(
                (
                    final == "PRINT"
                )
                & (
                    provisional != "PRINT"
                )
            ):
                raise BatchFailure(
                    f"{key}: run filtering created a new "
                    "PRINT decision"
                )

            if np.any(
                (
                    provisional == "ANNOTATION"
                )
                & (
                    final != "ANNOTATION"
                )
            ):
                raise BatchFailure(
                    f"{key}: run filtering altered an "
                    "ANNOTATION decision"
                )

            # --------------------------------------------------
            # Frozen V7.8 and V8.9 stage integrity
            # --------------------------------------------------

            if np.any(
                geometry_t80[~authority]
                != observed[~authority]
            ):
                raise BatchFailure(
                    f"{key}: V7.8 has outside-mask drift"
                )

            if np.any(
                telea[~authority]
                != observed[~authority]
            ):
                raise BatchFailure(
                    f"{key}: Telea has outside-mask drift"
                )

            if np.any(
                v8_9_corrected[~authority]
                != observed[~authority]
            ):
                raise BatchFailure(
                    f"{key}: V8.9 has outside-mask drift"
                )

            # --------------------------------------------------
            # Notebook 87 A1 contract
            # --------------------------------------------------

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
                != v8_9_corrected[~ambiguous]
            ):
                raise BatchFailure(
                    f"{key}: A1 differs from V8.9 outside "
                    "the V7.8 AMBIGUOUS route"
                )

            if np.any(
                a1[~authority]
                != observed[~authority]
            ):
                raise BatchFailure(
                    f"{key}: A1 changed pixels outside "
                    "the authoritative mask"
                )

            a1_changed_from_v89 = (
                a1 != v8_9_corrected
            )

            expected_a1_changed_from_v89 = (
                ambiguous
                & (
                    observed
                    != v8_9_corrected
                )
            )

            if not np.array_equal(
                a1_changed_from_v89,
                expected_a1_changed_from_v89,
            ):
                unexpected = int(
                    np.sum(
                        a1_changed_from_v89
                        & ~expected_a1_changed_from_v89
                    )
                )

                missing = int(
                    np.sum(
                        expected_a1_changed_from_v89
                        & ~a1_changed_from_v89
                    )
                )

                raise BatchFailure(
                    f"{key}: A1 delta from V8.9 does not "
                    "equal effective AMBIGUOUS-to-observed "
                    "restoration: "
                    f"unexpected={unexpected}, "
                    f"missing={missing}"
                )

            if np.any(
                a1_changed_from_v89
                & (
                    a1 != observed
                )
            ):
                raise BatchFailure(
                    f"{key}: an A1 changed byte does not "
                    "equal observed"
                )

            a1_changed_from_v78 = (
                a1 != geometry_t80
            )

            # --------------------------------------------------
            # A5-from-A1 contract
            # --------------------------------------------------

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
                != a1[~print_mask]
            ):
                raise BatchFailure(
                    f"{key}: A5 differs from Notebook 87 A1 "
                    "outside final PRINT"
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

            a5_changed_from_a1 = (
                a5 != a1
            )

            expected_a5_changed_from_a1 = (
                print_mask
                & (
                    observed != a1
                )
            )

            if not np.array_equal(
                a5_changed_from_a1,
                expected_a5_changed_from_a1,
            ):
                unexpected = int(
                    np.sum(
                        a5_changed_from_a1
                        & ~expected_a5_changed_from_a1
                    )
                )

                missing = int(
                    np.sum(
                        expected_a5_changed_from_a1
                        & ~a5_changed_from_a1
                    )
                )

                raise BatchFailure(
                    f"{key}: A5 delta from A1 does not equal "
                    "effective PRINT restoration: "
                    f"unexpected={unexpected}, "
                    f"missing={missing}"
                )

            if np.any(
                a5_changed_from_a1
                & (
                    a5 != observed
                )
            ):
                raise BatchFailure(
                    f"{key}: an A5 changed byte does not "
                    "equal observed"
                )

            a5_changed_from_v78 = (
                a5 != geometry_t80
            )

            provisional_print = int(
                np.sum(
                    provisional == "PRINT"
                )
            )

            final_print = int(
                np.sum(
                    final == "PRINT"
                )
            )

            run_demotions = int(
                np.sum(
                    (
                        provisional == "PRINT"
                    )
                    & (
                        final != "PRINT"
                    )
                )
            )

            effective_print_changes = int(
                a5_changed_from_a1.sum()
            )

            ineffective_print_decisions = int(
                np.sum(
                    print_mask
                    & (
                        observed == a1
                    )
                )
            )

            out = pages_dir / key

            out.mkdir(
                parents=False,
                exist_ok=False,
            )

            # --------------------------------------------------
            # Input and authoritative-mask artefacts
            # --------------------------------------------------

            _write_png(
                out / "observed.png",
                observed,
            )

            _write_png(
                out
                / "reviewed_authoritative_seed_mask.png",
                seed,
            )

            _write_png(
                out / "prepared_dilation2_mask.png",
                authority,
            )

            # --------------------------------------------------
            # Upstream reconstruction evidence
            # --------------------------------------------------

            _write_png(
                out / "telea_diagnostic.png",
                telea,
            )

            _write_png(
                out / "paper_estimate.png",
                paper,
            )

            _write_png(
                out / "normalised_observed.png",
                normalised,
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

            # --------------------------------------------------
            # Correction lineage
            # --------------------------------------------------

            _write_png(
                out / "v7_8_corrected.png",
                geometry_t80,
            )

            _write_png(
                out / "v8_9_corrected.png",
                v8_9_corrected,
            )

            _write_png(
                out / "a1_corrected.png",
                a1,
            )

            _write_png(
                out / "a5_corrected.png",
                a5,
            )

            # Compatibility filenames for existing consumers.

            _write_png(
                out / "v7_8_geometry_t80.png",
                geometry_t80,
            )

            _write_png(
                out / "a1_notebook87.png",
                a1,
            )

            _write_png(
                out / "a5_v1_corrected.png",
                a5,
            )

            # --------------------------------------------------
            # A1 and A5 masks/deltas
            # --------------------------------------------------

            _write_png(
                out / "a1_delta_from_v8_9.png",
                a1_changed_from_v89,
            )

            _write_png(
                out / "a1_delta_from_v7_8.png",
                a1_changed_from_v78,
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
                out / "a5_delta_from_a1.png",
                a5_changed_from_a1,
            )

            _write_png(
                out / "a5_delta_from_v7_8.png",
                a5_changed_from_v78,
            )

            _write_png(
                out / "a5_decision_map.png",
                _decision_visual(
                    provisional=provisional,
                    final=final,
                    eligible=eligible,
                ),
            )

            # --------------------------------------------------
            # Numeric decision artefacts
            # --------------------------------------------------

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

            # --------------------------------------------------
            # Page audit
            # --------------------------------------------------

            audit = {
                "page": key,
                "data_role": HISTORICAL_DATA_ROLE,
                "detector_training_sample_id": detector_receipt["sample_id"],
                "detector_training_already_existed": bool(
                    detector_receipt.get("already_existed", False)),
                "detector_training_commit_object": detector_receipt.get(
                    "commit_object"),
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

                "a1_changed_from_v8_9_pixels": int(
                    a1_changed_from_v89.sum()
                ),
                "a1_changed_from_v7_8_pixels": int(
                    a1_changed_from_v78.sum()
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
                "a5_run_demotions": run_demotions,
                "a5_effective_changed_from_a1_pixels": (
                    effective_print_changes
                ),
                "a5_ineffective_print_decisions": (
                    ineffective_print_decisions
                ),
                "a5_changed_from_a1_pixels": int(
                    a5_changed_from_a1.sum()
                ),
                "a5_changed_from_v7_8_pixels": int(
                    a5_changed_from_v78.sum()
                ),

                "v7_8_outside_mask_drift": int(
                    np.sum(
                        geometry_t80[~authority]
                        != observed[~authority]
                    )
                ),
                "telea_outside_mask_drift": int(
                    np.sum(
                        telea[~authority]
                        != observed[~authority]
                    )
                ),
                "v8_9_outside_mask_drift": int(
                    np.sum(
                        v8_9_corrected[~authority]
                        != observed[~authority]
                    )
                ),

                "a1_ambiguous_not_equal_observed": int(
                    np.sum(
                        a1[ambiguous]
                        != observed[ambiguous]
                    )
                ),
                "a1_non_ambiguous_v89_regression": int(
                    np.sum(
                        a1[~ambiguous]
                        != v8_9_corrected[~ambiguous]
                    )
                ),
                "a1_outside_mask_drift": int(
                    np.sum(
                        a1[~authority]
                        != observed[~authority]
                    )
                ),
                "a1_delta_from_v89_unexpected": int(
                    np.sum(
                        a1_changed_from_v89
                        & ~expected_a1_changed_from_v89
                    )
                ),
                "a1_delta_from_v89_missing": int(
                    np.sum(
                        expected_a1_changed_from_v89
                        & ~a1_changed_from_v89
                    )
                ),
                "a1_changed_byte_not_equal_observed": int(
                    np.sum(
                        a1_changed_from_v89
                        & (
                            a1 != observed
                        )
                    )
                ),

                "a5_outside_mask_drift": int(
                    np.sum(
                        a5[~authority]
                        != observed[~authority]
                    )
                ),
                "a5_non_print_a1_regression": int(
                    np.sum(
                        a5[~print_mask]
                        != a1[~print_mask]
                    )
                ),
                "a5_print_not_equal_observed": int(
                    np.sum(
                        a5[print_mask]
                        != observed[print_mask]
                    )
                ),
                "a5_delta_from_a1_unexpected": int(
                    np.sum(
                        a5_changed_from_a1
                        & ~expected_a5_changed_from_a1
                    )
                ),
                "a5_delta_from_a1_missing": int(
                    np.sum(
                        expected_a5_changed_from_a1
                        & ~a5_changed_from_a1
                    )
                ),
                "a5_changed_byte_not_equal_observed": int(
                    np.sum(
                        a5_changed_from_a1
                        & (
                            a5 != observed
                        )
                    )
                ),

                "observed_array_sha256": _array_sha256(
                    observed
                ),
                "seed_array_sha256": _array_sha256(
                    seed
                ),
                "authority_array_sha256": _array_sha256(
                    authority
                ),
                "v7_8_array_sha256": _array_sha256(
                    geometry_t80
                ),
                "v8_9_array_sha256": _array_sha256(
                    v8_9_corrected
                ),
                "a1_array_sha256": _array_sha256(
                    a1
                ),
                "a5_array_sha256": _array_sha256(
                    a5
                ),
                "a5_print_mask_array_sha256": _array_sha256(
                    print_mask
                ),

                "a1_policy_id": result.get(
                    "a1_policy_id",
                    getattr(
                        a1_operator,
                        "a1_policy_id",
                        None,
                    ),
                ),
                "v89_operator_sha256": result.get(
                    "v89_operator_sha256",
                    getattr(
                        a1_operator,
                        "v89_operator_sha256",
                        None,
                    ),
                ),
                "a1_assembly": result.get(
                    "a1_assembly",
                    (
                        "FROZEN_V8_9_CORRECTED_THEN_ALL_"
                        "FROZEN_V7_8_AMBIGUOUS_COORDINATES_"
                        "RESTORED_TO_EXACT_OBSERVED"
                    ),
                ),

                "a5_policy_id": result.get(
                    "a5_policy_id",
                    getattr(
                        operator,
                        "a5_policy_id",
                        None,
                    ),
                ),
                "a5_bundle_id": result.get(
                    "bundle_id",
                    getattr(
                        operator,
                        "bundle_id",
                        None,
                    ),
                ),
                "a5_model_sha256": result.get(
                    "model_sha256",
                    getattr(
                        operator,
                        "model_sha256",
                        None,
                    ),
                ),
                "a5_calibrator_sha256": result.get(
                    "calibrator_sha256",
                    getattr(
                        operator,
                        "calibrator_sha256",
                        None,
                    ),
                ),
                "a5_action": result.get(
                    "a5_action",
                    "COPY_EXACT_OBSERVED_BYTE",
                ),
                "a5_fallback": result.get(
                    "a5_fallback",
                    "Notebook87 A1",
                ),
            }

            zero_fields = [
                "v7_8_outside_mask_drift",
                "telea_outside_mask_drift",
                "v8_9_outside_mask_drift",

                "a1_ambiguous_not_equal_observed",
                "a1_non_ambiguous_v89_regression",
                "a1_outside_mask_drift",
                "a1_delta_from_v89_unexpected",
                "a1_delta_from_v89_missing",
                "a1_changed_byte_not_equal_observed",

                "a5_outside_mask_drift",
                "a5_non_print_a1_regression",
                "a5_print_not_equal_observed",
                "a5_delta_from_a1_unexpected",
                "a5_delta_from_a1_missing",
                "a5_changed_byte_not_equal_observed",
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

            _write_audit_json(
                out / "page_audit.json",
                audit,
            )

            summary_rows.append(audit)

        if not summary_rows:
            raise BatchFailure(
                "No pages were processed"
            )

        upload_receipt = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "stored": True,
            "data_role": HISTORICAL_DATA_ROLE,
            "ingestion_method": HISTORICAL_INGESTION_METHOD,
            "pages": len(detector_training_receipts),
            "new_detector_training_objects": sum(
                not bool(x.get("already_existed", False))
                for x in detector_training_receipts),
            "existing_detector_training_objects": sum(
                bool(x.get("already_existed", False))
                for x in detector_training_receipts),
            "objects": detector_training_receipts,
            "correction_success_objects_created": 0,
            "correction_success_reason": (
                "Historical corpus import is not a new human A1/A5 review."),
        }
        write_json(run / "detector_training_upload.json", upload_receipt)

        summary_path = (
            run
            / "batch_summary.csv"
        )

        fieldnames = list(
            summary_rows[0].keys()
        )

        for row in summary_rows[1:]:
            if list(row.keys()) != fieldnames:
                raise BatchFailure(
                    "Page audit schemas differ between pages"
                )

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                extrasaction="raise",
            )

            writer.writeheader()
            writer.writerows(summary_rows)

        manifest = {
            "version": "HF-04.5",
            "created_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "pages": len(summary_rows),
            "input_data_role": HISTORICAL_DATA_ROLE,
            "ingestion_method": HISTORICAL_INGESTION_METHOD,
            "detector_training_upload": {
                "stored": True,
                "new_objects": upload_receipt["new_detector_training_objects"],
                "existing_objects": upload_receipt["existing_detector_training_objects"],
                "receipt": "detector_training_upload.json",
            },
            "correction_success_upload": False,

            "a1_policy_id": getattr(
                a1_operator,
                "a1_policy_id",
                None,
            ),
            "v89_operator_sha256": getattr(
                a1_operator,
                "v89_operator_sha256",
                None,
            ),
            "p_print_bundle_root": str(
                getattr(
                    a1_operator,
                    "p_print_bundle_root",
                    "",
                )
            )
            or None,
            "remove_bundle_root": str(
                getattr(
                    a1_operator,
                    "remove_bundle_root",
                    "",
                )
            )
            or None,

            "a5_policy_id": getattr(
                operator,
                "a5_policy_id",
                None,
            ),
            "a5_bundle_id": getattr(
                operator,
                "bundle_id",
                None,
            ),
            "a5_model_sha256": getattr(
                operator,
                "model_sha256",
                None,
            ),
            "a5_calibrator_sha256": getattr(
                operator,
                "calibrator_sha256",
                None,
            ),

            "class_contract": {
                "model_output_0": "ANNOTATION",
                "model_output_1": "PRINT",
                "truth_code_2": "UNCERTAIN",
                "ineligible": "INELIGIBLE",
                "probability": "p_print",
            },

            "implementation": (
                "A1Operator.run once per page, followed by "
                "A5Operator.run_from_a1; frozen V7.8; "
                "frozen V8.9; Notebook 87 A1; "
                "A5-v1 observed-PRINT processing"
            ),

            "page_lineage": {
                "v7_8": (
                    "Frozen geometry_t80 baseline: PRESERVE "
                    "copies observed, REMOVE uses the paper "
                    "estimate and AMBIGUOUS uses Telea"
                ),
                "v8_9": (
                    "Frozen V8.9 corrected page generated "
                    "from the frozen V7.8 preparation"
                ),
                "a1_notebook87": (
                    "Frozen V8.9 corrected page with every "
                    "frozen V7.8 AMBIGUOUS coordinate "
                    "restored to the exact observed byte"
                ),
                "a5_v1": (
                    "Notebook 87 A1 everywhere except final "
                    "A5 PRINT coordinates, which copy the "
                    "exact observed byte"
                ),
            },

            "integrity_contract": {
                "outside_authoritative_mask": (
                    "EXACT_OBSERVED_BYTE"
                ),
                "a1_outside_v7_8_ambiguous": (
                    "EXACT_FROZEN_V8_9_BYTE"
                ),
                "a1_at_v7_8_ambiguous": (
                    "EXACT_OBSERVED_BYTE"
                ),
                "a5_outside_final_print": (
                    "EXACT_NOTEBOOK87_A1_BYTE"
                ),
                "a5_at_final_print": (
                    "EXACT_OBSERVED_BYTE"
                ),
                "every_a5_changed_byte": (
                    "EXACT_OBSERVED_BYTE"
                ),
            },

            "input_zip_sha256": input_zip_sha256,
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
            "failed_run_directory_retained": True,
        }

        manifest_path = (
            run
            / "batch_manifest.json"
        )

        _write_audit_json(
            manifest_path,
            manifest,
        )

        # ------------------------------------------------------
        # Package inventory
        # ------------------------------------------------------

        inventory_rows = []

        for path in sorted(
            run.rglob("*")
        ):
            if not path.is_file():
                continue

            inventory_rows.append(
                {
                    "relative_path": str(
                        path.relative_to(run)
                    ),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )

        inventory_path = (
            run
            / "package_inventory.csv"
        )

        with inventory_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "relative_path",
                    "bytes",
                    "sha256",
                ],
            )

            writer.writeheader()
            writer.writerows(inventory_rows)

        # Re-write the manifest after the inventory exists so the
        # final manifest records the complete package inventory.

        manifest["package_inventory_sha256"] = (
            _sha256_file(
                inventory_path
            )
        )

        manifest["package_inventory_files"] = len(
            inventory_rows
        )

        manifest["package_inventory_bytes"] = int(
            sum(
                row["bytes"]
                for row in inventory_rows
            )
        )

        _write_audit_json(
            manifest_path,
            manifest,
        )

        # ------------------------------------------------------
        # Atomic ZIP creation
        # ------------------------------------------------------

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
                        str(
                            path.relative_to(run)
                        ),
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

            # Validate the temporary ZIP before publishing it.

            with zipfile.ZipFile(
                temporary_zip,
                "r",
            ) as verification_archive:
                bad_member = (
                    verification_archive.testzip()
                )

                if bad_member is not None:
                    raise BatchFailure(
                        "Batch ZIP integrity test failed at "
                        f"{bad_member}"
                    )

                packaged_names = {
                    item.filename
                    for item
                    in verification_archive.infolist()
                    if not item.is_dir()
                }

                expected_names = {
                    str(
                        path.relative_to(run)
                    )
                    for path
                    in run.rglob("*")
                    if path.is_file()
                }

                if packaged_names != expected_names:
                    missing_package_files = sorted(
                        expected_names
                        - packaged_names
                    )

                    unexpected_package_files = sorted(
                        packaged_names
                        - expected_names
                    )

                    raise BatchFailure(
                        "Batch ZIP file roster mismatch: "
                        f"missing={missing_package_files[:20]}, "
                        f"unexpected="
                        f"{unexpected_package_files[:20]}"
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

        if zip_path.stat().st_size == 0:
            raise BatchFailure(
                "Batch completed with an empty final ZIP"
            )

        return zip_path

    except BatchFailure:
        # Keep the run directory for forensic examination.
        # Never return a partial ZIP.
        raise

    except Exception as exc:
        # Keep the run directory for forensic examination.
        # Convert unexpected failures into the batch exception type.
        raise BatchFailure(
            "Unexpected HF-04.5 batch failure: "
            f"{type(exc).__name__}: {exc}"
        ) from exc