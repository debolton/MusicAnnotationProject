# final_evidence_upload.py

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hashlib
import json
import os
import shutil

from huggingface_hub import HfApi


# ========================================================================
# Configuration
# ========================================================================

EVIDENCE_SCHEMA_VERSION = 1

HF_EVIDENCE_BUCKET = os.environ.get(
    "HF_EVIDENCE_BUCKET",
    "",
).strip()

HF_TOKEN = os.environ.get(
    "HF_TOKEN",
    "",
).strip()

MASK_STATUSES = {
    "approved",
    "rejected",
}

CORRECTION_STATUSES = {
    "approved",
    "rejected",
}

FAILURE_STAGES = {
    "MASK",
    "A1",
    "A5",
    "OTHER",
}

FAILURE_TYPES = {
    "MASK_FALSE_POSITIVE",
    "MASK_MISSED_ANNOTATION",
    "A1_DAMAGED_PRINT",
    "A1_RETAINED_ANNOTATION",
    "A5_FALSE_PRINT_RESTORED_ANNOTATION",
    "A5_MISSED_PRINT",
    "COSMETIC_FAILURE",
    "OTHER",
}


# ========================================================================
# Hugging Face client
# ========================================================================

_HF_API: HfApi | None = None


def get_hf_api() -> HfApi:
    global _HF_API

    if not HF_EVIDENCE_BUCKET:
        raise RuntimeError(
            "HF_EVIDENCE_BUCKET is not configured."
        )

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not configured."
        )

    if _HF_API is None:
        _HF_API = HfApi(
            token=HF_TOKEN,
        )

    return _HF_API


# ========================================================================
# JSON and hashing helpers
# ========================================================================

def canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        default=str,
    ).encode(
        "utf-8"
    )


def write_json(
    path: Path,
    value: Any,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def sha256_file(
    path: Path,
) -> str:
    path = Path(
        path
    )

    if not path.is_file():
        raise RuntimeError(
            f"Cannot hash missing file: {path}"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def file_record(
    path: Path,
) -> dict:
    path = Path(
        path
    )

    return {
        "filename": path.name,
        "bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(
            path
        ),
    }


def required_path(
    directory: Path,
    filename: str,
) -> Path:
    path = (
        Path(directory)
        / filename
    )

    if not path.is_file():
        raise RuntimeError(
            "Required final-approval evidence "
            f"is missing: {filename}"
        )

    return path


# ========================================================================
# Hugging Face Storage Bucket adapter
# ========================================================================

def normalise_remote_key(
    remote_key: str,
) -> str:
    key = str(
        remote_key
    ).replace(
        "\\",
        "/",
    ).strip(
        "/"
    )

    if not key:
        raise RuntimeError(
            "Remote bucket key is empty."
        )

    if ".." in Path(
        key
    ).parts:
        raise RuntimeError(
            "Remote bucket key must not contain '..'."
        )

    return key


def put_object(
    *,
    local_path: Path,
    remote_key: str,
) -> None:
    local_path = Path(
        local_path
    )

    remote_key = normalise_remote_key(
        remote_key
    )

    if not local_path.is_file():
        raise RuntimeError(
            "Local upload file does not exist: "
            f"{local_path}"
        )

    api = get_hf_api()

    api.batch_bucket_files(
        bucket_id=HF_EVIDENCE_BUCKET,
        add=[
            (
                local_path,
                remote_key,
            ),
        ],
    )


def put_objects(
    *,
    files: list[
        tuple[
            Path,
            str,
        ]
    ],
) -> None:
    if not files:
        raise RuntimeError(
            "No bucket objects were supplied "
            "for upload."
        )

    additions = []

    for local_path, remote_key in files:
        local_path = Path(
            local_path
        )

        remote_key = normalise_remote_key(
            remote_key
        )

        if not local_path.is_file():
            raise RuntimeError(
                "Local upload file does not exist: "
                f"{local_path}"
            )

        additions.append(
            (
                local_path,
                remote_key,
            )
        )

    api = get_hf_api()

    api.batch_bucket_files(
        bucket_id=HF_EVIDENCE_BUCKET,
        add=additions,
    )


def object_exists(
    *,
    remote_key: str,
) -> bool:
    remote_key = normalise_remote_key(
        remote_key
    )

    api = get_hf_api()

    matches = list(
        api.get_bucket_paths_info(
            bucket_id=HF_EVIDENCE_BUCKET,
            paths=[
                remote_key,
            ],
        )
    )

    return any(
        str(
            getattr(
                item,
                "path",
                "",
            )
        ).strip(
            "/"
        )
        == remote_key
        for item in matches
    )


def verify_object(
    *,
    local_path: Path,
    remote_key: str,
) -> None:
    local_path = Path(
        local_path
    )

    remote_key = normalise_remote_key(
        remote_key
    )

    if not local_path.is_file():
        raise RuntimeError(
            "Cannot verify missing local file: "
            f"{local_path}"
        )

    api = get_hf_api()

    metadata = api.get_bucket_file_metadata(
        bucket_id=HF_EVIDENCE_BUCKET,
        remote_path=remote_key,
    )

    remote_size = getattr(
        metadata,
        "size",
        None,
    )

    if remote_size is None:
        raise RuntimeError(
            "Hugging Face returned no size for "
            f"bucket object {remote_key!r}."
        )

    local_size = int(
        local_path.stat().st_size
    )

    remote_size = int(
        remote_size
    )

    if remote_size != local_size:
        raise RuntimeError(
            "Uploaded object size mismatch: "
            f"{remote_key}: "
            f"local={local_size}, "
            f"remote={remote_size}"
        )


# ========================================================================
# Review contract
# ========================================================================

def normalise_final_review(
    review: dict,
) -> dict:
    """
    Supports both:

        New review format:
            mask_status
            correction_status

        Legacy review format:
            status == "approved"

    A legacy approved review is interpreted as approval of both the
    mask and correction. New review records should use the two explicit
    fields.
    """

    if not isinstance(
        review,
        dict,
    ):
        raise RuntimeError(
            "Final review must be a dictionary."
        )

    result = dict(
        review
    )

    if (
        "mask_status"
        not in result
        and result.get(
            "status"
        )
        == "approved"
    ):
        result[
            "mask_status"
        ] = "approved"

    if (
        "correction_status"
        not in result
        and result.get(
            "status"
        )
        == "approved"
    ):
        result[
            "correction_status"
        ] = "approved"

    return result


def validate_final_review(
    review: dict,
) -> dict:
    review = normalise_final_review(
        review
    )

    mask_status = review.get(
        "mask_status"
    )

    correction_status = review.get(
        "correction_status"
    )

    if mask_status not in MASK_STATUSES:
        raise RuntimeError(
            "Final review must declare mask_status "
            f"as one of {sorted(MASK_STATUSES)}."
        )

    if correction_status not in CORRECTION_STATUSES:
        raise RuntimeError(
            "Final review must declare correction_status "
            f"as one of {sorted(CORRECTION_STATUSES)}."
        )

    if (
        mask_status == "approved"
        and correction_status == "rejected"
    ):
        failure_stage = review.get(
            "failure_stage"
        )

        failure_type = review.get(
            "failure_type"
        )

        if failure_stage not in FAILURE_STAGES:
            raise RuntimeError(
                "Rejected correction must declare "
                "a valid failure_stage."
            )

        if failure_type not in FAILURE_TYPES:
            raise RuntimeError(
                "Rejected correction must declare "
                "a valid failure_type."
            )

    return review


# ========================================================================
# Content-addressed identities
# ========================================================================

def training_sample_id(
    annotated_path: Path,
    authoritative_mask_path: Path,
) -> str:
    digest = hashlib.sha256()

    for path in (
        Path(
            annotated_path
        ),
        Path(
            authoritative_mask_path
        ),
    ):
        if not path.is_file():
            raise RuntimeError(
                "Cannot calculate training sample ID "
                f"from missing file: {path}"
            )

        with path.open(
            "rb"
        ) as handle:
            for block in iter(
                lambda: handle.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(
                    block
                )

    return digest.hexdigest()


def correction_evidence_id(
    *,
    sample_id: str,
    a1_path: Path,
    a5_path: Path,
    review: dict,
    a1_policy_id: str,
    a5_policy_id: str,
) -> str:
    digest = hashlib.sha256()

    digest.update(
        str(
            sample_id
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        sha256_file(
            a1_path
        ).encode(
            "ascii"
        )
    )

    digest.update(
        sha256_file(
            a5_path
        ).encode(
            "ascii"
        )
    )

    digest.update(
        canonical_json_bytes(
            review
        )
    )

    digest.update(
        str(
            a1_policy_id
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            a5_policy_id
        ).encode(
            "utf-8"
        )
    )

    return digest.hexdigest()


# ========================================================================
# Detector-training evidence staging
# ========================================================================

def stage_detector_training_object(
    *,
    directory: Path,
    staging_root: Path,
    sample_id: str,
    review: dict,
    sid: str,
    manifest: dict,
) -> Path:
    annotated = required_path(
        directory,
        "annotated_page.png",
    )

    authoritative_mask = required_path(
        directory,
        "reviewed_authoritative_seed_mask.png",
    )

    target = (
        Path(staging_root)
        / "detector_training"
    )

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    staged_annotated = (
        target
        / "annotated.png"
    )

    staged_mask = (
        target
        / "authoritative_mask.png"
    )

    shutil.copy2(
        annotated,
        staged_annotated,
    )

    shutil.copy2(
        authoritative_mask,
        staged_mask,
    )

    metadata = {
        "schema_version": (
            EVIDENCE_SCHEMA_VERSION
        ),

        "evidence_role": (
            "DETECTOR_TRAINING_CANDIDATE"
        ),

        "sample_id": sample_id,

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_session_id": sid,

        "mask_status": review[
            "mask_status"
        ],

        "correction_status": review[
            "correction_status"
        ],

        "annotated_image": file_record(
            staged_annotated
        ),

        "authoritative_mask": {
            **file_record(
                staged_mask
            ),

            "semantics": (
                "HUMAN_REVIEWED_UNDILATED_"
                "ANNOTATION_SEED_MASK"
            ),

            "dilation": 0,
        },

        "raw_yolo_prediction_is_truth": False,

        "prepared_dilation2_mask_is_training_label": False,

        "a1_policy_id": manifest.get(
            "a1_policy_id"
        ),

        "a5_policy_id": manifest.get(
            "a5_policy_id"
        ),

        "a5_bundle_id": manifest.get(
            "a5_bundle_id"
        ),

        "training_eligibility": {
            "detector_training": True,
            "a5_pixel_training": False,
            "independent_holdout": False,
        },

        "review": review,
    }

    write_json(
        target
        / "metadata.json",
        metadata,
    )

    return target


# ========================================================================
# Correction evidence staging
# ========================================================================

def stage_correction_object(
    *,
    directory: Path,
    staging_root: Path,
    evidence_id: str,
    sample_id: str,
    review: dict,
    sid: str,
    manifest: dict,
) -> tuple[
    Path,
    str,
]:
    correction_status = review[
        "correction_status"
    ]

    if correction_status == "approved":
        role = (
            "CORRECTION_SUCCESS_MONITORING"
        )

        prefix = (
            "correction_success"
        )

        filenames = [
            "a1_corrected.png",
            "a5_corrected.png",
            "a5_print_mask.png",
            "operational_correction_audit.json",
            "whole_page_review.json",
        ]

    else:
        role = (
            "CORRECTION_FAILURE_EVIDENCE"
        )

        prefix = (
            "correction_failure"
        )

        filenames = [
            "annotated_page.png",
            "reviewed_authoritative_seed_mask.png",

            "v7_8_corrected.png",
            "v8_9_corrected.png",

            "a1_corrected.png",
            "a5_corrected.png",

            "a5_eligible.png",
            "a5_print_mask.png",
             "a5_delta_from_a1.png",

            "a5_raw_print_probability.npy",
            "a5_print_probability.npy",

            "a5_provisional_decision.npy",
            "a5_final_decision.npy",

            "a5_channels.npz",

            "operational_correction_audit.json",
            "whole_page_review.json",
        ]

    target = (
        Path(staging_root)
        / prefix
    )

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    copied_files = []

    for ilename in filenames:
        source = required_path(
            directory,
            filename,
        )

        destination = (
            target
            / filename
        )

        shutil.copy2(
            source,
            destination,
        )

        copied_files.append(
            file_record(
                destination
            )
        )

    metadata = {
        "schema_version": (
            EVIDENCE_SCHEMA_VERSION
        ),

        "evidence_role": role,

        "sample_id": sample_id,

        "evidence_id": evidence_id,

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_session_id": sid,

        "mask_status": review[
            "mask_status"
        ],

        "correction_status": (
            correction_status
        ),

        "failure_stage": review.get(
            "failure_stage"
        ),

        "failure_type": review.get(
            "failure_type"
        ),

        "notes": review.get(
            "notes",
            "",
        ),

        "a1_policy_id": manifest.get(
            "a1_policy_id"
        ),

        "a5_policy_id": manifest.get(
            "a5_policy_id"
        ),

        "a5_bundle_id": manifest.get(
            "a5_bundle_id"
        ),

        "files": copied_files,

        "training_eligibility": {
            "detector_training": True,

            "a5_pixel_training": False,

            "correction_monitoring": (
                correction_status
                == "approved"
            ),

            "correction_failure_research": (
                correction_status
                == "rejected"
            ),
        },

        "review": review,
    }

    write_json(
        target
        / "metadata.json",
        metadata,
    )

    return (
        target,
        prefix,
    )


# ========================================================================
# Upload and commit-marker protocol
# ========================================================================

def commit_staged_object(
    *,
    staged_directory: Path,
    final_prefix: str,
    object_id: str,
) -> dict:
    staged_directory = Path(
        staged_directory
    )

    if not staged_directory.is_dir():
        raise RuntimeError(
            "Staged evidence directory is missing: "
            f"{staged_directory}"
        )

    final_prefix = str(
        final_prefix
    ).strip(
        "/"
    )

    object_id = str(
        object_id
    ).strip()

    if not final_prefix:
        raise RuntimeError(
            "Evidence prefix is empty."
        )

    if not object_id:
        raise RuntimeError(
            "Evidence object ID is empty."
        )

    remote_root = (
        f"{final_prefix}/"
        f"objects/"
        f"{object_id}"
    )

    commit_key = (
        f"{remote_root}/COMMIT.json"
    )

    # Idempotent retry. A valid commit marker is the remote declaration
    # that the object has already been completed.
    if object_exists(
        remote_key=commit_key
    ):
        return {
            "committed": True,
            "already_existed": True,
            "prefix": remote_root,
            "commit_object": commit_key,
        }

    files = sorted(
        path
        for path in staged_directory.iterdir()
        if (
            path.is_file()
            and path.name
            != "COMMIT.json"
        )
    )

    if not files:
        raise RuntimeError(
            "Refusing to commit an empty "
            "evidence object."
        )

    upload_pairs = [
        (
            path,
            f"{remote_root}/{path.name}",
        )
        for path in files
    ]

    # Hugging Face bucket batch operations are not transactional.
    # COMMIT.json is therefore deliberately excluded from this batch.
    put_objects(
        files=upload_pairs
    )

    uploaded = []

    for path, remote_key in upload_pairs:
        verify_object(
            local_path=path,
            remote_key=remote_key,
        )

        uploaded.append(
            {
                "name": path.name,
                "remote_key": remote_key,
                "bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

    object_manifest = {
        "schema_version": (
            EVIDENCE_SCHEMA_VERSION
        ),

        "object_id": object_id,

        "remote_prefix": (
            remote_root
        ),

        "files": uploaded,
    }

    manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(
            object_manifest
        )
    ).hexdigest()

    commit_record = {
        "schema_version": (
            EVIDENCE_SCHEMA_VERSION
        ),

        "status": "COMMITTED",

        "object_id": object_id,

        "committed_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "manifest_sha256": (
            manifest_sha256
        ),

        "object_count": len(
            uploaded
        ),

        "total_bytes": int(
            sum(
                item[
                    "bytes"
                ]
                for item in uploaded
            )
        ),

        "files": uploaded,
    }

    local_commit = (
        staged_directory
        / "COMMIT.json"
    )

    write_json(
        local_commit,
        commit_record,
    )

    # COMMIT.json must always be uploaded last.
    put_object(
        local_path=local_commit,
        remote_key=commit_key,
    )

    verify_object(
        local_path=local_commit,
        remote_key=commit_key,
    )

    if not object_exists(
        remote_key=commit_key
    ):
        raise RuntimeError(
            "Evidence payload was uploaded, but "
            "COMMIT.json cannot be verified."
        )

    return {
        "committed": True,
        "already_existed": False,
        "prefix": remote_root,
        "commit_object": commit_key,
        "manifest_sha256": manifest_sha256,
        "object_count": len(
            uploaded
        ),
        "total_bytes": (
            commit_record[
                "total_bytes"
            ]
        ),
    }


# ========================================================================
# Main final-approval operation
# ========================================================================

def commit_final_approval_evidence(
    *,
    directory: Path,
    sid: str,
    review: dict,
    package_manifest: dict,
) -> dict:
    """
    Persist final reviewed evidence.

    Behaviour:

        mask rejected
            -> no authoritative training upload

        mask approved + correction approved
            -> detector-training pair
            -> correction-success monitoring pack

        mask approved + correction rejected
            -> detector-training pair
            -> correction-failure evidence pack
    """

    directory = Path(
        directory
    )

    if not directory.is_dir():
        raise RuntimeError(
            f"Session directory does not exist: {directory}"
        )

    # Fail early if bucket access is not configured.
    get_hf_api()

    review = validate_final_review(
        review
    )

    if review[
        "mask_status"
    ] != "approved":
        receipt = {
            "schema_version": (
                EVIDENCE_SCHEMA_VERSION
            ),

            "stored": False,

            "reason": (
                "MASK_NOT_APPROVED"
            ),

            "mask_status": review[
                "mask_status"
            ],

            "correction_status": review[
                "correction_status"
            ],

            "detector_training": None,

            "correction_evidence": None,
        }

        write_json(
            directory
            / "training_evidence_upload.json",
            receipt,
        )

        return receipt

    annotated = required_path(
        directory,
        "annotated_page.png",
    )

    authoritative_mask = required_path(
        directory,
        "reviewed_authoritative_seed_mask.png",
    )

    a1_path = required_path(
        directory,
        "a1_corrected.png",
    )

    a5_path = required_path(
        directory,
        "a5_corrected.png",
    )

    sample_id = training_sample_id(
        annotated,
        authoritative_mask,
    )

    evidence_id = correction_evidence_id(
        sample_id=sample_id,

        a1_path=a1_path,

        a5_path=a5_path,

        review=review,

        a1_policy_id=str(
            package_manifest.get(
                "a1_policy_id",
                "",
            )
        ),

        a5_policy_id=str(
            package_manifest.get(
                "a5_policy_id",
                "",
            )
        ),
    )

    staging_root = (
        directory
        / "_final_evidence_staging"
    )

    if staging_root.exists():
        shutil.rmtree(
            staging_root
        )

    staging_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        detector_stage = (
            stage_detector_training_object(
                directory=directory,
                staging_root=staging_root,
                sample_id=sample_id,
                review=review,
                sid=sid,
                manifest=package_manifest,
            )
        )

        (
            correction_stage,
            correction_prefix,
        ) = stage_correction_object(
            directory=directory,
            staging_root=staging_root,
            evidence_id=evidence_id,
            sample_id=sample_id,
            review=review,
            sid=sid,
            manifest=package_manifest,
        )

        detector_receipt = (
            commit_staged_object(
                staged_directory=(
                    detector_stage
                ),

                final_prefix=(
                    "detector_training"
                ),

                object_id=sample_id,
            )
        )

        correction_receipt = (
            commit_staged_object(
                staged_directory=(
                    correction_stage
                ),

                final_prefix=(
                    correction_prefix
                ),

                object_id=evidence_id,
            )
        )

        receipt = {
            "schema_version": (
                EVIDENCE_SCHEMA_VERSION
            ),

            "stored": True,

            "created_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            "bucket": (
                HF_EVIDENCE_BUCKET
            ),

            "sample_id": (
                sample_id
            ),

            "evidence_id": (
                evidence_id
            ),

            "mask_status": review[
                "mask_status"
            ],

            "correction_status": review[
                "correction_status"
            ],

            "failure_stage": review.get(
                "failure_stage"
            ),

            "failure_type": review.get(
                "failure_type"
            ),

            "detector_training": (
                detector_receipt
            ),

            "correction_evidence": (
                correction_receipt
            ),
        }

        write_json(
            directory
            / "training_evidence_upload.json",
            receipt,
        )

        return receipt

    finally:
        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )