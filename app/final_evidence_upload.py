# final_evidence_upload.py

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import hashlib
import json
import os
import shutil

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError


# ========================================================================
# Configuration
# ========================================================================

EVIDENCE_SCHEMA_VERSION = 3

# Hugging Face namespace, for example: debolton
HF_S3_NAMESPACE = os.environ.get(
    "HF_S3_NAMESPACE",
    "debolton",
).strip()

# Bare bucket name only, for example: MusicAnnotationProject-storage
# Do not include the namespace and do not use s3://.
HF_EVIDENCE_BUCKET = os.environ.get(
    "HF_EVIDENCE_BUCKET",
    "MusicAnnotationProject-storage",
).strip()

# S3 credentials generated from the dropdown for a write-enabled HF token.
HF_S3_ACCESS_KEY_ID = os.environ.get(
    "HF_S3_ACCESS_KEY_ID",
    os.environ.get("AWS_ACCESS_KEY_ID", ""),
).strip()

HF_S3_SECRET_ACCESS_KEY = os.environ.get(
    "HF_S3_SECRET_ACCESS_KEY",
    os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
).strip()

HF_S3_ENDPOINT_URL = os.environ.get(
    "HF_S3_ENDPOINT_URL",
    "https://debolton.s3.hf.co",
).strip()

HF_S3_REGION = os.environ.get(
    "HF_S3_REGION",
    "us-east-1",
).strip() or "us-east-1"

HF_S3_PREFIX = os.environ.get(
    "HF_S3_PREFIX",
    "music_annotation_evidence",
).strip().strip("/")

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
    "A5_N88",
    "OTHER",
}

FAILURE_TYPES = {
    "MASK_FALSE_POSITIVE",
    "MASK_MISSED_ANNOTATION",
    "A1_DAMAGED_PRINT",
    "A1_RETAINED_ANNOTATION",
    "A5_N88_FALSE_PRINT_RESTORED_ANNOTATION",
    "A5_N88_MISSED_PRINT",
    "COSMETIC_FAILURE",
    "OTHER",
}


# ========================================================================
# S3 client
# ========================================================================

_S3_CLIENT = None

_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=256 * 1024 * 1024,
    multipart_chunksize=256 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)


def get_s3_client():
    global _S3_CLIENT

    if not HF_S3_NAMESPACE:
        raise RuntimeError(
            "HF_S3_NAMESPACE is not configured. "
            "Use the Hugging Face username or organisation that owns the bucket."
        )

    if not HF_EVIDENCE_BUCKET:
        raise RuntimeError(
            "HF_EVIDENCE_BUCKET is not configured. "
            "Use the bare bucket name, without the namespace."
        )

    if "/" in HF_EVIDENCE_BUCKET:
        raise RuntimeError(
            "HF_EVIDENCE_BUCKET must be the bare bucket name only. "
            "Put the namespace in HF_S3_NAMESPACE."
        )

    if not HF_S3_ACCESS_KEY_ID:
        raise RuntimeError(
            "HF_S3_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID is not configured."
        )

    if not HF_S3_SECRET_ACCESS_KEY:
        raise RuntimeError(
            "HF_S3_SECRET_ACCESS_KEY or AWS_SECRET_ACCESS_KEY is not configured."
        )

    if not HF_S3_ACCESS_KEY_ID.startswith("HFAK"):
        raise RuntimeError(
            "The configured access key does not look like a Hugging Face "
            "S3 access key. Generate S3 credentials from the HF token dropdown."
        )

    endpoint_url = (
        HF_S3_ENDPOINT_URL
        or f"https://{HF_S3_NAMESPACE}.s3.hf.co"
    )

    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=HF_S3_ACCESS_KEY_ID,
            aws_secret_access_key=HF_S3_SECRET_ACCESS_KEY,
            config=Config(
                region_name=HF_S3_REGION,
                s3={
                    "addressing_style": "path",
                },
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                retries={
                    "max_attempts": 8,
                    "mode": "standard",
                },
            ),
        )

    return _S3_CLIENT


# ========================================================================
# JSON and hashing helpers
# ========================================================================


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
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
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    path = Path(path)

    if not path.is_file():
        raise RuntimeError(f"Cannot hash missing file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def file_record(path: Path) -> dict:
    path = Path(path)
    return {
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def required_path(directory: Path, filename: str) -> Path:
    path = Path(directory) / filename
    if not path.is_file():
        raise RuntimeError(
            "Required final-approval evidence is missing: "
            f"{filename}"
        )
    return path


# ========================================================================
# Hugging Face S3-compatible Storage Bucket adapter
# ========================================================================


def normalise_remote_key(remote_key: str) -> str:
    raw = str(remote_key).strip()

    if not raw:
        raise RuntimeError("Remote bucket key is empty.")

    if "\\" in raw or "\x00" in raw:
        raise RuntimeError(
            "Remote bucket key must not contain backslashes or null bytes."
        )

    if raw.startswith("/") or raw.endswith("/"):
        raise RuntimeError(
            "Remote bucket key must not start or end with '/'."
        )

    if "//" in raw:
        raise RuntimeError(
            "Remote bucket key must not contain consecutive slashes."
        )

    if raw.startswith("./") or raw.endswith(".."):
        raise RuntimeError("Remote bucket key has an invalid relative form.")

    parts = PurePosixPath(raw).parts
    if ".." in parts:
        raise RuntimeError(
            "Remote bucket key must not contain '..' path segments."
        )

    return "/".join(parts)


def prefixed_key(remote_key: str) -> str:
    key = normalise_remote_key(remote_key)
    if not HF_S3_PREFIX:
        return key
    return normalise_remote_key(f"{HF_S3_PREFIX}/{key}")


def object_exists(*, remote_key: str) -> bool:
    key = prefixed_key(remote_key)
    client = get_s3_client()

    try:
        client.head_object(
            Bucket=HF_EVIDENCE_BUCKET,
            Key=key,
        )
        return True
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode"
        )
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if status == 404 or code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False
        raise RuntimeError(
            f"Unable to inspect bucket object {key!r}: {exc}"
        ) from exc


def put_object(*, local_path: Path, remote_key: str) -> None:
    local_path = Path(local_path)
    key = prefixed_key(remote_key)

    if not local_path.is_file():
        raise RuntimeError(
            f"Local upload file does not exist: {local_path}"
        )

    client = get_s3_client()
    client.upload_file(
        Filename=str(local_path),
        Bucket=HF_EVIDENCE_BUCKET,
        Key=key,
        Config=_TRANSFER_CONFIG,
    )


def put_objects(*, files: list[tuple[Path, str]]) -> None:
    if not files:
        raise RuntimeError("No bucket objects were supplied for upload.")

    for local_path, remote_key in files:
        put_object(
            local_path=Path(local_path),
            remote_key=remote_key,
        )


def verify_object(*, local_path: Path, remote_key: str) -> dict:
    local_path = Path(local_path)
    key = prefixed_key(remote_key)

    if not local_path.is_file():
        raise RuntimeError(
            f"Cannot verify missing local file: {local_path}"
        )

    client = get_s3_client()

    try:
        metadata = client.head_object(
            Bucket=HF_EVIDENCE_BUCKET,
            Key=key,
        )
    except ClientError as exc:
        raise RuntimeError(
            f"Uploaded object cannot be verified: {key!r}: {exc}"
        ) from exc

    remote_size = int(metadata.get("ContentLength", -1))
    local_size = int(local_path.stat().st_size)

    if remote_size != local_size:
        raise RuntimeError(
            "Uploaded object size mismatch: "
            f"{key}: local={local_size}, remote={remote_size}"
        )

    return {
        "bucket": HF_EVIDENCE_BUCKET,
        "key": key,
        "bytes": remote_size,
        "etag": str(metadata.get("ETag", "")).strip('"'),
        "local_sha256": sha256_file(local_path),
    }


# ========================================================================
# Review contract
# ========================================================================


def normalise_final_review(review: dict) -> dict:
    if not isinstance(review, dict):
        raise RuntimeError("Final review must be a dictionary.")

    result = dict(review)

    if (
        "mask_status" not in result
        and result.get("status") == "approved"
    ):
        result["mask_status"] = "approved"

    if (
        "correction_status" not in result
        and result.get("status") == "approved"
    ):
        result["correction_status"] = "approved"

    return result


def validate_final_review(review: dict) -> dict:
    review = normalise_final_review(review)

    mask_status = review.get("mask_status")
    correction_status = review.get("correction_status")

    if mask_status not in MASK_STATUSES:
        raise RuntimeError(
            "Final review must declare mask_status as one of "
            f"{sorted(MASK_STATUSES)}."
        )

    if correction_status not in CORRECTION_STATUSES:
        raise RuntimeError(
            "Final review must declare correction_status as one of "
            f"{sorted(CORRECTION_STATUSES)}."
        )

    if mask_status == "approved" and correction_status == "rejected":
        failure_stage = review.get("failure_stage")
        failure_type = review.get("failure_type")

        if failure_stage not in FAILURE_STAGES:
            raise RuntimeError(
                "Rejected correction must declare a valid failure_stage."
            )

        if failure_type not in FAILURE_TYPES:
            raise RuntimeError(
                "Rejected correction must declare a valid failure_type."
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
        Path(annotated_path),
        Path(authoritative_mask_path),
    ):
        if not path.is_file():
            raise RuntimeError(
                "Cannot calculate training sample ID from missing file: "
                f"{path}"
            )

        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)

    return digest.hexdigest()


def correction_evidence_id(
    *,
    sample_id: str,
    operational_path: Path,
    a1_path: Path,
    a5_n88_path: Path,
    review: dict,
    operational_policy_id: str,
    a1_policy_id: str,
    a5_n88_policy_id: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(sample_id).encode("utf-8"))
    digest.update(sha256_file(operational_path).encode("ascii"))
    digest.update(sha256_file(a1_path).encode("ascii"))
    digest.update(sha256_file(a5_n88_path).encode("ascii"))
    digest.update(canonical_json_bytes(review))
    digest.update(str(operational_policy_id).encode("utf-8"))
    digest.update(str(a1_policy_id).encode("utf-8"))
    digest.update(str(a5_n88_policy_id).encode("utf-8"))
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
    annotated = required_path(directory, "annotated_page.png")
    authoritative_mask = required_path(
        directory,
        "reviewed_authoritative_seed_mask.png",
    )

    target = Path(staging_root) / "detector_training"
    target.mkdir(parents=True, exist_ok=False)

    staged_annotated = target / "annotated.png"
    staged_mask = target / "authoritative_mask.png"

    shutil.copy2(annotated, staged_annotated)
    shutil.copy2(authoritative_mask, staged_mask)

    metadata = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_role": "DETECTOR_TRAINING_CANDIDATE",
        "sample_id": sample_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_session_id": sid,
        "mask_status": review["mask_status"],
        "correction_status": review["correction_status"],
        "annotated_image": file_record(staged_annotated),
        "authoritative_mask": {
            **file_record(staged_mask),
            "semantics": (
                "HUMAN_REVIEWED_UNDILATED_ANNOTATION_SEED_MASK"
            ),
            "dilation": 0,
        },
        "raw_yolo_prediction_is_truth": False,
        "prepared_dilation2_mask_is_training_label": False,
        "operational_output_policy_id": manifest.get(
            "operational_output_policy_id"
        ),
        "a1_policy_id": manifest.get("a1_policy_id"),
        "a5_n88_policy_id": manifest.get("a5_n88_policy_id"),
        "a5_n88_bundle_id": manifest.get("a5_n88_bundle_id"),
        "training_eligibility": {
            "detector_training": True,
            "a5_n88_pixel_training": False,
            "independent_holdout": False,
        },
        "review": review,
    }

    write_json(target / "metadata.json", metadata)
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
) -> tuple[Path, str]:
    correction_status = review["correction_status"]

    if correction_status == "approved":
        role = "CORRECTION_SUCCESS_MONITORING"
        prefix = "correction_success"
        filenames = [
            "operational_corrected.png",
            "a1_corrected.png",
            "a5_n88_corrected.png",
            "a5_n88_decision_overlay.png",
            "a5_n88_live_pixel_diagnostics.csv",
            "a5_n88_final_print_diagnostics.csv",
            "a5_n88_diagnostic_export_manifest.json",
            "operational_correction_audit.json",
            "whole_page_review.json",
        ]
    else:
        role = "CORRECTION_FAILURE_EVIDENCE"
        prefix = "correction_failure"
        filenames = [
            "annotated_page.png",
            "reviewed_authoritative_seed_mask.png",
            "operational_corrected.png",
            "a1_corrected.png",
            "a5_n88_corrected.png",
            "a5_n88_decision_overlay.png",
            "a5_n88_live_pixel_diagnostics.csv",
            "a5_n88_final_print_diagnostics.csv",
            "a5_n88_diagnostic_export_manifest.json",
            "operational_correction_audit.json",
            "whole_page_review.json",
        ]
    target = Path(staging_root) / prefix
    target.mkdir(parents=True, exist_ok=False)

    copied_files = []

    for filename in filenames:
        source = required_path(directory, filename)
        destination = target / filename
        shutil.copy2(source, destination)
        copied_files.append(file_record(destination))

    metadata = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_role": role,
        "sample_id": sample_id,
        "evidence_id": evidence_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_session_id": sid,
        "mask_status": review["mask_status"],
        "correction_status": correction_status,
        "failure_stage": review.get("failure_stage"),
        "failure_type": review.get("failure_type"),
        "notes": review.get("notes", ""),
        "operational_output_stage": review.get(
            "operational_output_stage"
        ),
        "a5_n88_applied_to_operational_output": bool(
            review.get("a5_n88_applied_to_operational_output", False)
        ),
        "a5_n88_output_role": (
            "OPERATIONAL"
            if review.get("a5_n88_applied_to_operational_output", False)
            else "SHADOW_PROPOSAL"
        ),
        "operational_output_policy_id": manifest.get(
            "operational_output_policy_id"
        ),
        "a1_policy_id": manifest.get("a1_policy_id"),
        "a5_n88_policy_id": manifest.get("a5_n88_policy_id"),
        "a5_n88_bundle_id": manifest.get("a5_n88_bundle_id"),
        "files": copied_files,
        "training_eligibility": {
            "detector_training": True,
            "a5_n88_pixel_training": False,
            "correction_monitoring": correction_status == "approved",
            "correction_failure_research": correction_status == "rejected",
        },
        "review": review,
    }

    write_json(target / "metadata.json", metadata)
    return target, prefix


# ========================================================================
# Upload and commit-marker protocol
# ========================================================================


def commit_staged_object(
    *,
    staged_directory: Path,
    final_prefix: str,
    object_id: str,
) -> dict:
    staged_directory = Path(staged_directory)

    if not staged_directory.is_dir():
        raise RuntimeError(
            f"Staged evidence directory is missing: {staged_directory}"
        )

    final_prefix = normalise_remote_key(str(final_prefix).strip("/"))
    object_id = str(object_id).strip()

    if not object_id:
        raise RuntimeError("Evidence object ID is empty.")

    remote_root = normalise_remote_key(
        f"{final_prefix}/objects/{object_id}"
    )
    commit_key = normalise_remote_key(f"{remote_root}/COMMIT.json")

    # Idempotent retry. COMMIT.json is the declaration that the object is
    # complete. Payload objects without COMMIT.json are treated as incomplete
    # and are safely overwritten on retry.
    if object_exists(remote_key=commit_key):
        return {
            "committed": True,
            "already_existed": True,
            "bucket": HF_EVIDENCE_BUCKET,
            "prefix": prefixed_key(remote_root),
            "commit_object": prefixed_key(commit_key),
        }

    files = sorted(
        path
        for path in staged_directory.iterdir()
        if path.is_file() and path.name != "COMMIT.json"
    )

    if not files:
        raise RuntimeError("Refusing to commit an empty evidence object.")

    upload_pairs = [
        (path, f"{remote_root}/{path.name}")
        for path in files
    ]

    # S3 multi-object uploads are not transactional. Upload and verify the
    # payload first, then upload COMMIT.json last.
    put_objects(files=upload_pairs)

    uploaded = []
    for path, remote_key in upload_pairs:
        verification = verify_object(
            local_path=path,
            remote_key=remote_key,
        )
        uploaded.append(
            {
                "name": path.name,
                "remote_key": verification["key"],
                "bytes": verification["bytes"],
                "sha256": verification["local_sha256"],
                "etag": verification["etag"],
            }
        )

    object_manifest = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "object_id": object_id,
        "bucket": HF_EVIDENCE_BUCKET,
        "remote_prefix": prefixed_key(remote_root),
        "files": uploaded,
    }

    manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(object_manifest)
    ).hexdigest()

    commit_record = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "COMMITTED",
        "object_id": object_id,
        "committed_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": HF_EVIDENCE_BUCKET,
        "remote_prefix": prefixed_key(remote_root),
        "manifest_sha256": manifest_sha256,
        "object_count": len(uploaded),
        "total_bytes": int(sum(item["bytes"] for item in uploaded)),
        "files": uploaded,
    }

    local_commit = staged_directory / "COMMIT.json"
    write_json(local_commit, commit_record)

    put_object(
        local_path=local_commit,
        remote_key=commit_key,
    )
    commit_verification = verify_object(
        local_path=local_commit,
        remote_key=commit_key,
    )

    if not object_exists(remote_key=commit_key):
        raise RuntimeError(
            "Evidence payload was uploaded, but COMMIT.json cannot be verified."
        )

    return {
        "committed": True,
        "already_existed": False,
        "bucket": HF_EVIDENCE_BUCKET,
        "prefix": prefixed_key(remote_root),
        "commit_object": commit_verification["key"],
        "manifest_sha256": manifest_sha256,
        "object_count": len(uploaded),
        "total_bytes": commit_record["total_bytes"],
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
    """Persist final reviewed evidence to the HF S3-compatible bucket.

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

    directory = Path(directory)

    if not directory.is_dir():
        raise RuntimeError(
            f"Session directory does not exist: {directory}"
        )

    # Fail before staging if S3 access is not configured.
    get_s3_client()
    review = validate_final_review(review)

    if review["mask_status"] != "approved":
        receipt = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "stored": False,
            "reason": "MASK_NOT_APPROVED",
            "mask_status": review["mask_status"],
            "correction_status": review["correction_status"],
            "detector_training": None,
            "correction_evidence": None,
        }
        write_json(directory / "training_evidence_upload.json", receipt)
        return receipt

    annotated = required_path(directory, "annotated_page.png")
    authoritative_mask = required_path(
        directory,
        "reviewed_authoritative_seed_mask.png",
    )
    operational_path = required_path(
        directory,
        "operational_corrected.png",
    )
    a1_path = required_path(directory, "a1_corrected.png")
    a5_n88_path = required_path(directory, "a5_n88_corrected.png")

    sample_id = training_sample_id(
        annotated,
        authoritative_mask,
    )

    evidence_id = correction_evidence_id(
        sample_id=sample_id,
        operational_path=operational_path,
        a1_path=a1_path,
        a5_n88_path=a5_n88_path,
        review=review,
        operational_policy_id=str(
            package_manifest.get("operational_output_policy_id", "")
        ),
        a1_policy_id=str(package_manifest.get("a1_policy_id", "")),
        a5_n88_policy_id=str(package_manifest.get("a5_n88_policy_id", "")),
    )

    staging_root = directory / "_final_evidence_staging"

    if staging_root.exists():
        shutil.rmtree(staging_root)

    staging_root.mkdir(parents=True, exist_ok=False)

    try:
        detector_stage = stage_detector_training_object(
            directory=directory,
            staging_root=staging_root,
            sample_id=sample_id,
            review=review,
            sid=sid,
            manifest=package_manifest,
        )

        correction_stage, correction_prefix = stage_correction_object(
            directory=directory,
            staging_root=staging_root,
            evidence_id=evidence_id,
            sample_id=sample_id,
            review=review,
            sid=sid,
            manifest=package_manifest,
        )

        detector_receipt = commit_staged_object(
            staged_directory=detector_stage,
            final_prefix="detector_training",
            object_id=sample_id,
        )

        correction_receipt = commit_staged_object(
            staged_directory=correction_stage,
            final_prefix=correction_prefix,
            object_id=evidence_id,
        )

        receipt = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "storage_protocol": "HF_S3_COMPATIBLE",
            "stored": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "namespace": HF_S3_NAMESPACE,
            "bucket": HF_EVIDENCE_BUCKET,
            "prefix": HF_S3_PREFIX or None,
            "sample_id": sample_id,
            "evidence_id": evidence_id,
            "mask_status": review["mask_status"],
            "correction_status": review["correction_status"],
            "failure_stage": review.get("failure_stage"),
            "failure_type": review.get("failure_type"),
            "operational_output_stage": review.get(
                "operational_output_stage"
            ),
            "a5_n88_applied_to_operational_output": bool(
                review.get("a5_n88_applied_to_operational_output", False)
            ),
            "detector_training": detector_receipt,
            "correction_evidence": correction_receipt,
        }

        write_json(
            directory / "training_evidence_upload.json",
            receipt,
        )
        return receipt

    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
