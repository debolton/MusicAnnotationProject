# modules/pilot/hitl_resolution.py

import json
import hashlib
import cv2
import numpy as np

from pathlib import Path


class HITLResolver:

    def __init__(
        self,
        review_audit_dir,
        draft_audit_dir,
    ):

        self.review_audit_dir = Path(
            review_audit_dir
        )

        self.draft_audit_dir = Path(
            draft_audit_dir
        )

        self.review_audits = []
        self.draft_audits = []

        self.load_audits()

    # --------------------------------------------------
    # Hashing
    # --------------------------------------------------

    @staticmethod
    def md5_file(path):

        h = hashlib.md5()

        with open(path, "rb") as f:

            for block in iter(
                lambda: f.read(1048576),
                b""
            ):
                h.update(block)

        return h.hexdigest()

    @staticmethod
    def sha256_file(path):

        h = hashlib.sha256()

        with open(path, "rb") as f:

            for block in iter(
                lambda: f.read(1048576),
                b""
            ):
                h.update(block)

        return h.hexdigest()

    # --------------------------------------------------
    # Audit loading
    # --------------------------------------------------

    def read_audit_records(
        self,
        folder,
        label,
    ):

        records = []
        failures = []

        for path in sorted(
            Path(folder).glob("*.json")
        ):

            try:

                value = json.loads(
                    path.read_text()
                )

                if not isinstance(
                    value,
                    dict
                ):

                    raise ValueError(
                        "audit root "
                        "must be object"
                    )

                value[
                    "_audit_path"
                ] = str(path)

                records.append(
                    value
                )

            except Exception as ex:

                failures.append(

                    {
                        "audit_type":
                            label,

                        "audit_path":
                            str(path),

                        "error":
                            str(ex),
                    }

                )

        return records, failures

    def load_audits(self):

        (
            self.review_audits,
            review_failures,
        ) = self.read_audit_records(
            self.review_audit_dir,
            "review",
        )

        (
            self.draft_audits,
            draft_failures,
        ) = self.read_audit_records(
            self.draft_audit_dir,
            "draft",
        )

        failures = (
            review_failures
            + draft_failures
        )

        if failures:

            raise RuntimeError(
                f"Unreadable audit "
                f"records: "
                f"{len(failures)}"
            )

    # --------------------------------------------------
    # Audit lookup helpers
    # --------------------------------------------------

    @staticmethod
    def matches_by_hash(
        records,
        source_md5,
    ):

        return [

            record

            for record in records

            if str(
                record.get(
                    "source_md5",
                    ""
                )
            ).lower()

            ==

            source_md5.lower()

        ]

    @staticmethod
    def required_path(
        record,
        key,
        audit_type,
    ):

        raw = record.get(key)

        if not raw:

            raise RuntimeError(
                f"{audit_type} audit "
                f"{record.get('_audit_path')} "
                f"is stale: missing {key}"
            )

        path = Path(raw)

        if not path.exists():

            raise RuntimeError(
                f"{audit_type} audit "
                f"{record.get('_audit_path')} "
                f"references missing "
                f"file: {path}"
            )

        return path

    # --------------------------------------------------
    # Main resolution
    # --------------------------------------------------

    def resolve(
        self,
        source_path,
    ):

        source_path = Path(
            source_path
        )

        source_md5 = self.md5_file(
            source_path
        )

        review_matches = (
            self.matches_by_hash(
                self.review_audits,
                source_md5,
            )
        )

        draft_matches = (
            self.matches_by_hash(
                self.draft_audits,
                source_md5,
            )
        )

        if len(review_matches) > 1:

            raise RuntimeError(

                f"{source_path.name}: "
                "duplicate review "
                "audits detected"

            )

        if len(draft_matches) > 1:

            raise RuntimeError(

                f"{source_path.name}: "
                "duplicate draft "
                "audits detected"

            )

        review = (

            review_matches[0]

            if review_matches

            else None

        )

        draft = (

            draft_matches[0]

            if draft_matches

            else None

        )

        if review is None and draft is None:

            raise RuntimeError(

                f"{source_path.name}: "
                "no matching HITL "
                "audit found"

            )

        confidence_path = None
        draft_mask_path = None
        draft_audit_path = None

        #
        # Draft lookup
        #

        if draft is not None:

            draft_audit_path = (
                draft[
                    "_audit_path"
                ]
            )

            draft_mask_path = (
                self.required_path(
                    draft,
                    "draft_mask",
                    "draft",
                )
            )

            confidence_path = (
                self.required_path(
                    draft,
                    "confidence_map",
                    "draft",
                )
            )

            draft_source = (
                draft.get(
                    "source_file"
                )
            )

            if (

                draft_source

                and

                Path(
                    draft_source
                ).exists()

            ):

                actual = (
                    self.md5_file(
                        Path(
                            draft_source
                        )
                    )
                )

                if actual != source_md5:

                    raise RuntimeError(

                        f"{source_path.name}: "
                        "draft audit "
                        "source hash stale"

                    )

        #
        # Default resolution
        #

        observed_path = (
            source_path
        )

        mask_path = (
            draft_mask_path
        )

        mask_status = (
            "unreviewed_draft"
        )

        review_audit_path = None

        #
        # Reviewed override
        #

        if review is not None:

            review_audit_path = (
                review[
                    "_audit_path"
                ]
            )

            observed_path = (
                self.required_path(
                    review,
                    "aligned_output",
                    "review",
                )
            )

            mask_path = (
                self.required_path(
                    review,
                    "corrected_mask_output",
                    "review",
                )
            )

            mask_status = (
                "reviewed_corrected"
            )

            review_source = (
                review.get(
                    "source"
                )
            )

            if (
                review_source
                and
                Path(
                    review_source
                ).exists()
            ):

                actual = (
                    self.md5_file(
                        Path(
                            review_source
                        )
                    )
                )

                if actual != source_md5:

                    raise RuntimeError(

                        f"{source_path.name}: "
                        "review audit "
                        "source hash stale"

                    )

        #
        # Read images
        #

        observed = cv2.imread(
            str(observed_path),
            0,
        )

        mask = cv2.imread(
            str(mask_path),
            0,
        )

        if observed is None:

            raise RuntimeError(
                f"{source_path.name}: "
                "observed page "
                "unreadable"
            )

        if mask is None:

            raise RuntimeError(
                f"{source_path.name}: "
                "mask unreadable"
            )

        if (
            observed.shape
            != mask.shape
        ):

            raise RuntimeError(

                f"{source_path.name}: "
                "observed/mask "
                "shape mismatch"

            )

        #
        # Confidence
        #

        if confidence_path:

            confidence = np.load(
                confidence_path
            ).astype(np.float32)

            if (
                confidence.shape
                != observed.shape
            ):

                raise RuntimeError(

                    f"{source_path.name}: "
                    "confidence map "
                    "shape mismatch"

                )

        else:

            confidence = (
                (mask > 0)
                .astype(np.float32)
                * 0.20
            )

        #
        # Return canonical structure
        #

        return {

            "source_original":
                source_path,

            "source_md5":
                source_md5,

            "observed_path":
                observed_path,

            "mask_path":
                mask_path,

            "confidence_path":
                confidence_path,

            "review_audit_path":
                review_audit_path,

            "draft_audit_path":
                draft_audit_path,

            "mask_status":
                mask_status,

            "observed":
                observed,

            "seed":
                mask,

            "confidence":
                confidence,

            "observed_sha256":
                self.sha256_file(
                    observed_path
                ),

            "authoritative_mask_sha256":
                self.sha256_file(
                    mask_path
                ),

            "confidence_sha256":
                (
                    self.sha256_file(
                        confidence_path
                    )
                    if confidence_path
                    else None
                ),
        }
