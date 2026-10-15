# modules/alignment/alignment_state.py

from pathlib import Path
from datetime import datetime, timezone

from modules.alignment.alignment_utils import (
    md5_file,
    read_json,
    write_json_atomic,
    remove_if_exists,
)


class AlignmentStateManager:

    def __init__(
        self,
        state_path,
        settings_hash,
        force_rebuild_all=False,
        remove_outputs_for_deleted=False,
    ):

        self.state_path = Path(
            state_path
        )

        self.settings_hash = (
            settings_hash
        )

        self.force_rebuild_all = (
            force_rebuild_all
        )

        self.remove_outputs_for_deleted = (
            remove_outputs_for_deleted
        )

        self.state = None

    # --------------------------------------------------
    # State I/O
    # --------------------------------------------------

    def load(self):

        self.state = read_json(

            self.state_path,

            {
                "version": 1,
                "pages": {},
            },

        )

        return self.state

    def save(
        self,
        page_state,
    ):

        payload = {

            "version": 2,

            "updated_utc":

                datetime.now(
                    timezone.utc
                ).isoformat(),

            "settings_hash":
                self.settings_hash,

            "pages":
                page_state,
        }

        write_json_atomic(
            self.state_path,
            payload,
        )

    # --------------------------------------------------
    # Fingerprinting
    # --------------------------------------------------

    def fingerprint_page(
        self,
        clean_path,
        annotated_path,
    ):

        return {

            "clean_path":
                str(clean_path),

            "annotated_path":
                str(annotated_path),

            "clean_md5":
                md5_file(
                    clean_path
                ),

            "annotated_md5":
                md5_file(
                    annotated_path
                ),

            "settings_hash":
                self.settings_hash,
        }

    # --------------------------------------------------
    # Expected outputs
    # --------------------------------------------------

    @staticmethod
    def outputs_exist(
        expected_outputs
    ):

        return all(

            Path(path).exists()

            for path

            in expected_outputs

        )

    # --------------------------------------------------
    # Rebuild determination
    # --------------------------------------------------

    def is_current(
        self,
        page,
        fingerprint,
        expected_outputs,
    ):

        if self.force_rebuild_all:

            return False

        previous = (

            self.state
            .get("pages", {})
            .get(page, {})

        )

        if (

            previous.get(
                "fingerprint"
            )

            != fingerprint

        ):

            return False

        if not self.outputs_exist(
            expected_outputs
        ):

            return False

        return True

    # --------------------------------------------------
    # Page analysis
    # --------------------------------------------------

    def determine_pages(
        self,
        matched_pages,
        clean_lookup,
        annotated_lookup,
        output_lookup,
    ):

        current_state = {}

        to_process = []

        skipped = []

        for page in matched_pages:

            clean_path = clean_lookup[
                page
            ]

            annotated_path = (
                annotated_lookup[
                    page
                ]
            )

            fingerprint = (
                self.fingerprint_page(
                    clean_path,
                    annotated_path,
                )
            )

            current = (
                self.is_current(
                    page,
                    fingerprint,
                    output_lookup[
                        page
                    ],
                )
            )

            current_state[
                page
            ] = {

                "fingerprint":
                    fingerprint,

                "status":
                    (
                        "current"
                        if current
                        else "pending"
                    ),
            }

            if current:

                skipped.append(
                    page
                )

            else:

                to_process.append(
                    page
                )

        return {

            "current_state":
                current_state,

            "to_process":
                to_process,

            "skipped":
                skipped,
        }

    # --------------------------------------------------
    # Deleted source cleanup
    # --------------------------------------------------

    def cleanup_deleted_pages(
        self,
        matched_pages,
        output_builder,
    ):

        if (
            not
            self.remove_outputs_for_deleted
        ):
            return

        old_pages = set(

            self.state
            .get("pages", {})
            .keys()

        )

        current_pages = set(
            matched_pages
        )

        removed_pages = (
            old_pages
            - current_pages
        )

        for page in sorted(
            removed_pages
        ):

            outputs = output_builder(
                page
            )

            for path in outputs:

                remove_if_exists(
                    path
                )

    # --------------------------------------------------
    # Successful page update
    # --------------------------------------------------

    @staticmethod
    def update_success(
        current_state,
        page,
        diagnostics,
        outputs,
    ):

        current_state[
            page
        ].update({

            "status":
                "current",

            "processed_utc":

                datetime.now(
                    timezone.utc
                ).isoformat(),

            "diagnostics":
                diagnostics,

            "outputs": {

                key: str(value)

                for key, value

                in outputs.items()

            },
        })

    # --------------------------------------------------
    # Failed page update
    # --------------------------------------------------

    @staticmethod
    def update_failure(
        current_state,
        page,
        error,
    ):

        current_state[
            page
        ].update({

            "status":
                "failed",

            "error":
                str(error),
        })
