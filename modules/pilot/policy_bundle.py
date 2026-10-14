# modules/pilot/policy_bundle.py

import json
import hashlib
import joblib

from pathlib import Path


class PolicyBundle:

    REQUIRED_FILES = [

        "artefacts/policy.json",

        "artefacts/predictor_schema.json",

        "artefacts/structural_model.joblib",

        "artefacts/isotonic_calibrator.joblib",

        "artefacts/replay_test.json",
    ]

    def __init__(
        self,
        project_root,
        policy_run_override=None,
    ):

        self.project_root = Path(
            project_root
        )

        self.policy_root = (

            self.project_root
            / "structural_policy_freeze_v7_7"

        )

        self.policy_run_override = (
            Path(policy_run_override)
            if policy_run_override
            else None
        )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

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

    @classmethod
    def latest_complete(
        cls,
        root,
        required,
    ):

        root = Path(root)

        candidates = []

        if root.exists():

            for path in root.rglob("*"):

                if not path.is_dir():
                    continue

                complete = all(

                    (
                        path / item
                    ).exists()

                    for item

                    in required

                )

                if complete:
                    candidates.append(path)

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda p: p.stat().st_mtime
        )

    # --------------------------------------------------
    # Locate run
    # --------------------------------------------------

    def find_policy_run(self):

        if self.policy_run_override:

            return self.policy_run_override

        return self.latest_complete(
            self.policy_root,
            self.REQUIRED_FILES,
        )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_hashes(self):

        checks = {

            "replay":

                self.replay.get(
                    "passed"
                ) is True

                and

                self.replay.get(
                    "policy_id"
                )

                ==

                self.policy.get(
                    "policy_id"
                ),

            "model_hash":

                self.policy[
                    "model_sha256"
                ]

                ==

                self.sha256_file(
                    self.model_path
                ),

            "calibrator_hash":

                self.policy[
                    "calibrator_sha256"
                ]

                ==

                self.sha256_file(
                    self.calibrator_path
                ),

            "schema_hash":

                self.policy[
                    "schema_sha256"
                ]

                ==

                self.sha256_file(
                    self.schema_path
                ),
        }

        if not all(checks.values()):

            raise RuntimeError(

                "Frozen policy integrity "
                f"failure: {checks}"

            )

        return checks

    # --------------------------------------------------
    # Load bundle
    # --------------------------------------------------

    def load(self):

        self.run_path = (
            self.find_policy_run()
        )

        if self.run_path is None:

            raise RuntimeError(
                "No complete V7.7F "
                "policy bundle found."
            )

        artefacts = (
            self.run_path /
            "artefacts"
        )

        self.policy_path = (
            artefacts /
            "policy.json"
        )

        self.schema_path = (
            artefacts /
            "predictor_schema.json"
        )

        self.model_path = (
            artefacts /
            "structural_model.joblib"
        )

        self.calibrator_path = (
            artefacts /
            "isotonic_calibrator.joblib"
        )

        self.replay_path = (
            artefacts /
            "replay_test.json"
        )

        self.policy = json.loads(
            self.policy_path.read_text()
        )

        self.schema = json.loads(
            self.schema_path.read_text()
        )

        self.replay = json.loads(
            self.replay_path.read_text()
        )

        self.hash_checks = (
            self.validate_hashes()
        )

        self.model = joblib.load(
            self.model_path
        )

        self.calibrator = joblib.load(
            self.calibrator_path
        )

        self.predictors = (
            self.schema[
                "ordered_predictors"
            ]
        )

        self.threshold = float(
            self.policy[
                "threshold"
            ]
        )

        return {

            "run_path":
                self.run_path,

            "policy":
                self.policy,

            "schema":
                self.schema,

            "predictors":
                self.predictors,

            "threshold":
                self.threshold,

            "model":
                self.model,

            "calibrator":
                self.calibrator,

            "hash_checks":
                self.hash_checks,
        }

    # --------------------------------------------------
    # Convenience properties
    # --------------------------------------------------

    @property
    def policy_id(self):

        return self.policy[
            "policy_id"
        ]

    @property
    def predictor_count(self):

        return len(
            self.predictors
        )

    def summary(self):

        return {

            "policy_id":
                self.policy_id,

            "threshold":
                self.threshold,

            "predictor_count":
                self.predictor_count,

            "run_path":
                str(self.run_path),

            "hash_checks":
                self.hash_checks,
        }
