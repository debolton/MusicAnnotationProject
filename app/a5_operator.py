from pathlib import Path
import importlib.util
import numpy as np

from .canonical_v78 import build_v78

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "frozen" / "a5_v1"


class A5Operator:
    def __init__(self, device=None):
        runtime_path = BUNDLE / "inference_runtime.py"
        spec = importlib.util.spec_from_file_location("a5_frozen_runtime", runtime_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load frozen A5 runtime: {runtime_path}")
        self.rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.rt)

        required = ["load_bundle", "make_padded", "score_coordinates", "decisions", "assemble"]
        missing = [name for name in required if not callable(getattr(self.rt, name, None))]
        if missing:
            raise RuntimeError(f"A5 frozen runtime API incomplete: {missing}")

        self.model, self.calibrator, self.policy, self.schema, self.device = self.rt.load_bundle(
            BUNDLE, device=device
        )
        self._assert_contract()

        from .hf043_replay import run_ardross_replay
        self.startup_replay = run_ardross_replay(self)

    def _assert_contract(self):
        expected_channels = [
            "obs", "norm", "mask", "depth", "horizontal", "vertical",
            "compact", "preserve", "remove", "ambiguous", "edge",
        ]
        if self.schema.get("channels") != expected_channels:
            raise RuntimeError("A5 channel order differs from the frozen 11-channel contract")
        if int(self.schema.get("patch", -1)) != 33:
            raise RuntimeError("A5 patch size differs from the frozen 33x33 contract")
        expected_policy = {
            "print_threshold": 0.980,
            "print_margin": 0.960,
            "annotation_threshold": 0.995,
            "minimum_print_run": 2,
        }
        for key, expected in expected_policy.items():
            actual = self.policy.get(key)
            if float(actual) != float(expected):
                raise RuntimeError(f"A5 policy mismatch for {key}: {actual} != {expected}")
        truth_codes = self.policy.get("truth_codes", {})
        if truth_codes and truth_codes != {"0": "ANNOTATION", "1": "PRINT", "2": "UNCERTAIN"}:
            raise RuntimeError(f"Unexpected A5 class contract: {truth_codes}")

    def run(self, observed, reviewed_seed, page=None, batch=512):
        v = build_v78(observed, reviewed_seed)
        result = self.process_arrays(v["channels"], v["a1"], authoritative_mask=v["mask"], batch=batch)
        return {
            **v,
            "a5": result["corrected"],
            "p_print": result["probability"],
            "provisional": result["provisional"],
            "final": result["final"],
            "print_mask": result["print_mask"],
        }

    def process_arrays(self, arrays, a1, authoritative_mask=None, batch=512):
        required = set(self.schema["channels"])
        missing = sorted(required - set(arrays))
        if missing:
            raise RuntimeError(f"A5 arrays missing channels: {missing}")

        observed = np.asarray(arrays["obs"], dtype=np.uint8)
        a1 = np.asarray(a1, dtype=np.uint8)
        mask = np.asarray(authoritative_mask if authoritative_mask is not None else arrays["mask"]) > 0
        preserve = np.asarray(arrays["preserve"]) > 0
        if not (observed.shape == a1.shape == mask.shape == preserve.shape):
            raise RuntimeError("Observed/A1/mask/preserve shape mismatch")

        eligible = mask & (observed < 200) & ~preserve
        y, x = np.where(eligible)
        y = y.astype(np.int64)
        x = x.astype(np.int64)

        padded = self.rt.make_padded(arrays, self.schema)
        probability = self.rt.score_coordinates(
            self.model, self.calibrator, padded, self.schema,
            y, x, self.device, batch=batch,
        )
        provisional, final = self.rt.decisions(
            probability, self.policy, a1.shape, y, x
        )
        corrected, print_mask = self.rt.assemble(
            observed, a1, mask, y, x, final
        )

        if np.any(corrected[~mask] != observed[~mask]):
            raise RuntimeError("A5 outside-mask drift")
        if np.any(corrected[~print_mask] != a1[~print_mask]):
            raise RuntimeError("A5 non-PRINT A1 regression")
        if np.any((corrected != a1) & (corrected != observed)):
            raise RuntimeError("A5 changed byte not equal to observed")

        return {
            "eligible": eligible,
            "probability": np.asarray(probability),
            "provisional": np.asarray(provisional),
            "final": np.asarray(final),
            "y": y,
            "x": x,
            "print_mask": np.asarray(print_mask, dtype=bool),
            "corrected": corrected,
        }
