from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _version(dist: str):
    try:
        return md.version(dist)
    except Exception:
        return None


def runtime_identities() -> dict[str, Any]:
    out = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scikit_learn": _version("scikit-learn"),
        "joblib": _version("joblib"),
        "opencv_python_headless": _version("opencv-python-headless"),
        "pillow": _version("Pillow"),
        "torch": _version("torch"),
        "torchvision": _version("torchvision"),
        "ultralytics": _version("ultralytics"),
        "fastapi": _version("fastapi"),
        "uvicorn": _version("uvicorn"),
    }
    try:
        import cv2
        out["opencv_runtime"] = cv2.__version__
    except Exception:
        out["opencv_runtime"] = None
    try:
        import torch
        out["torch_runtime"] = torch.__version__
        out["cuda_runtime"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["torch_runtime"] = None
        out["cuda_runtime"] = None
        out["cuda_available"] = False
        out["gpu"] = None
    return out


def write_commissioning_evidence(
    startup_root: Path,
    a5_replay_result: dict,
    v78_summary: dict,
) -> dict:
    """Persist one commissioning pack proving both HF-04.4 startup gates."""
    startup_root = Path(startup_root)
    replay_dir = startup_root / "ardross_a5_replay"
    equiv_dir = startup_root / "ardross_v78_equivalence"
    replay_dir.mkdir(parents=True, exist_ok=True)
    equiv_dir.mkdir(parents=True, exist_ok=True)

    if a5_replay_result.get("passed") is not True:
        raise RuntimeError("Refusing to write passing commissioning verdict: A5 replay did not pass")
    if v78_summary.get("passed") is not True:
        raise RuntimeError("Refusing to write passing commissioning verdict: V7.8 equivalence did not pass")

    replay_path = replay_dir / "replay_result.json"
    replay_path.write_text(json.dumps(a5_replay_result, indent=2, sort_keys=True, default=str))

    runtime = runtime_identities()
    runtime_path = startup_root / "runtime_software_identities.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True, default=str))

    verdict = {
        "version": "HF-04.4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "gates": {
            "a5_frozen_replay": {
                "passed": True,
                "name": a5_replay_result.get("name", "ardross_a5_startup_replay"),
                "rows": a5_replay_result.get("rows"),
                "probability_atol": a5_replay_result.get("probability_atol"),
                "probability_exact": a5_replay_result.get("probability_exact"),
                "probability_within_atol": a5_replay_result.get("probability_within_atol"),
                "probability_max_abs_error": a5_replay_result.get("probability_max_abs_error"),
                "provisional_exact": a5_replay_result.get("provisional_exact"),
                "provisional_mismatches": a5_replay_result.get("provisional_mismatches", 0),
                "final_exact": a5_replay_result.get("final_exact"),
                "final_mismatches": a5_replay_result.get("final_mismatches", 0),
                "evidence": "ardross_a5_replay/replay_result.json",
            },
            "v78_11_channel_equivalence": {
                "passed": True,
                "name": v78_summary.get("name", "ardross_v78_11_channel_startup_equivalence"),
                "channels": v78_summary.get("channels"),
                "exact_channels": v78_summary.get("exact_channels"),
                "mismatched_channels": v78_summary.get("mismatched_channels", []),
                "evidence": [
                    "ardross_v78_equivalence/summary.json",
                    "ardross_v78_equivalence/channel_comparison.csv",
                ],
            },
        },
        "runtime_evidence": "runtime_software_identities.json",
    }
    verdict_path = startup_root / "startup_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    return verdict
