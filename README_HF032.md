# HF-03.2 corrected startup gate

This package contains the corrected HF-03.2 runtime files. The startup gate accepts the preserved historical replay verification record, verifies every recorded replay check, verifies frozen object hashes and policy identity, and performs a schema-valid live model/calibrator execution under the pinned runtime.

The smoke execution is explicitly not described as a replay. It exists to detect deserialisation/runtime failures such as the HF-03.1 `SimpleImputer._fill_dtype` failure.

Place the unchanged HF-03.1 application files alongside these files. Required unchanged application modules include `app/main.py`, `app/authorise.py`, `app/store.py`, `app/yolo.py`, `app/v78.py`, `app/policy.py`, `app/static/`, and `app/templates/`.
