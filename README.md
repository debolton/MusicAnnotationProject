# Music Annotation shared Python library

This extraction centralises the canonical V7.8 correction engine, V8.2 tensor/CNN definitions, policy helpers, metrics, hashing and Pilot 09 component-signature logic.

It intentionally leaves notebook-specific orchestration, Drive paths, experiment outputs, Gradio wiring and diagnostic visualisation in the notebooks.

## Local smoke test
```bash
python -m pytest -q
```

## Important
“Exact behaviour” is protected by source-faithful functions plus the existing project equivalence/replay contracts. The package has been syntax-checked and smoke-tested, but exact equality against your Google Drive arrays must still be established by running the project’s V7.8.2 equivalence audit and old-versus-imported notebook replay in Colab.
