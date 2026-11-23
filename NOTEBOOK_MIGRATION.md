# Notebook migration guide

## Colab import cell
Place `music_annotation_lib` under the project repository root, then use:

```python
import sys
from pathlib import Path
REPO = Path('/content/MusicAnnotationProject')
sys.path.insert(0, str(REPO/'music_annotation_lib'))
```

## Notebook 09, compact state-preserving reviewer
Replace the complete **Canonical V7.8 structural functions** cell with:

```python
from music_annotation.constants import *
from music_annotation.hashing import sha256_file, md5_file
from music_annotation.v78_engine import (
    immutable, binary, prepare_mask, padded_components, telea_reconstruct,
    background_surface, outside_dark, support_maps, geometry_t80,
    surrounding_features,
)
from music_annotation.review_state import array_signature as array_sig, component_signature
```

Keep notebook orchestration, path configuration, policy-bundle loading, regression reports, HITL audit resolution and Gradio UI in the notebook. Delete the duplicate `array_sig` and `component_signature` definitions. Do not change `PROTECTED_PREFIX_SHA256` until the revised notebook prefix is deliberately re-hashed.

## Notebook 28, V8.2 development whole-page experiment
Replace duplicated constants/helpers with imports from `constants`, `hashing`, `v78_engine`, `observed_print`, `policy` and `metrics`.

```python
from music_annotation.constants import CHANNELS, PATCH, HALF, DARK_THRESHOLD, SEED
from music_annotation.hashing import sha256_file, object_hash, array_hash
from music_annotation.v78_engine import background_surface
from music_annotation.observed_print import (
    printed_ink, truth_maps, build_padded_channels, get_patch,
    ArrayPatchDataset, TopologyCNN, make_loader,
)
from music_annotation.policy import provisional_decision, apply_run_filter, assemble_page
from music_annotation.metrics import wilson_upper, page_clustered_bootstrap, catastrophic
```

Keep `fit_fold` and `predict` in Notebook 28 for now because they bind notebook-specific `BATCH`, `EPOCHS`, `LR`, `WEIGHT_DECAY`, `DEVICE` and persisted output paths. Replace the decision block with `provisional_decision` plus `apply_run_filter`; pass runtime page shapes to `shape_by_page` to avoid relying on candidate-coordinate maxima.

## Notebook 29, V8.2a policy sensitivity
Replace `effective_print_threshold`, `provisional_decision`, `apply_run_filter`, and `wilson_upper` with:

```python
from music_annotation.policy import effective_print_threshold, provisional_decision, apply_run_filter, assemble_page
from music_annotation.metrics import wilson_upper, page_clustered_bootstrap, catastrophic
from music_annotation.hashing import sha256_file
```

Keep `summarise_policy` in the notebook initially because it writes V8.2a-specific tables and embeds the exact retrospective policy audit schema. Supply exact page shapes to `apply_run_filter` from the authoritative V7.5 mask maps.

## Notebook 30, V8.2b spatial probability-band audit
Use:

```python
from music_annotation.hashing import sha256_file
from music_annotation.policy import apply_run_filter, assemble_page
from music_annotation.metrics import catastrophic
```

Keep band definitions, route-context enrichment, heatmaps, overlays and contact sheets in Notebook 30. They are experiment-specific diagnostics rather than canonical inference behaviour. Its `connected_run_features` can later be merged into the package only after a regression test proves exact identity with the existing page-global logic.

## Behaviour-preservation rules
1. Keep all existing notebook constants unchanged.
2. Do not replace canonical V7.8 functions with “cleaned up” equivalents.
3. Run the V7.8.2 development equivalence audit before using the revised Notebook 09.
4. For Notebook 28, compare saved raw/calibrated OOF arrays, final decisions and whole-page hashes between old and imported versions using the same run inputs and environment.
5. For Notebooks 29 and 30, compare each generated CSV and page-output integrity table before deleting the old inline functions permanently.
6. Do not mix package extraction with threshold, path, cohort, model, calibration or UI changes.
