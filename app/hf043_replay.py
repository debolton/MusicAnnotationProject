from __future__ import annotations
from pathlib import Path
import csv, hashlib, json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARDROSS = ROOT / 'frozen' / 'a5_v1' / 'replay' / 'ardross'
CHANNELS = ['obs','norm','mask','depth','horizontal','vertical','compact','preserve','remove','ambiguous','edge']

class ReplayFailure(RuntimeError): pass

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def _read_vector(path: Path, kind='float'):
    with path.open(newline='') as f:
        rows=list(csv.reader(f))
    vals=[]
    for row in rows:
        if not row: continue
        s=row[-1].strip()
        try: vals.append(float(s) if kind=='float' else int(float(s)))
        except ValueError: continue
    return np.asarray(vals, dtype=np.float64 if kind=='float' else np.int64)

def _read_string_vector(path: Path):

    with path.open(
        newline='',
        encoding='utf-8-sig',
    ) as f:
        rows = list(csv.reader(f))

    values = []

    VALID = {
        'PRINT',
        'ANNOTATION',
        'UNCERTAIN',
    }

    for row in rows:

        if not row:
            continue

        value = row[-1].strip().upper()

        # This automatically ignores the CSV header:
        # expected_provisional / expected_final
        if value in VALID:
            values.append(value)

    return np.asarray(
        values,
        dtype=object,
    )


def _read_image(name, flags=cv2.IMREAD_GRAYSCALE):
    p=ARDROSS/name
    a=cv2.imread(str(p),flags)
    if a is None: raise ReplayFailure(f'Unreadable replay image: {p}')
    return a

def _read_matrix(path: Path, dtype=np.float32):
    try:
        a = np.loadtxt(
            path,
            delimiter=',',
            dtype=dtype,
        )
    except Exception as e:
        raise ReplayFailure(
            f'Cannot read replay matrix {path}: {e}'
        )

    if a.ndim != 2:
        raise ReplayFailure(
            f'{path.name}: expected 2-D matrix, got shape {a.shape}'
        )

    if not np.isfinite(a).all():
        raise ReplayFailure(
            f'{path.name}: contains non-finite values'
        )

    return a

def load_replay():
    required=['obs.png','norm.png','mask.png','depth.csv','horizontal.png','vertical.png','compact.png','preserve.png','remove.png','ambiguous.png','edge.png','x.csv','y.csv','expected_probability.csv','expected_provisional.csv','expected_final.csv']
    missing=[x for x in required if not (ARDROSS/x).exists()]
    if missing: raise ReplayFailure('Ardross replay incomplete: '+', '.join(missing))
    arrays={k:_read_image(k+'.png') for k in ['obs','norm','mask','horizontal','vertical','compact','preserve','remove','ambiguous','edge']}
    h,w=arrays['obs'].shape
    for k,a in arrays.items():
        if a.shape!=(h,w): raise ReplayFailure(f'Ardross {k} shape {a.shape} != {(h,w)}')
    
    depth = _read_matrix(
        ARDROSS / 'depth.csv',
        np.float32,
    )

    if depth.shape != (h, w):
        raise ReplayFailure(
            'Ardross depth shape mismatch: '
            f'depth={depth.shape}, '
            f'obs={(h, w)}'
        )

    arrays['depth'] = depth
        
    arrays['depth']=depth.astype(np.float32)
    arrays['obs'] = arrays['obs'].astype(np.uint8)
    arrays['norm'] = arrays['norm'].astype(np.uint8)
    for k in ['mask','horizontal','vertical','compact','preserve','remove','ambiguous','edge']:
        arrays[k]=(arrays[k]>0)
    x=_read_vector(ARDROSS/'x.csv','int').astype(np.int64)
    y=_read_vector(ARDROSS/'y.csv','int').astype(np.int64)
    expected_p=_read_vector(ARDROSS/'expected_probability.csv').astype(np.float64)
    expected_prov=_read_string_vector(ARDROSS/'expected_provisional.csv')
    expected_final=_read_string_vector(ARDROSS/'expected_final.csv')
    
    n=len(x)
    if not all(len(v)==n for v in [y,expected_p,expected_prov,expected_final]): raise ReplayFailure('Ardross replay vectors have inconsistent row counts')
    if n!=5000: raise ReplayFailure(f'Ardross replay must contain 5000 coordinates, found {n}')
    if (x<0).any() or (x>=w).any() or (y<0).any() or (y>=h).any(): raise ReplayFailure('Ardross replay coordinate out of bounds')
    return arrays,y,x,expected_p,expected_prov,expected_final

def _canonical_labels(v):
    a=np.asarray(v)
    if a.dtype.kind in 'OUS':
        m={'ANNOTATION':0,'UNCERTAIN':1,'PRINT':2}
        return np.asarray([m.get(str(x).upper(),-99) for x in a],np.int64)
    return a.astype(np.int64)

def run_ardross_replay(
    operator,
    probability_atol=1e-7,
):
    """
    Mandatory frozen Ardross A5 replay.

    IMPORTANT:
    - arrays['obs'] and arrays['norm'] must enter this function
      in their ORIGINAL 0..255 representation.
    - Do NOT pre-normalise them.
    - frozen inference_runtime.make_padded() owns the /255 scaling.
    - expected probabilities remain float64.
    - expected provisional/final decisions remain semantic strings:
        PRINT / ANNOTATION / UNCERTAIN
    """

    # --------------------------------------------------------
    # 1. Load the frozen Colab replay inputs / expectations
    # --------------------------------------------------------

    (
        arrays,
        y,
        x,
        expected_p,
        expected_prov,
        expected_final,
    ) = load_replay()

    rt = operator.rt

    # --------------------------------------------------------
    # 2. Fail-closed channel/schema validation
    # --------------------------------------------------------

    required_channels = list(
        operator.schema["channels"]
    )

    missing = [
        name
        for name in required_channels
        if name not in arrays
    ]

    if missing:
        raise ReplayFailure(
            "Ardross replay missing frozen channels: "
            f"{missing}"
        )

    shape = arrays["obs"].shape

    for name in required_channels:

        a = np.asarray(arrays[name])

        if a.ndim != 2:
            raise ReplayFailure(
                f"Ardross {name}: expected 2-D array, "
                f"got {a.shape}"
            )

        if a.shape != shape:
            raise ReplayFailure(
                f"Ardross {name}: shape mismatch "
                f"{a.shape} != {shape}"
            )

        if not np.isfinite(a).all():
            raise ReplayFailure(
                f"Ardross {name}: non-finite values"
            )

    # --------------------------------------------------------
    # 3. Explicitly guard against the HF-04.3
    #    double-normalisation bug
    # --------------------------------------------------------

    obs = np.asarray(arrays["obs"])
    norm = np.asarray(arrays["norm"])

    # These should still be byte-scale channels here.
    #
    # inference_runtime.make_padded() contains:
    #
    #     v = a.astype(np.float32)
    #     if name in ('obs', 'norm'):
    #         v /= 255.0
    #
    # Therefore obs/norm MUST NOT already be scaled to 0..1.

    if obs.max() <= 1.0:
        raise ReplayFailure(
            "Ardross obs appears already normalised. "
            "Do not divide obs by 255 before make_padded()."
        )

    if norm.max() <= 1.0:
        raise ReplayFailure(
            "Ardross norm appears already normalised. "
            "Do not divide norm by 255 before make_padded()."
        )

        
    # --------------------------------------------------------
    # 4. Validate replay vectors
    # --------------------------------------------------------

    y = np.asarray(y, dtype=np.int64).reshape(-1)
    x = np.asarray(x, dtype=np.int64).reshape(-1)

    expected_p = np.asarray(
        expected_p,
        dtype=np.float64,
    ).reshape(-1)

    expected_prov = np.asarray(
        expected_prov,
        dtype=str,
    ).reshape(-1)

    expected_final = np.asarray(
        expected_final,
        dtype=str,
    ).reshape(-1)

    n = len(x)

    vector_lengths = {
        "x": len(x),
        "y": len(y),
        "expected_probability": len(expected_p),
        "expected_provisional": len(expected_prov),
        "expected_final": len(expected_final),
    }

    if len(set(vector_lengths.values())) != 1:
        raise ReplayFailure(
            "Ardross replay vectors have inconsistent "
            f"row counts: {vector_lengths}"
        )

    if n != 5000:
        raise ReplayFailure(
            "Ardross replay expected 5000 coordinates, "
            f"found {n}"
        )

    h, w = shape

    if (
        (x < 0).any()
        or (x >= w).any()
        or (y < 0).any()
        or (y >= h).any()
    ):
        raise ReplayFailure(
            "Ardross replay coordinates are out of bounds"
        )

    # --------------------------------------------------------
    # 5. Build tensors using the FROZEN Notebook-91 runtime
    #
    # DO NOT normalise obs/norm here.
    # make_padded() does it exactly once.
    # --------------------------------------------------------

    padded = rt.make_padded(
        arrays,
        operator.schema,
    )

    # --------------------------------------------------------
    # 6. Frozen CNN + frozen isotonic calibration
    # --------------------------------------------------------

    got_p = np.asarray(
        rt.score_coordinates(
            operator.model,
            operator.calibrator,
            padded,
            operator.schema,
            y,
            x,
            operator.device,
            batch=512,
        ),
        dtype=np.float64,
    )

    if got_p.shape != expected_p.shape:
        raise ReplayFailure(
            "Ardross probability-vector shape mismatch: "
            f"HF={got_p.shape}, "
            f"frozen={expected_p.shape}"
        )

    if not np.isfinite(got_p).all():
        raise ReplayFailure(
            "Ardross HF replay produced non-finite "
            "probabilities"
        )

    # --------------------------------------------------------
    # 7. Apply frozen A5 decisions exactly
    # --------------------------------------------------------

    got_prov, got_final = rt.decisions(
        got_p,
        operator.policy,
        shape,
        y,
        x,
    )

    got_prov = np.asarray(
        got_prov,
        dtype=str,
    ).reshape(-1)

    got_final = np.asarray(
        got_final,
        dtype=str,
    ).reshape(-1)

    # --------------------------------------------------------
    # 8. Probability comparison
    # --------------------------------------------------------

    probability_exact = bool(
        np.array_equal(
            got_p,
            expected_p,
        )
    )

    probability_max_abs_error = float(
        np.max(
            np.abs(
                got_p - expected_p
            )
        )
    )

    probability_within_atol = bool(
        np.allclose(
            got_p,
            expected_p,
            rtol=0.0,
            atol=probability_atol,
            equal_nan=False,
        )
    )

    # --------------------------------------------------------
    # 9. Semantic decision comparisons
    #
    # No integer class remapping.
    # Compare the actual frozen semantic classes.
    # --------------------------------------------------------

    provisional_exact = bool(
        np.array_equal(
            got_prov,
            expected_prov,
        )
    )

    final_exact = bool(
        np.array_equal(
            got_final,
            expected_final,
        )
    )

    provisional_mismatches = int(
        np.sum(
            got_prov != expected_prov
        )
    )

    final_mismatches = int(
        np.sum(
            got_final != expected_final
        )
    )

    # --------------------------------------------------------
    # 10. Useful diagnostics
    # --------------------------------------------------------

    result = {
        "name": "ardross_a5_startup_replay",

        "rows": int(n),

        "probability_exact":
            probability_exact,

        "probability_atol":
            float(probability_atol),

        "probability_within_atol":
            probability_within_atol,

        "probability_max_abs_error":
            probability_max_abs_error,

        "provisional_exact":
            provisional_exact,

        "provisional_mismatches":
            provisional_mismatches,

        "final_exact":
            final_exact,

        "final_mismatches":
            final_mismatches,

        "expected_probability_min":
            float(expected_p.min()),

        "expected_probability_max":
            float(expected_p.max()),

        "hf_probability_min":
            float(got_p.min()),

        "hf_probability_max":
            float(got_p.max()),

        "obs_input_min":
            float(obs.min()),

        "obs_input_max":
            float(obs.max()),

        "norm_input_min":
            float(norm.min()),

        "norm_input_max":
            float(norm.max()),
    }

    # --------------------------------------------------------
    # 11. Mandatory fail-closed verdict
    #
    # Decisions must match exactly.
    # Probabilities use only the explicitly declared tolerance.
    # --------------------------------------------------------

    result["passed"] = bool(
        probability_within_atol
        and provisional_exact
        and final_exact
    )

    # --------------------------------------------------------
    # 12. Print diagnostics BEFORE raising
    # --------------------------------------------------------

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    if not result["passed"]:
        raise ReplayFailure(
            "Mandatory Ardross A5 replay failed: "
            + json.dumps(
                result,
               sort_keys=True,
            )
        )

    return result
