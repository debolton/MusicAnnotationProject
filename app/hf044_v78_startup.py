from __future__ import annotations
from pathlib import Path
import csv, hashlib, json
import cv2
import numpy as np
from .canonical_v78 import telea_reconstruct, geometry_t80
from .hf043_replay import ARDROSS, CHANNELS, ReplayFailure, _read_matrix

ROOT=Path(__file__).resolve().parents[1]
OUTPUT_DIR=ROOT/'startup_evidence'/'ardross_v78_equivalence'
BINARY_CHANNELS={'mask','horizontal','vertical','compact','preserve','remove','ambiguous','edge'}

def _array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def _read_png(name):
    p=ARDROSS/f'{name}.png'
    a=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ReplayFailure(f'Unreadable Ardross V7.8 reference: {p}')
    return a

def _load_reference_channels():
    reference={k:_read_png(k) for k in CHANNELS if k!='depth'}
    reference['depth']=_read_matrix(ARDROSS/'depth.csv',np.float32)
    shape=reference['obs'].shape
    for k,a in reference.items():
        if a.shape!=shape:
            raise ReplayFailure(f'Ardross reference {k} shape {a.shape} != {shape}')
        if not np.isfinite(a).all():
            raise ReplayFailure(f'Ardross reference {k} contains non-finite values')
    return reference

def _rebuild_channels(reference):
    observed=reference['obs'].astype(np.uint8)
    prepared_mask=np.where(reference['mask']>0,255,0).astype(np.uint8)
    telea=telea_reconstruct(observed,prepared_mask)
    _,routes=geometry_t80(observed,prepared_mask,telea)
    rebuilt={
        'obs':observed,
        'norm':routes['normalised'].astype(np.uint8),
        'mask':prepared_mask>0,
        'depth':cv2.distanceTransform((prepared_mask>0).astype(np.uint8),cv2.DIST_L2,5).astype(np.float32),
        'horizontal':routes['horizontal'],
        'vertical':routes['vertical'],
        'compact':routes['compact'],
        'preserve':routes['preserve'],
        'remove':routes['remove'],
        'ambiguous':routes['ambiguous'],
        'edge':cv2.Canny(observed,70,170)>0,
    }
    return rebuilt

def _canonical(name,a):
    a=np.asarray(a)
    if name in BINARY_CHANNELS:
        return (a>0).astype(np.uint8)
    if name in {'obs','norm'}:
        return a.astype(np.uint8)
    if name=='depth':
        return a.astype(np.float32)
    raise ReplayFailure(f'Unknown channel: {name}')

def run_ardross_v78_equivalence(out_dir: Path|None=None):
    out=Path(out_dir or OUTPUT_DIR)
    out.mkdir(parents=True,exist_ok=True)
    reference=_load_reference_channels()
    rebuilt=_rebuild_channels(reference)
    rows=[]
    for name in CHANNELS:
        ref=_canonical(name,reference[name]); got=_canonical(name,rebuilt[name])
        shape_ok=ref.shape==got.shape
        if shape_ok:
            diff=np.abs(got.astype(np.float64)-ref.astype(np.float64))
            different=int(np.count_nonzero(diff))
            exact=bool(np.array_equal(got,ref))
            max_abs=float(diff.max(initial=0))
            mean_abs=float(diff.mean())
            if different:
                scale=diff
                if name=='depth' and max_abs>0:
                    scale=diff*(255.0/max_abs)
                cv2.imwrite(str(out/f'{name}_absdiff.png'),np.clip(scale,0,255).astype(np.uint8))
        else:
            different=-1; exact=False; max_abs=float('nan'); mean_abs=float('nan')
        rows.append({
            'channel':name,'shape_reference':str(ref.shape),'shape_hf':str(got.shape),
            'shape_ok':shape_ok,'exact':exact,'different_pixels':different,
            'max_abs_difference':max_abs,'mean_abs_difference':mean_abs,
            'reference_sha256':_array_hash(ref),'hf_sha256':_array_hash(got),
        })

    csv_path=out/'channel_comparison.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary={
        'name':'ardross_v78_11_channel_startup_equivalence',
        'channels':len(rows),'exact_channels':int(sum(r['exact'] for r in rows)),
        'mismatched_channels':[r['channel'] for r in rows if not r['exact']],
        'channel_comparison_csv':str(csv_path),'passed':bool(all(r['exact'] for r in rows)),
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    if not summary['passed']:
        raise ReplayFailure('Mandatory Ardross V7.8 channel equivalence failed: '+json.dumps(summary,sort_keys=True))
    return summary
