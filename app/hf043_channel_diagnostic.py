from __future__ import annotations
from pathlib import Path
import csv, hashlib, json
import cv2
import numpy as np
from .canonical_v78 import build_v78
from .hf043_replay import ARDROSS, CHANNELS, ReplayFailure, _read_vector

ROOT=Path(__file__).resolve().parents[1]

def array_digest(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes()); return h.hexdigest()

def _historical_channels():
    hist={}
    for k in CHANNELS:
        if k=='depth':
            obs=cv2.imread(str(ARDROSS/'obs.png'),0)
            v=_read_vector(ARDROSS/'depth.csv')
            if obs is None or len(v)!=obs.size: raise ReplayFailure('Historical depth.csv does not match obs.png')
            hist[k]=v.reshape(obs.shape).astype(np.float32)
        else:
            a=cv2.imread(str(ARDROSS/f'{k}.png'),0)
            if a is None: raise ReplayFailure(f'Missing historical channel {k}.png')
            hist[k]=a
    return hist

def compare_hf_to_historical(observed=None, reviewed_seed=None, hf_channels=None, out_dir=None):
    if hf_channels is None:
        if observed is None or reviewed_seed is None: raise ValueError('Provide hf_channels or observed + reviewed_seed')
        hf_channels=build_v78(observed,reviewed_seed)['channels']
    historical=_historical_channels()
    out=Path(out_dir or ROOT/'runtime_sessions'/'diagnostics'/'hf_vs_historical_v78')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for k in CHANNELS:
        h=np.asarray(historical[k]); g=np.asarray(hf_channels[k])
        shape_ok=h.shape==g.shape
        if shape_ok:
            if k in ['obs','norm'] and g.dtype.kind=='f' and g.max(initial=0)<=1.0: gcmp=np.rint(g*255).astype(np.uint8)
            elif k=='depth': gcmp=g.astype(np.float32)
            else: gcmp=np.where(g>0,255,0).astype(np.uint8)
            hcmp=h.astype(gcmp.dtype,copy=False)
            exact=bool(np.array_equal(gcmp,hcmp)); diff=np.abs(gcmp.astype(np.float64)-hcmp.astype(np.float64)); different=int(np.count_nonzero(diff)); max_abs=float(diff.max(initial=0)); mean_abs=float(diff.mean())
            if different:
                vis=np.clip(diff,0,255).astype(np.uint8); cv2.imwrite(str(out/f'{k}_absdiff.png'),vis)
        else:
            exact=False; different=-1; max_abs=float('nan'); mean_abs=float('nan'); gcmp=g
        rows.append({'channel':k,'shape_historical':str(h.shape),'shape_hf':str(g.shape),'shape_ok':shape_ok,'exact':exact,'different_pixels':different,'max_abs_difference':max_abs,'mean_abs_difference':mean_abs,'historical_sha256':array_digest(h),'hf_sha256':array_digest(gcmp)})
    with (out/'channel_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={'diagnostic_only':True,'blocks_startup':False,'channels':len(rows),'exact_channels':sum(r['exact'] for r in rows),'mismatched_channels':[r['channel'] for r in rows if not r['exact']],'all_exact':all(r['exact'] for r in rows)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary,rows
