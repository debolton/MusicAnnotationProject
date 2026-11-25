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

def _read_image(name, flags=cv2.IMREAD_GRAYSCALE):
    p=ARDROSS/name
    a=cv2.imread(str(p),flags)
    if a is None: raise ReplayFailure(f'Unreadable replay image: {p}')
    return a

def load_replay():
    required=['obs.png','norm.png','mask.png','depth.csv','horizontal.png','vertical.png','compact.png','preserve.png','remove.png','ambiguous.png','edge.png','x.csv','y.csv','expected_probability.csv','expected_provisional.csv','expected_final.csv']
    missing=[x for x in required if not (ARDROSS/x).exists()]
    if missing: raise ReplayFailure('Ardross replay incomplete: '+', '.join(missing))
    arrays={k:_read_image(k+'.png') for k in ['obs','norm','mask','horizontal','vertical','compact','preserve','remove','ambiguous','edge']}
    h,w=arrays['obs'].shape
    for k,a in arrays.items():
        if a.shape!=(h,w): raise ReplayFailure(f'Ardross {k} shape {a.shape} != {(h,w)}')
    depth=_read_vector(ARDROSS/'depth.csv')
    if depth.size!=h*w: raise ReplayFailure(f'Ardross depth rows {depth.size} != image pixels {h*w}')
    arrays['depth']=depth.reshape(h,w).astype(np.float32)
    arrays['obs']=arrays['obs'].astype(np.float32)/255.0
    arrays['norm']=arrays['norm'].astype(np.float32)/255.0
    for k in ['mask','horizontal','vertical','compact','preserve','remove','ambiguous','edge']:
        arrays[k]=(arrays[k]>0).astype(np.float32)
    x=_read_vector(ARDROSS/'x.csv','int').astype(np.int64)
    y=_read_vector(ARDROSS/'y.csv','int').astype(np.int64)
    expected_p=_read_vector(ARDROSS/'expected_probability.csv')
    expected_prov=_read_vector(ARDROSS/'expected_provisional.csv','int')
    expected_final=_read_vector(ARDROSS/'expected_final.csv','int')
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

def run_ardross_replay(operator, probability_atol=1e-7):
    arrays,y,x,ep,eprov,efinal=load_replay()
    rt=operator.rt
    padded=rt.make_padded(arrays,operator.schema)
    got_p=np.asarray(rt.score_coordinates(operator.model,operator.calibrator,padded,operator.schema,y,x,operator.device,batch=512),dtype=np.float64)
    got_prov,got_final=rt.decisions(got_p,operator.policy,arrays['obs'].shape,y,x)
    got_prov=_canonical_labels(got_prov); got_final=_canonical_labels(got_final)
    eprov=_canonical_labels(eprov); efinal=_canonical_labels(efinal)
    probability_exact=bool(np.array_equal(got_p,ep))
    probability_close=bool(np.allclose(got_p,ep,rtol=0,atol=probability_atol,equal_nan=False))
    provisional_exact=bool(np.array_equal(got_prov,eprov))
    final_exact=bool(np.array_equal(got_final,efinal))
    result={'name':'ardross_a5_startup_replay','rows':int(len(ep)),'probability_exact':probability_exact,'probability_atol':probability_atol,'probability_within_atol':probability_close,'probability_max_abs_error':float(np.max(np.abs(got_p-ep))),'provisional_exact':provisional_exact,'final_exact':final_exact,'passed':bool(probability_close and provisional_exact and final_exact)}
    if not result['passed']: raise ReplayFailure('Mandatory Ardross A5 replay failed: '+json.dumps(result,sort_keys=True))
    return result
