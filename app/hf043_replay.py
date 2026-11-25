from __future__ import annotations
from pathlib import Path
import csv, hashlib, json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARDROSS = ROOT / 'frozen' / 'a5_v1' / 'replay' / 'ardross'
CHANNELS = ['obs','norm','mask','depth','horizontal','vertical','compact','preserve','remove','ambiguous','edge']

class ReplayFailure(RuntimeError):
    pass

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()

def _read_vector(path: Path, kind='float'):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows=list(csv.reader(f))
    vals=[]
    for row in rows:
        if not row:
            continue
        s=row[-1].strip()
        try:
            vals.append(float(s) if kind=='float' else int(float(s)))
        except ValueError:
            continue
    return np.asarray(vals, dtype=np.float64 if kind=='float' else np.int64)

def _read_string_vector(path: Path):
    valid={'PRINT','ANNOTATION','UNCERTAIN'}
    values=[]
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row:
                continue
            value=row[-1].strip().upper()
            if value in valid:
                values.append(value)
    return np.asarray(values,dtype=str)

def _read_matrix(path: Path, dtype=np.float32):
    try:
        a=np.loadtxt(path,delimiter=',',dtype=dtype)
    except Exception as e:
        raise ReplayFailure(f'Cannot read replay matrix {path}: {e}')
    if a.ndim!=2:
        raise ReplayFailure(f'{path.name}: expected 2-D matrix, got {a.shape}')
    if not np.isfinite(a).all():
        raise ReplayFailure(f'{path.name}: non-finite values')
    return a

def _read_image(name, flags=cv2.IMREAD_GRAYSCALE):
    p=ARDROSS/name
    a=cv2.imread(str(p),flags)
    if a is None:
        raise ReplayFailure(f'Unreadable replay image: {p}')
    return a

def load_replay():
    required=['obs.png','norm.png','mask.png','depth.csv','horizontal.png','vertical.png','compact.png','preserve.png','remove.png','ambiguous.png','edge.png','x.csv','y.csv','expected_probability.csv','expected_provisional.csv','expected_final.csv']
    missing=[x for x in required if not (ARDROSS/x).exists()]
    if missing:
        raise ReplayFailure('Ardross replay incomplete: '+', '.join(missing))

    arrays={k:_read_image(k+'.png') for k in ['obs','norm','mask','horizontal','vertical','compact','preserve','remove','ambiguous','edge']}
    h,w=arrays['obs'].shape
    for k,a in arrays.items():
        if a.shape!=(h,w):
            raise ReplayFailure(f'Ardross {k} shape {a.shape} != {(h,w)}')

    depth=_read_matrix(ARDROSS/'depth.csv',np.float32)
    if depth.shape!=(h,w):
        raise ReplayFailure(f'Ardross depth shape mismatch: depth={depth.shape}, obs={(h,w)}')
    arrays['depth']=depth

    # Keep obs/norm byte-scale. Frozen make_padded() owns /255 exactly once.
    arrays['obs']=arrays['obs'].astype(np.uint8)
    arrays['norm']=arrays['norm'].astype(np.uint8)
    for k in ['mask','horizontal','vertical','compact','preserve','remove','ambiguous','edge']:
        arrays[k]=(arrays[k]>0)

    x=_read_vector(ARDROSS/'x.csv','int').astype(np.int64)
    y=_read_vector(ARDROSS/'y.csv','int').astype(np.int64)
    expected_p=_read_vector(ARDROSS/'expected_probability.csv','float').astype(np.float64)
    expected_prov=_read_string_vector(ARDROSS/'expected_provisional.csv')
    expected_final=_read_string_vector(ARDROSS/'expected_final.csv')
    lengths={
        'x':len(x),'y':len(y),'expected_probability':len(expected_p),
        'expected_provisional':len(expected_prov),'expected_final':len(expected_final),
    }
    if len(set(lengths.values()))!=1:
        raise ReplayFailure(f'Ardross replay vectors have inconsistent row counts: {lengths}')
    if len(x)!=5000:
        raise ReplayFailure(f'Ardross replay must contain 5000 coordinates, found {len(x)}')
    if (x<0).any() or (x>=w).any() or (y<0).any() or (y>=h).any():
        raise ReplayFailure('Ardross replay coordinate out of bounds')
    return arrays,y,x,expected_p,expected_prov,expected_final

def run_ardross_replay(operator, probability_atol=1e-7):
    arrays,y,x,expected_p,expected_prov,expected_final=load_replay()
    obs=np.asarray(arrays['obs']); norm=np.asarray(arrays['norm'])
    if obs.max()<=1 or norm.max()<=1:
        raise ReplayFailure('Ardross obs/norm appear pre-normalised; make_padded() must own /255 scaling.')

    padded=operator.rt.make_padded(arrays,operator.schema)
    got_p=np.asarray(operator.rt.score_coordinates(
        operator.model,operator.calibrator,padded,operator.schema,y,x,operator.device,batch=512
    ),dtype=np.float64).reshape(-1)
    got_prov,got_final=operator.rt.decisions(got_p,operator.policy,obs.shape,y,x)
    got_prov=np.asarray(got_prov,dtype=str).reshape(-1)
    got_final=np.asarray(got_final,dtype=str).reshape(-1)

    result={
        'name':'ardross_a5_startup_replay','rows':int(len(got_p)),
        'probability_exact':bool(np.array_equal(got_p,expected_p)),
        'probability_atol':float(probability_atol),
        'probability_within_atol':bool(np.allclose(got_p,expected_p,rtol=0,atol=probability_atol,equal_nan=False)),
        'probability_max_abs_error':float(np.max(np.abs(got_p-expected_p))),
        'provisional_exact':bool(np.array_equal(got_prov,expected_prov)),
        'provisional_mismatches':int(np.sum(got_prov!=expected_prov)),
        'final_exact':bool(np.array_equal(got_final,expected_final)),
        'final_mismatches':int(np.sum(got_final!=expected_final)),
    }
    result['passed']=bool(result['probability_within_atol'] and result['provisional_exact'] and result['final_exact'])
    if not result['passed']:
        raise ReplayFailure('Mandatory Ardross A5 replay failed: '+json.dumps(result,sort_keys=True))
    return result
