from pathlib import Path
import json, hashlib, joblib, numpy as np, pandas as pd

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify_and_load(root):
    p=Path(root)
    req=[
        'policy.json',
        'predictor_schema.json',
        'structural_model.joblib',
        'isotonic_calibrator.joblib',
        'replay_test.json'
    ]
    missing=[x for x in req if not (p/x).exists()]
    
    if missing:
        return None,{
            'available':False,'missing':missing
        }
    
    pol=json.loads((p/'policy.json').read_text())
    schema=json.loads((p/'predictor_schema.json').read_text())
    replay=json.loads((p/'replay_test.json').read_text())
    checks={
        'replay':replay.get('passed') is True and replay.get('policy_id')==pol.get('policy_id'),
        'model_hash':pol.get('model_sha256')==sha(p/'structural_model.joblib'),
        'calibrator_hash':pol.get('calibrator_sha256')==sha(p/'isotonic_calibrator.joblib'),
        'schema_hash':pol.get('schema_sha256')==sha(p/'predictor_schema.json')
    }
    
    if not all(checks.values()):
        raise RuntimeError(f'Frozen policy integrity failure: {checks}')
    
    return {
        'policy':pol,
        'schema':schema,
        'model':joblib.load(p/'structural_model.joblib'),
        'calibrator':joblib.load(p/'isotonic_calibrator.joblib')
    },
    {
        'available':True,
        'checks':checks,
        'policy_id':pol.get('policy_id'),
        'threshold':float(pol['threshold'])
    }

def score(frame,b):
    if b is None:
        f=frame.copy()
        f['raw_structural_safe_score']=np.nan
        f['prob_structural_safe']=np.nan
        f['passes_frozen_structural_threshold']=False
        f['pilot_category']='unscored'
        return f

    f=frame.copy();pred=list(b['schema']['ordered_predictors'])
    for old,new in {'component_pixels_x':'component_pixels','component_pixels_y':'component_pixels'}.items():
        if old in pred and old not in f.columns and new in f.columns:f[old]=f[new]
    
    missing=[x for x in pred if x not in f.columns]
    if missing:
        raise RuntimeError(f'Missing frozen predictors: {missing}')
    
    X=f.loc[:,pred].copy()
    
    for c in pred:X[c]=pd.to_numeric(X[c],errors='coerce')
    
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise RuntimeError('Infinite predictor value found.')
    
    raw=b['model'].predict_proba(X)[:,1]
    prob=b['calibrator'].predict(raw)
    f['raw_structural_safe_score']=raw
    f['prob_structural_safe']=prob
    f['passes_frozen_structural_threshold']=prob>=float(b['policy']['threshold'])
    red=(f['ambiguous_fraction']>.5)|((f['preserve_fraction']>.2)&~f['passes_frozen_structural_threshold'])
    f['pilot_category']=np.where(f['passes_frozen_structural_threshold'],'green',np.where(red,'red','amber'))
    
    return f
