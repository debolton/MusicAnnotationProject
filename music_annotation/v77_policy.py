from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from .hashing import sha256_file

def load_policy(root):
    p=Path(root);req=['policy.json','predictor_schema.json','structural_model.joblib','isotonic_calibrator.joblib','replay_test.json'];missing=[x for x in req if not(p/x).exists()]
    if missing:raise RuntimeError(f'Incomplete V7.7F bundle: {missing}')
    policy=json.loads((p/'policy.json').read_text());schema=json.loads((p/'predictor_schema.json').read_text());replay=json.loads((p/'replay_test.json').read_text());checks={'replay':replay.get('passed') is True and replay.get('policy_id')==policy.get('policy_id'),'model':policy['model_sha256']==sha256_file(p/'structural_model.joblib'),'calibrator':policy['calibrator_sha256']==sha256_file(p/'isotonic_calibrator.joblib'),'schema':policy['schema_sha256']==sha256_file(p/'predictor_schema.json')}
    if not all(checks.values()):raise RuntimeError(f'V7.7F integrity failure: {checks}')
    return {'policy':policy,'model':joblib.load(p/'structural_model.joblib'),'calibrator':joblib.load(p/'isotonic_calibrator.joblib'),'predictors':schema['ordered_predictors'],'threshold':float(policy['threshold'])}
def score(frame,b):
    frame=frame.copy()
    for old,new in {'component_pixels_x':'component_pixels','component_pixels_y':'component_pixels'}.items():
        if old in b['predictors'] and old not in frame:frame[old]=frame[new]
    missing=[x for x in b['predictors'] if x not in frame]
    if missing:raise RuntimeError(f'Missing frozen predictors: {missing}')
    X=frame[b['predictors']].apply(pd.to_numeric,errors='coerce')
    if np.isinf(X.to_numpy(float)).any():raise RuntimeError('Infinite predictor value.')
    raw=b['model'].predict_proba(X)[:,1];prob=b['calibrator'].predict(raw);frame['prob_structural_safe']=prob;frame['passes_frozen_structural_threshold']=prob>=b['threshold'];red=(frame.ambiguous_fraction>.5)|((frame.preserve_fraction>.2)&~frame.passes_frozen_structural_threshold);frame['pilot_category']=np.where(frame.passes_frozen_structural_threshold,'green',np.where(red,'red','amber'));return frame
