from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from .hashing import sha256_file

def load_policy(root):
 p=Path(root); required=['policy.json','predictor_schema.json','structural_model.joblib','isotonic_calibrator.joblib','replay_test.json']; missing=[x for x in required if not (p/x).exists()]
 if missing:raise RuntimeError(f'Incomplete V7.7F policy bundle: {missing}')
 policy=json.loads((p/'policy.json').read_text());schema=json.loads((p/'predictor_schema.json').read_text());replay=json.loads((p/'replay_test.json').read_text())
 checks={'replay':replay.get('passed') is True and replay.get('policy_id')==policy.get('policy_id'),'model_hash':policy['model_sha256']==sha256_file(p/'structural_model.joblib'),'calibrator_hash':policy['calibrator_sha256']==sha256_file(p/'isotonic_calibrator.joblib'),'schema_hash':policy['schema_sha256']==sha256_file(p/'predictor_schema.json')}
 if not all(checks.values()):raise RuntimeError(f'Frozen policy integrity failure: {checks}')
 return {'policy':policy,'schema':schema,'replay':replay,'model':joblib.load(p/'structural_model.joblib'),'calibrator':joblib.load(p/'isotonic_calibrator.joblib'),'predictors':schema['ordered_predictors'],'threshold':float(policy['threshold']),'checks':checks}
def score(frame,bundle,aliases={'component_pixels_x':'component_pixels','component_pixels_y':'component_pixels'}):
 frame=frame.copy()
 for old,new in aliases.items():
  if old in bundle['predictors'] and old not in frame:frame[old]=frame[new]
 missing=[x for x in bundle['predictors'] if x not in frame]
 if missing:raise RuntimeError(f'Missing frozen predictors: {missing}')
 X=frame[bundle['predictors']].apply(pd.to_numeric,errors='coerce')
 if np.isinf(X.to_numpy(float)).any():raise RuntimeError('Infinite predictor value found.')
 raw=bundle['model'].predict_proba(X)[:,1];prob=bundle['calibrator'].predict(raw);frame['raw_structural_safe_score']=raw;frame['prob_structural_safe']=prob;frame['passes_frozen_structural_threshold']=prob>=bundle['threshold'];red=(frame.ambiguous_fraction>.50)|((frame.preserve_fraction>.20)&~frame.passes_frozen_structural_threshold);frame['pilot_category']=np.where(frame.passes_frozen_structural_threshold,'green',np.where(red,'red','amber'));return frame
