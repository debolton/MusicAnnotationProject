from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from .hashing import sha256_file
def load_policy(root):
 p=Path(root);req=['policy.json','predictor_schema.json','structural_model.joblib','isotonic_calibrator.joblib','replay_test.json'];missing=[x for x in req if not(p/x).exists()]
 if missing:raise RuntimeError(f'Incomplete V7.7F bundle: {missing}')
 policy=json.loads((p/'policy.json').read_text());schema=json.loads((p/'predictor_schema.json').read_text());replay=json.loads((p/'replay_test.json').read_text());checks=[replay.get('passed') is True,replay.get('policy_id')==policy.get('policy_id'),policy['model_sha256']==sha256_file(p/'structural_model.joblib'),policy['calibrator_sha256']==sha256_file(p/'isotonic_calibrator.joblib'),policy['schema_sha256']==sha256_file(p/'predictor_schema.json')]
 if not all(checks):raise RuntimeError('V7.7F policy integrity failure.')
 return {'model':joblib.load(p/'structural_model.joblib'),'calibrator':joblib.load(p/'isotonic_calibrator.joblib'),'predictors':schema['ordered_predictors'],'threshold':float(policy['threshold']),'policy':policy}
def score(frame,b):
 frame=frame.copy()
 for old,new in {'component_pixels_x':'component_pixels','component_pixels_y':'component_pixels'}.items():
  if old in b['predictors'] and old not in frame:frame[old]=frame[new]
 X=frame[b['predictors']].apply(pd.to_numeric,errors='coerce')
 if np.isinf(X.to_numpy(float)).any():raise RuntimeError('Infinite predictor.')
 raw=b['model'].predict_proba(X)[:,1];p=b['calibrator'].predict(raw);frame['prob_structural_safe']=p;frame['passes_frozen_structural_threshold']=p>=b['threshold'];red=(frame.ambiguous_fraction>.5)|((frame.preserve_fraction>.2)&~frame.passes_frozen_structural_threshold);frame['pilot_category']=np.where(frame.passes_frozen_structural_threshold,'green',np.where(red,'red','amber'));return frame
