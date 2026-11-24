from pathlib import Path
import json,hashlib,joblib,numpy as np,pandas as pd
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(root):
 p=Path(root);req=['policy.json','predictor_schema.json','structural_model.joblib','isotonic_calibrator.joblib','replay_test.json']
 if not all((p/x).exists() for x in req):return None
 policy=json.loads((p/'policy.json').read_text());schema=json.loads((p/'predictor_schema.json').read_text());replay=json.loads((p/'replay_test.json').read_text())
 if not(replay.get('passed') is True and replay.get('policy_id')==policy.get('policy_id') and policy['model_sha256']==sha(p/'structural_model.joblib') and policy['calibrator_sha256']==sha(p/'isotonic_calibrator.joblib') and policy['schema_sha256']==sha(p/'predictor_schema.json')):raise RuntimeError('V7.7F integrity failure')
 return {'policy':policy,'schema':schema,'model':joblib.load(p/'structural_model.joblib'),'calibrator':joblib.load(p/'isotonic_calibrator.joblib')}
def score(frame,b):
 if b is None:return frame.assign(prob_structural_safe=np.nan,passes_frozen_structural_threshold=False,pilot_category='unscored')
 f=frame.copy();pred=b['schema']['ordered_predictors']
 for old,new in {'component_pixels_x':'component_pixels','component_pixels_y':'component_pixels'}.items():
  if old in pred and old not in f:f[old]=f[new]
 X=f[pred].apply(pd.to_numeric,errors='coerce');raw=b['model'].predict_proba(X)[:,1];p=b['calibrator'].predict(raw);f['prob_structural_safe']=p;f['passes_frozen_structural_threshold']=p>=float(b['policy']['threshold']);red=(f.ambiguous_fraction>.5)|((f.preserve_fraction>.2)&~f.passes_frozen_structural_threshold);f['pilot_category']=np.where(f.passes_frozen_structural_threshold,'green',np.where(red,'red','amber'));return f
