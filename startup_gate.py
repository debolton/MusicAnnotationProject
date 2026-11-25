from pathlib import Path
import csv,hashlib,json,platform
from importlib import metadata
ROOT=Path(__file__).resolve().parent; BUNDLE=ROOT/'frozen/a5_v1'; POLICY_ID='A5-v1-print-p0980-margin0960-run2'
EXPECTED=json.loads((ROOT/'expected_environment.json').read_text()); REQUIRED=['policy.json','channel_schema.json','model.pt','model_config.json','isotonic_calibrator.joblib','inference_runtime.py','artifact_manifest.csv','freeze_manifest.json']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def gate():
 names={'numpy':'numpy','opencv':'opencv-python-headless','scikit_learn':'scikit-learn','torch':'torch','joblib':'joblib','pandas':'pandas','pillow':'Pillow'}; actual={'python':platform.python_version()}
 for k,v in names.items():actual[k]=metadata.version(v).split('+')[0]
 bad=[f'{k}: expected {v}, loaded {actual.get(k)}' for k,v in EXPECTED.items() if actual.get(k)!=v]
 if bad:raise RuntimeError('MODULE VERSION GATE FAILED: '+' | '.join(bad))
 miss=[x for x in REQUIRED if not (BUNDLE/x).exists()]
 if miss:raise RuntimeError('A5 bundle incomplete: '+', '.join(miss))
 p=json.loads((BUNDLE/'policy.json').read_text())
 if p.get('policy_id')!=POLICY_ID:raise RuntimeError('Wrong A5 policy')
 for k,v in {'print_threshold':.98,'print_margin':.96,'annotation_threshold':.995,'minimum_print_run':2}.items():
  if float(p.get(k,-1))!=v:raise RuntimeError(f'A5 constant mismatch: {k}')
 for r in csv.DictReader((BUNDLE/'artifact_manifest.csv').open()):
  rel=r.get('relative_path') or r.get('path') or r.get('file'); exp=r.get('sha256') or r.get('SHA-256'); q=BUNDLE/rel
  if not q.exists() or not sha(q)==exp:
    raise RuntimeError(f'A5 artefact missing or incorrect: {rel}')
 
 
 f=json.loads((BUNDLE/'freeze_manifest.json').read_text()); 
 
 freeze_path = BUNDLE/'freeze_manifest.json'
 print("A5 freeze manifest:", freeze_path.resolve())
 print("A5 freeze manifest SHA-256:", sha(freeze_path))
 
 print("A5 freeze manifest keys:", sorted(f.keys()))
 print("A5 replay_passed raw:", repr(f.get("replay_passed")))
 print("A5 replay raw:", repr(f.get("replay")))

 if f.get('policy_id')!=POLICY_ID:
  raise RuntimeError('A5 freeze failed')
 print(f'Replay passed status: {f.get('replay_passed')}')
 if replay is not True:
  raise RuntimeError('Replay gate failed')
 return p
POLICY=gate()
if __name__=='__main__':print(json.dumps({'ok':True,'policy_id':POLICY_ID},indent=2))
