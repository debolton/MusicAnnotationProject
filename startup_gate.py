from pathlib import Path
import csv,hashlib,json,platform
from importlib import metadata
ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'frozen/a5_v1'
POLICY_ID='A5-V1-P0980-M0960-A0995-RUN2'
IGNORE_NAMES={'.DS_Store','Thumbs.db'}
def generated(rel):
    p=Path(rel); return '__pycache__' in p.parts or p.suffix.lower() in {'.pyc','.pyo'} or p.name in IGNORE_NAMES
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def gate():
    required=['policy.json','channel_schema.json','model.pt','model_config.json','isotonic_calibrator.joblib','inference_runtime.py','artifact_manifest.csv','freeze_manifest.json']
    miss=[x for x in required if not (BUNDLE/x).exists()]
    if miss: raise RuntimeError('A5 bundle incomplete: '+', '.join(miss))
    policy=json.loads((BUNDLE/'policy.json').read_text())
    if policy.get('policy_id')!=POLICY_ID: raise RuntimeError('Wrong A5 policy ID')
    for k,v in {'print_threshold':.980,'print_margin':.960,'annotation_threshold':.995,'minimum_print_run':2}.items():
        if float(policy.get(k,-1))!=float(v): raise RuntimeError(f'A5 policy mismatch {k}')
    for r in csv.DictReader((BUNDLE/'artifact_manifest.csv').open(newline='')):
        rel=r.get('relative_path') or r.get('path') or r.get('file'); expected=r.get('sha256') or r.get('SHA-256')
        if not rel or generated(rel): continue
        q=BUNDLE/rel
        if not q.exists() or sha(q)!=expected: raise RuntimeError(f'A5 artefact failure: {rel}')
    if metadata.version('scikit-learn')!='1.6.1': raise RuntimeError('scikit-learn must be 1.6.1')
    if metadata.version('joblib')!='1.6.0': raise RuntimeError('joblib must be 1.6.0')
    from app.a5_operator import A5Operator
    operator=A5Operator()  # Runs mandatory frozen A5 replay.
    from app.hf044_v78_startup import run_ardross_v78_equivalence
    v78=run_ardross_v78_equivalence()  # Writes channel_comparison.csv and blocks on mismatch.
    return policy,operator.startup_replay,v78
POLICY,ARDROSS_REPLAY,ARDROSS_V78_EQUIVALENCE=gate()
if __name__=='__main__':
    print(json.dumps({'ready':True,'version':'HF-04.4','policy_id':POLICY_ID,'python':platform.python_version(),'ardross_a5_replay':ARDROSS_REPLAY,'ardross_v78_equivalence':ARDROSS_V78_EQUIVALENCE},indent=2))
