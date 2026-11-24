from pathlib import Path
import uuid,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]/'runtime_sessions'; ROOT.mkdir(exist_ok=True)
def create(gray,confidence,draft,name='uploaded-page'):
    sid=uuid.uuid4().hex; d=ROOT/sid; d.mkdir()
    np.savez_compressed(d/'page.npz',gray=gray,confidence=confidence.astype(np.float16),draft=draft)
    (d/'metadata.json').write_text(json.dumps({'name':name},indent=2)); return sid
def load(sid,name='page'):
    p=ROOT/sid/f'{name}.npz'
    if not p.exists(): raise RuntimeError('Session expired. Generate the YOLO draft again.')
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(sid,name,**arrays): np.savez_compressed(ROOT/sid/f'{name}.npz',**arrays)
def write_json(sid,name,value):(ROOT/sid/f'{name}.json').write_text(json.dumps(value,indent=2,default=str))
def folder(sid):return ROOT/sid
