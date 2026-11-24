from pathlib import Path
import json,uuid,numpy as np
ROOT=Path(__file__).resolve().parents[1]/'runtime_sessions';ROOT.mkdir(exist_ok=True)
def create(gray,confidence,draft,name):
 sid=uuid.uuid4().hex;d=ROOT/sid;d.mkdir();np.savez_compressed(d/'state.npz',gray=gray,confidence=confidence.astype(np.float16),draft=draft,working=draft);(d/'audit.json').write_text(json.dumps({'session_id':sid,'source_name':name,'formula':'(YOLO draft OR ADD) AND NOT ERASE','revision':0,'events':[]},indent=2));return sid
def folder(sid):
 d=ROOT/sid
 if not d.exists():raise FileNotFoundError('Session not found or expired')
 return d
def load(sid):
 with np.load(folder(sid)/'state.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(sid,state):np.savez_compressed(folder(sid)/'state.npz',**state)
def read_audit(sid):return json.loads((folder(sid)/'audit.json').read_text())
def write_audit(sid,a):(folder(sid)/'audit.json').write_text(json.dumps(a,indent=2,default=str))
