from pathlib import Path
import uuid, numpy as np
ROOT=Path(__file__).resolve().parents[1]/'runtime_sessions'; ROOT.mkdir(exist_ok=True)
def create(gray,conf,draft):
 d=ROOT/uuid.uuid4().hex; d.mkdir(); np.savez_compressed(d/'page.npz',gray=gray,conf=conf.astype(np.float16),draft=draft); return d.name
def load(sid,name='page'):
 with np.load(ROOT/sid/f'{name}.npz') as z:return {k:z[k] for k in z.files}
def save(sid,name,**arrays):np.savez_compressed(ROOT/sid/f'{name}.npz',**arrays)
