from pathlib import Path
import uuid, time, json, numpy as np
ROOT = Path(__file__).resolve().parents[1] / 'runtime_sessions'
ROOT.mkdir(exist_ok=True)

def create(gray, conf, draft):
    sid = uuid.uuid4().hex
    d = ROOT / sid
    d.mkdir()
    np.savez_compressed(d/'page.npz', gray=gray, conf=conf.astype(np.float16), draft=draft)
    (d/'meta.json').write_text(json.dumps({'created': time.time()}))
    return sid

def load(sid, name='page'):
    p = ROOT / sid / f'{name}.npz'
    if not p.exists():
        raise RuntimeError('Session expired. Generate the YOLO draft again.')
    with np.load(p) as z:
        return {k:z[k] for k in z.files}

def save(sid, name, **arrays):
    np.savez_compressed(ROOT/sid/f'{name}.npz', **arrays)
