import hashlib
import numpy as np

def array_signature(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()+str(a.shape).encode()+str(a.dtype).encode()).hexdigest()

def component_signature(labels,cid,proposal,routes):
    m=labels==int(cid); ys,xs=np.where(m)
    if not len(xs):return None
    x0,x1=xs.min(),xs.max()+1; y0,y1=ys.min(),ys.max()+1; parts=[m[y0:y1,x0:x1].astype(np.uint8),proposal[y0:y1,x0:x1]]+[routes[k][y0:y1,x0:x1].astype(np.uint8) for k in ['preserve','remove','ambiguous']]
    return (int(x0),int(y0),int(x1),int(y1),*(array_signature(a) for a in parts))
