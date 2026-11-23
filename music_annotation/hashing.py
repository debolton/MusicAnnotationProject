import hashlib, json
from pathlib import Path
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def md5_file(path):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()

def array_hash(a):
    a=np.ascontiguousarray(a); h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes()); return h.hexdigest()

def digest_legacy(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()+str(a.shape).encode()+str(a.dtype).encode()).hexdigest()
