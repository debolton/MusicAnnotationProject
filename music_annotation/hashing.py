import hashlib

def file_hash(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def sha256_file(path): return file_hash(path,'sha256')
def md5_file(path): return file_hash(path,'md5')
