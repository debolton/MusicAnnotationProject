import base64,cv2,numpy as np
MAX_W=1200
def size(shape):
 h,w=shape;s=min(1.,MAX_W/w);return max(1,round(w*s)),max(1,round(h*s))
def resize(a,sz,nearest=False):return cv2.resize(a,sz,interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
def url(a,ext='.png'):
 ok,b=cv2.imencode(ext,a); assert ok; mime='image/png' if ext=='.png' else 'image/jpeg';return 'data:'+mime+';base64,'+base64.b64encode(b).decode()
def package(gray,draft,conf,name):
 sz=size(gray.shape);g=resize(gray,sz);d=resize(draft,sz,True);c=np.clip(resize(conf,sz)*255,0,255).astype(np.uint8)
 return {'name':name,'width':sz[0],'height':sz[1],'image':url(g,'.jpg'),'draft':url(d,'.png'),'confidence':url(c,'.png'),'full_width':gray.shape[1],'full_height':gray.shape[0]}
def decode_data_url(v,shape):
 raw=base64.b64decode(v.split(',',1)[1]);im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
 if im.shape!=shape:im=cv2.resize(im,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
 return np.where(im>0,255,0).astype(np.uint8)
