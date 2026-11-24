import base64,cv2,numpy as np,pandas as pd,zipfile,json
from pathlib import Path
from .v78_engine import prepare_mask,telea,geometry,features
from .v77_policy import score
REVIEW_W=960
def psize(shape):h,w=shape;s=min(1.,REVIEW_W/w);return max(1,round(w*s)),max(1,round(h*s))
def resize(a,sz,near=False):return cv2.resize(a,sz,interpolation=cv2.INTER_NEAREST if near else cv2.INTER_AREA)
def url(a,ext='.png'):
 ok,b=cv2.imencode(ext,a);mime='image/png' if ext=='.png' else 'image/jpeg';return 'data:'+mime+';base64,'+base64.b64encode(b).decode()
def package(g,d,c,name):
 sz=psize(g.shape);return {'name':name,'width':sz[0],'height':sz[1],'image':url(resize(g,sz),'.jpg'),'draft':url(resize(d,sz,True)), 'confidence':url(np.clip(resize(c,sz)*255,0,255).astype(np.uint8))}
def decode(v,shape):
 if not v:return np.zeros(shape,np.uint8)
 a=cv2.imdecode(np.frombuffer(base64.b64decode(v.split(',',1)[1]),np.uint8),0)
 return cv2.resize(np.where(a>0,255,0).astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST) if a.shape!=shape else np.where(a>0,255,0).astype(np.uint8)
def build(o,seed,conf,bundle):
 m=prepare_mask(seed);t=telea(o,m);p,r=geometry(o,m,t);n,l,s,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8);rows=[]
 for cid in range(1,n):
  if s[cid,4]<20:continue
  comp=l==cid;row={'component':cid,'x':int(s[cid,0]),'y':int(s[cid,1]),'bbox_w':int(s[cid,2]),'bbox_h':int(s[cid,3]),'annotation_confidence_mean':float(conf[comp].mean()),'annotation_confidence_max':float(conf[comp].max())};row.update(features(o,comp,r));rows.append(row)
 f=score(pd.DataFrame(rows),bundle) if rows else pd.DataFrame();return m,p,l,f
def current(o,p,l,d):
 q=o.copy()
 for k,v in d.items():
  if v=='accepted':q[l==int(k)]=p[l==int(k)]
 return q
def archive(folder,gray,draft,conf,seed,prepared,proposal,decisions):
 cv2.imwrite(str(folder/'annotated_page.png'),gray);cv2.imwrite(str(folder/'yolo_draft_mask.png'),draft);np.save(folder/'yolo_confidence.npy',conf.astype(np.float16));cv2.imwrite(str(folder/'reviewed_authoritative_mask.png'),seed);cv2.imwrite(str(folder/'prepared_dilation2_mask.png'),prepared);cv2.imwrite(str(folder/'v78_proposal.png'),proposal);(folder/'review_audit.json').write_text(json.dumps({'formula':'(draft OR ADD) AND NOT ERASE','decisions':decisions},indent=2));z=folder/'training_and_audit_artefacts.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
  for p in folder.iterdir():
   if p.is_file() and p!=z:q.write(p,p.name)
 return z
