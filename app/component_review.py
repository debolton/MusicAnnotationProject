from pathlib import Path
from datetime import datetime,timezone
import json,cv2,numpy as np,pandas as pd
VALID={'accepted','left_unchanged','deferred','unreviewed'}
def load(d):
 d=Path(d); frame=pd.read_csv(d/'components.csv'); dec=json.loads((d/'decisions.json').read_text());
 with np.load(d/'build.npz') as z: arrays={k:z[k] for k in z.files}
 observed=cv2.imread(str(d/'annotated_page.png'),0)
 if observed is None: raise RuntimeError('annotated_page.png unavailable')
 return frame,dec,arrays,observed
def save_decisions(d,dec):Path(d,'decisions.json').write_text(json.dumps(dec,indent=2))
def row(frame,c):
 q=frame[frame.component.astype(int)==int(c)]
 if q.empty:raise KeyError(f'Component {c} not found')
 return q.iloc[0]
def crop_box(r,shape,margin=100):
 h,w=shape;x0=max(0,int(r.x)-margin);y0=max(0,int(r.y)-margin);x1=min(w,int(r.x+r.bbox_w)+margin);y1=min(h,int(r.y+r.bbox_h)+margin);return x0,y0,x1,y1
def component_payload(d,c):
 frame,dec,a,obs=load(d);r=row(frame,c);labels=a['labels'];proposal=a['proposal'];mask=labels==int(c);x0,y0,x1,y1=crop_box(r,obs.shape)
 original=obs[y0:y1,x0:x1];proposed=proposal[y0:y1,x0:x1];m=mask[y0:y1,x0:x1]
 return {'row':r,'status':dec[str(int(c))],'original':original,'proposed':proposed,'mask':m,'box':(x0,y0,x1,y1)}
def queue(d):
 frame,dec,_,_=load(d);rows=[]
 order={'red':0,'amber':1,'green':2}
 for _,r in frame.iterrows():
  c=int(r.component);v=r.to_dict();v['component']=c;v['status']=dec.get(str(c),'unreviewed');rows.append(v)
 rows.sort(key=lambda x:(x['status'] not in {'unreviewed','deferred'},order.get(x.get('pilot_category','amber'),1),-float(x.get('ambiguous_fraction',0)),x['component']))
 return rows
def decide(d,c,status,note=''):
 if status not in VALID:raise ValueError(status)
 d=Path(d);dec=json.loads((d/'decisions.json').read_text());dec[str(int(c))]=status;save_decisions(d,dec)
 audit=d/'component_review_audit.json';a=json.loads(audit.read_text()) if audit.exists() else {'events':[]};a['events'].append({'utc':datetime.now(timezone.utc).isoformat(),'component':int(c),'status':status,'note':note});audit.write_text(json.dumps(a,indent=2));return summary(d)
def summary(d):
 _,dec,_,_=load(d);vals=list(dec.values());return {k:vals.count(k) for k in ['unreviewed','accepted','left_unchanged','deferred']}
def assemble(d,approval_note=''):
 d=Path(d);frame,dec,a,obs=load(d);pending=[int(k) for k,v in dec.items() if v in {'unreviewed','deferred'}]
 if pending:raise RuntimeError(f'Page not ready: {len(pending)} component(s) unreviewed/deferred: {pending}')
 out=obs.copy();labels=a['labels'];proposal=a['proposal']
 for _,r in frame.iterrows():
  c=int(r.component)
  if dec[str(c)]=='accepted':m=labels==c;out[m]=proposal[m]
 mask=a['mask']>0
 if not np.array_equal(out[~mask],obs[~mask]):raise RuntimeError('Outside-mask immutability failure')
 cv2.imwrite(str(d/'final_approved_page.png'),out);record={'utc':datetime.now(timezone.utc).isoformat(),'approval_note':approval_note,'decisions':dec,'accepted':sum(v=='accepted' for v in dec.values()),'left_unchanged':sum(v=='left_unchanged' for v in dec.values()),'outside_mask_immutable':True};(d/'final_approval.json').write_text(json.dumps(record,indent=2));return out,record
