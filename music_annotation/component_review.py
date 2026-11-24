import cv2,numpy as np,pandas as pd
from .v78_engine import prepare_mask,telea_reconstruct,geometry_t80,surrounding_features
LEGACY={'component_pixels_x':'component_pixels','component_pixels_y':'component_pixels'}
def build(obs,seed,conf,bundle):
 mask=prepare_mask(seed,2);telea=telea_reconstruct(obs,mask);proposal,routes=geometry_t80(obs,mask,telea);n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8);rows=[]
 for cid in range(1,n):
  if stats[cid,cv2.CC_STAT_AREA]<20:continue
  comp=lab==cid;row={'component':cid,'x':int(stats[cid,0]),'y':int(stats[cid,1]),'bbox_w':int(stats[cid,2]),'bbox_h':int(stats[cid,3]),'annotation_confidence_mean':float(conf[comp].mean()),'annotation_confidence_max':float(conf[comp].max())};row.update(surrounding_features(obs,comp,routes));rows.append(row)
 f=pd.DataFrame(rows)
 for old,new in LEGACY.items():
  if old in bundle['predictors'] and old not in f:f[old]=f[new]
 X=f[bundle['predictors']].apply(pd.to_numeric,errors='coerce');raw=bundle['model'].predict_proba(X)[:,1];prob=bundle['calibrator'].predict(raw);f['prob_structural_safe']=prob;f['passes_frozen_structural_threshold']=prob>=bundle['threshold'];red=(f.ambiguous_fraction>.5)|((f.preserve_fraction>.2)&~f.passes_frozen_structural_threshold);f['pilot_category']=np.where(f.passes_frozen_structural_threshold,'green',np.where(red,'red','amber'))
 return mask,proposal,routes,lab,f
def current(obs,proposal,labels,decisions):
 out=obs.copy()
 for k,v in decisions.items():
  if v=='accepted':m=labels==int(k);out[m]=proposal[m]
 return out
def crop_pair(obs,proposal,labels,frame,cid,margin=110):
 r=frame[frame.component.astype(int)==int(cid)].iloc[0];H,W=obs.shape;x0=max(0,int(r.x)-margin);y0=max(0,int(r.y)-margin);x1=min(W,int(r.x+r.bbox_w)+margin);y1=min(H,int(r.y+r.bbox_h)+margin);a=cv2.cvtColor(obs[y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);b=cv2.cvtColor(proposal[y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);div=np.full((a.shape[0],8,3),235,np.uint8);return np.hstack([a,div,b]),f'C{cid} | {r.pilot_category.upper()} | p_safe={r.prob_structural_safe:.4f} | preserve={r.preserve_fraction:.3f} | remove={r.remove_fraction:.3f} | ambiguous={r.ambiguous_fraction:.3f}'
