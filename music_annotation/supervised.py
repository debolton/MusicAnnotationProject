import cv2,numpy as np,pandas as pd
from .v78_engine import prepare_mask,telea_reconstruct,geometry_t80
from .v77_policy import score
MIN_COMPONENT_AREA=20;CONTEXT_RING_RADIUS=18

def surrounding_features(observed,component,routes):
 ys,xs=np.where(component);x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1;area=int(component.sum());w=x1-x0;h=y1-y0;k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37));ring=(cv2.dilate(component.astype(np.uint8),k)>0)&~component;vals=observed[ring];edges=cv2.Canny(observed,70,170)>0;depth=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
 def f(name):return float(np.sum(routes[name]&component)/area)
 return {'w':w,'h':h,'area':area,'aspect_ratio':max(w,h)/max(min(w,h),1),'fill_ratio':area/max(w*h,1),'touches_page_border':bool(x0==0 or y0==0 or x1==observed.shape[1] or y1==observed.shape[0]),'context_pixels':int(ring.sum()),'context_edge_density':float(edges[ring].mean()) if ring.any() else 0.0,'context_dark_density':float((vals<190).mean()) if vals.size else 0.0,'context_mean_intensity':float(vals.mean()) if vals.size else 255.0,'context_intensity_std':float(vals.std()) if vals.size else 0.0,'maximum_mask_depth':float(depth[component].max()),'horizontal_fraction':f('horizontal'),'vertical_fraction':f('vertical'),'compact_fraction':f('compact'),'preserve_fraction':f('preserve'),'remove_fraction':f('remove'),'ambiguous_fraction':f('ambiguous'),'reconstruction_dependency_fraction':f('ambiguous'),'component_pixels':area}
def build(observed,seed,confidence,bundle):
 seed=np.where(seed>0,255,0).astype(np.uint8);mask=prepare_mask(seed,2);telea=telea_reconstruct(observed,mask);proposal,routes=geometry_t80(observed,mask,telea);n,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8);rows=[]
 for cid in range(1,n):
  if stats[cid,cv2.CC_STAT_AREA]<MIN_COMPONENT_AREA:continue
  comp=labels==cid;row={'component':cid,'x':int(stats[cid,0]),'y':int(stats[cid,1]),'bbox_w':int(stats[cid,2]),'bbox_h':int(stats[cid,3]),'annotation_confidence_mean':float(confidence[comp].mean()),'annotation_confidence_max':float(confidence[comp].max())};row.update(surrounding_features(observed,comp,routes));rows.append(row)
 frame=score(pd.DataFrame(rows),bundle) if rows else pd.DataFrame()
 return {'seed':seed,'mask':mask,'proposal':proposal,'routes':routes,'labels':labels,'components':frame,'observed':observed}
def render_review(build,decisions):
 out=build['observed'].copy()
 for _,r in build['components'].iterrows():
  if decisions.get(int(r.component))=='accepted':
   m=build['labels']==int(r.component);out[m]=build['proposal'][m]
 assert np.array_equal(out[build['mask']==0],build['observed'][build['mask']==0]);return out
