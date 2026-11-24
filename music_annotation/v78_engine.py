import cv2,numpy as np
DIL=2;MARGIN=48
def immutable(o,c,m):
 x=o.copy();x[m>0]=c[m>0];assert np.array_equal(x[m==0],o[m==0]);return x
def prepare_mask(seed):return cv2.dilate(np.where(seed>0,255,0).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
def telea(o,m):return immutable(o,cv2.inpaint(o,m,3,cv2.INPAINT_TELEA),m)
def background(o,m):
 v=(m==0)&(o>=200);num=cv2.GaussianBlur(o.astype(np.float32)*v,(0,0),45);den=cv2.GaussianBlur(v.astype(np.float32),(0,0),45);fb=cv2.GaussianBlur(o,(0,0),45);return np.clip(np.where(den>.02,num/np.maximum(den,1e-4),fb),0,255).astype(np.uint8)
def support(o,m):
 d=(o<190)&(m==0);h=cv2.morphologyEx(d.astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((1,35),np.uint8))>0;v=cv2.morphologyEx(d.astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((15,1),np.uint8))>0;n,l,s,_=cv2.connectedComponentsWithStats(d.astype(np.uint8),8);c=np.zeros_like(d)
 for i in range(1,n):
  x,y,w,hg,a=s[i]
  if 3<=a<=650 and w<=40 and hg<=40:c[l==i]=1
 k=np.ones((9,9),np.uint8);return cv2.dilate(h.astype(np.uint8),k)>0,cv2.dilate(v.astype(np.uint8),k)>0,cv2.dilate(c.astype(np.uint8),k)>0
def geometry(o,m,t):
 paper=background(o,m);norm=cv2.divide(o,np.maximum(paper,1),scale=255);h,v,c=support(o,m);q=m>0;pre=q&(h|v|c)&(o<200);rem=q&~pre&(o>=80)&(norm>=80);amb=q&~pre&~rem;r=o.copy();r[rem]=paper[rem];r[amb]=t[amb];return immutable(o,r,m),{'preserve':pre,'remove':rem,'ambiguous':amb,'horizontal':h,'vertical':v,'compact':c}
def features(o,c,r):
 ys,xs=np.where(c);x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1;a=int(c.sum());w=x1-x0;h=y1-y0;k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37));ring=(cv2.dilate(c.astype(np.uint8),k)>0)&~c;vals=o[ring];edge=cv2.Canny(o,70,170)>0;depth=cv2.distanceTransform(c.astype(np.uint8),cv2.DIST_L2,5);f=lambda n:float(np.sum(r[n]&c)/a)
 return {'w':w,'h':h,'area':a,'aspect_ratio':max(w,h)/max(min(w,h),1),'fill_ratio':a/max(w*h,1),'touches_page_border':bool(x0==0 or y0==0 or x1==o.shape[1] or y1==o.shape[0]),'context_pixels':int(ring.sum()),'context_edge_density':float(edge[ring].mean()) if ring.any() else 0.,'context_dark_density':float((vals<190).mean()) if vals.size else 0.,'context_mean_intensity':float(vals.mean()) if vals.size else 255.,'context_intensity_std':float(vals.std()) if vals.size else 0.,'maximum_mask_depth':float(depth[c].max()),'horizontal_fraction':f('horizontal'),'vertical_fraction':f('vertical'),'compact_fraction':f('compact'),'preserve_fraction':f('preserve'),'remove_fraction':f('remove'),'ambiguous_fraction':f('ambiguous'),'reconstruction_dependency_fraction':f('ambiguous'),'component_pixels':a}
