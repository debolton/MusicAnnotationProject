import cv2,numpy as np
MASK_DILATION=2;CROP_MARGIN=48;MIN_COMPONENT_AREA=20;GEOMETRY_THRESHOLD=80;BAND_WIDTH=4;CONTEXT_RING_RADIUS=18

def immutable(original,candidate,mask):
    out=original.copy();out[mask>0]=candidate[mask>0];assert np.array_equal(out[mask==0],original[mask==0]);return out
def prepare_mask(seed,dilation=2):
    m=np.where(seed>0,255,0).astype(np.uint8)
    if dilation:m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))
    return m
def padded_components(mask):
    n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8);h,w=mask.shape;boxes=[]
    for i in range(1,n):
        x,y,bw,bh,a=stats[i]
        if a>=8:boxes.append((max(0,x-CROP_MARGIN),max(0,y-CROP_MARGIN),min(w,x+bw+CROP_MARGIN),min(h,y+bh+CROP_MARGIN)))
    changed=True
    while changed:
        changed=False;merged=[]
        while boxes:
            ax0,ay0,ax1,ay1=boxes.pop();rest=[]
            for bx0,by0,bx1,by1 in boxes:
                if not(ax1<bx0 or bx1<ax0 or ay1<by0 or by1<ay0):ax0=min(ax0,bx0);ay0=min(ay0,by0);ax1=max(ax1,bx1);ay1=max(ay1,by1);changed=True
                else:rest.append((bx0,by0,bx1,by1))
            boxes=rest;merged.append((ax0,ay0,ax1,ay1))
        boxes=merged
    return boxes
def telea_reconstruct(observed,mask):
    result=observed.copy()
    for x0,y0,x1,y1 in padded_components(mask):
        crop=observed[y0:y1,x0:x1];mk=mask[y0:y1,x0:x1];cand=cv2.inpaint(crop,mk,3,cv2.INPAINT_TELEA);local=result[y0:y1,x0:x1];local[mk>0]=cand[mk>0]
    return immutable(observed,result,mask)
def background_surface(observed,mask,sigma=45.):
    visible=(mask==0)&(observed>=200);num=cv2.GaussianBlur(observed.astype(np.float32)*visible,(0,0),sigma);den=cv2.GaussianBlur(visible.astype(np.float32),(0,0),sigma);fallback=cv2.GaussianBlur(observed,(0,0),sigma);return np.clip(np.where(den>.02,num/np.maximum(den,1e-4),fallback),0,255).astype(np.uint8)
def support_maps(observed,mask):
    dark=(observed<190)&(mask==0);h=cv2.morphologyEx(dark.astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((1,35),np.uint8))>0;v=cv2.morphologyEx(dark.astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((15,1),np.uint8))>0;n,lab,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8),8);c=np.zeros_like(dark)
    for i in range(1,n):
        x,y,w,hg,a=stats[i]
        if 3<=a<=650 and w<=40 and hg<=40:c[lab==i]=True
    k=np.ones((9,9),np.uint8);return cv2.dilate(h.astype(np.uint8),k)>0,cv2.dilate(v.astype(np.uint8),k)>0,cv2.dilate(c.astype(np.uint8),k)>0
def geometry_t80(observed,mask,telea):
    paper=background_surface(observed,mask);norm=cv2.divide(observed,np.maximum(paper,1),scale=255);horizontal,vertical,compact=support_maps(observed,mask);m=mask>0;preserve=m&(horizontal|vertical|compact)&(observed<200);remove=m&~preserve&(observed>=80)&(norm>=80);ambiguous=m&~preserve&~remove;result=observed.copy();result[remove]=paper[remove];result[ambiguous]=telea[ambiguous];result[preserve]=observed[preserve];return immutable(observed,result,mask),{'preserve':preserve,'remove':remove,'ambiguous':ambiguous,'horizontal':horizontal,'vertical':vertical,'compact':compact}
def surrounding_features(observed,component,routes):
    ys,xs=np.where(component);x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1;a=int(component.sum());ww=x1-x0;hh=y1-y0;k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37));ring=(cv2.dilate(component.astype(np.uint8),k)>0)&~component;vals=observed[ring];edges=cv2.Canny(observed,70,170)>0;depth=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
    def f(n):return float(np.sum(routes[n]&component)/a)
    return {'w':ww,'h':hh,'area':a,'aspect_ratio':max(ww,hh)/max(min(ww,hh),1),'fill_ratio':a/max(ww*hh,1),'touches_page_border':bool(x0==0 or y0==0 or x1==observed.shape[1] or y1==observed.shape[0]),'context_pixels':int(ring.sum()),'context_edge_density':float(edges[ring].mean()) if ring.any() else 0.,'context_dark_density':float((vals<190).mean()) if vals.size else 0.,'context_mean_intensity':float(vals.mean()) if vals.size else 255.,'context_intensity_std':float(vals.std()) if vals.size else 0.,'maximum_mask_depth':float(depth[component].max()),'horizontal_fraction':f('horizontal'),'vertical_fraction':f('vertical'),'compact_fraction':f('compact'),'preserve_fraction':f('preserve'),'remove_fraction':f('remove'),'ambiguous_fraction':f('ambiguous'),'reconstruction_dependency_fraction':f('ambiguous'),'component_pixels':a}
