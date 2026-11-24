import base64,cv2,numpy as np,pandas as pd
from .v78_engine import prepare_mask,telea_reconstruct,geometry_t80,surrounding_features
from .v77_policy import score
MAX_W=1200

def preview_size(shape):
    h,w=shape;s=min(1.,MAX_W/w);return max(1,round(w*s)),max(1,round(h*s))
def resize(a,size,nearest=False):return cv2.resize(a,size,interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
def data_url(a,ext='.png'):
    ok,b=cv2.imencode(ext,a); assert ok;mime='image/png' if ext=='.png' else 'image/jpeg';return 'data:'+mime+';base64,'+base64.b64encode(b).decode()
def package(gray,draft,conf,name):
    sz=preview_size(gray.shape);g=resize(gray,sz);d=resize(draft,sz,True);c=np.clip(resize(conf,sz)*255,0,255).astype(np.uint8);return {'name':name,'width':sz[0],'height':sz[1],'image':data_url(g,'.jpg'),'draft':data_url(d,'.png'),'confidence':data_url(c,'.png')}
def decode_url(value,shape):
    if not value:return np.zeros(shape,np.uint8)
    raw=base64.b64decode(value.split(',',1)[1] if ',' in value else value);im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    if im is None:raise RuntimeError('Browser correction mask could not be decoded.')
    if im.shape!=shape:im=cv2.resize(im,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return np.where(im>0,255,0).astype(np.uint8)
def build(observed,seed,confidence,bundle):
    mask=prepare_mask(seed,2);telea=telea_reconstruct(observed,mask);proposal,routes=geometry_t80(observed,mask,telea);n,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8);rows=[]
    for cid in range(1,n):
        if stats[cid,cv2.CC_STAT_AREA]<20:continue
        comp=labels==cid;row={'component':cid,'x':int(stats[cid,0]),'y':int(stats[cid,1]),'bbox_w':int(stats[cid,2]),'bbox_h':int(stats[cid,3]),'annotation_confidence_mean':float(confidence[comp].mean()),'annotation_confidence_max':float(confidence[comp].max())};row.update(surrounding_features(observed,comp,routes));rows.append(row)
    frame=score(pd.DataFrame(rows),bundle) if rows else pd.DataFrame();return mask,proposal,labels,frame
def current(observed,proposal,labels,decisions):
    out=observed.copy()
    for k,v in decisions.items():
        if v=='accepted':m=labels==int(k);out[m]=proposal[m]
    return out
def component_pair(obs,proposal,labels,frame,cid,margin=110):
    r=frame[frame.component.astype(int)==int(cid)].iloc[0];H,W=obs.shape;x0=max(0,int(r.x)-margin);y0=max(0,int(r.y)-margin);x1=min(W,int(r.x+r.bbox_w)+margin);y1=min(H,int(r.y+r.bbox_h)+margin);a=cv2.cvtColor(obs[y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);b=cv2.cvtColor(proposal[y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);div=np.full((a.shape[0],8,3),235,np.uint8);return np.hstack([a,div,b]),f'C{cid} | {r.pilot_category.upper()} | p_safe={r.prob_structural_safe:.4f} | preserve={r.preserve_fraction:.3f} | remove={r.remove_fraction:.3f} | ambiguous={r.ambiguous_fraction:.3f}'
