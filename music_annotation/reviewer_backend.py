import base64,cv2,numpy as np,pandas as pd,zipfile,json
from .canonical_v78 import prepare_mask,telea_reconstruct,geometry_t80,surrounding_features,MIN_COMPONENT_AREA
from .v77f import score
REVIEW_MAX_WIDTH=960

def preview_size(shape):
    h,w=shape;s=min(1.0,REVIEW_MAX_WIDTH/w);return max(1,round(w*s)),max(1,round(h*s))
def resize(a,size,nearest=False):return cv2.resize(a,size,interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
def data_url(a,ext='.png'):
    ok,b=cv2.imencode(ext,a);assert ok;mime='image/png' if ext=='.png' else 'image/jpeg';return 'data:'+mime+';base64,'+base64.b64encode(b).decode()
def overlay_preview(gray,draft,alpha=.48):
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);colour=rgb.copy();colour[draft>0]=(60,60,255);return cv2.addWeighted(rgb,1-alpha,colour,alpha,0)
def package(gray,draft,confidence,name):
    size=preview_size(gray.shape);small=resize(gray,size);dm=resize(draft,size,True);cm=np.clip(resize(confidence,size)*255,0,255).astype(np.uint8)
    return {'name':name,'width':size[0],'height':size[1],'page_url':data_url(small,'.jpg'),'overlay_url':data_url(overlay_preview(small,dm),'.jpg'),'mask_url':data_url(dm,'.png'),'confidence_url':data_url(cm,'.png')}
def rasterise_strokes(strokes,preview_shape,full_shape):
    add=np.zeros(preview_shape,np.uint8);erase=np.zeros(preview_shape,np.uint8)
    for s in strokes:
        target=add if s.get('mode')=='add' else erase if s.get('mode')=='erase' else None
        if target is None:continue
        pts=s.get('points') or [];brush=max(1,int(s.get('brush',14)))
        if len(pts)==1:cv2.circle(target,(round(pts[0][0]),round(pts[0][1])),max(1,brush//2),255,-1)
        for a,b in zip(pts,pts[1:]):cv2.line(target,(round(a[0]),round(a[1])),(round(b[0]),round(b[1])),255,brush,cv2.LINE_AA)
    W,H=full_shape[1],full_shape[0]
    return cv2.resize(add,(W,H),interpolation=cv2.INTER_NEAREST),cv2.resize(erase,(W,H),interpolation=cv2.INTER_NEAREST)
def build(observed,seed,confidence,bundle):
    mask=prepare_mask(seed,2);telea=telea_reconstruct(observed,mask);proposal,routes=geometry_t80(observed,mask,telea);n,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8);rows=[]
    for cid in range(1,n):
        if stats[cid,cv2.CC_STAT_AREA]<MIN_COMPONENT_AREA:continue
        comp=labels==cid;row={'component':cid,'x':int(stats[cid,0]),'y':int(stats[cid,1]),'bbox_w':int(stats[cid,2]),'bbox_h':int(stats[cid,3]),'annotation_confidence_mean':float(confidence[comp].mean()),'annotation_confidence_max':float(confidence[comp].max())};row.update(surrounding_features(observed,comp,routes));rows.append(row)
    frame=score(pd.DataFrame(rows),bundle) if rows else pd.DataFrame();return mask,telea,proposal,routes,labels,frame
def current(observed,proposal,labels,decisions):
    out=observed.copy()
    for k,v in decisions.items():
        if v=='accepted':out[labels==int(k)]=proposal[labels==int(k)]
    return out
def archive(folder,gray,confidence,draft,add,erase,seed,mask,telea,proposal,routes,frame,decisions):
    cv2.imwrite(str(folder/'annotated_page.png'),gray);np.save(folder/'yolo_confidence.npy',confidence.astype(np.float16));cv2.imwrite(str(folder/'yolo_draft_mask.png'),draft);cv2.imwrite(str(folder/'review_additions.png'),add);cv2.imwrite(str(folder/'review_erasures.png'),erase);cv2.imwrite(str(folder/'reviewed_authoritative_mask.png'),seed);cv2.imwrite(str(folder/'prepared_dilation2_mask.png'),mask);cv2.imwrite(str(folder/'telea.png'),telea);cv2.imwrite(str(folder/'geometry_t80.png'),proposal)
    np.savez_compressed(folder/'routing_maps.npz',**{k:v.astype(np.uint8) for k,v in routes.items() if k in ['preserve','remove','ambiguous','horizontal','vertical','compact']});frame.to_csv(folder/'component_evidence.csv',index=False);(folder/'review_audit.json').write_text(json.dumps({'formula':'(draft OR additions) AND NOT erasures','decisions':decisions},indent=2))
    z=folder/'training_and_audit_artefacts.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
        for p in folder.iterdir():
            if p.is_file() and p!=z:q.write(p,p.name)
    return z
