"""Exact extraction of Notebook 06 HITL A V3 generator semantics."""
from pathlib import Path
from datetime import datetime,timezone
import cv2,json,torch
import numpy as np
from .hashing import md5_file

SUPPORTED={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
PREDICTION_CONFIDENCE=.20
TILE_SIZE=640
TILE_STRIDE=320
MODEL_IMAGE_SIZE=640
MINIMUM_DRAFT_COMPONENT_AREA=10  # retained exactly; Notebook 06 declares but does not apply it
FORCE_REBUILD_ALL=False

def files(folder): return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
def pos(n,tile_size=TILE_SIZE,tile_stride=TILE_STRIDE):
    if n<=tile_size:return [0]
    p=list(range(0,n-tile_size+1,tile_stride))
    if p[-1]!=n-tile_size:p.append(n-tile_size)
    return p

def load_model(model_path):
    if not Path(model_path).exists():raise RuntimeError(f'Model not found {model_path}')
    if not torch.cuda.is_available():raise RuntimeError('Select T4 GPU')
    from ultralytics import YOLO
    return YOLO(str(model_path))

def reviewed(review_audit_dir):
    out=set()
    for p in Path(review_audit_dir).glob('Page*.json'):
        try:
            v=json.loads(p.read_text()).get('source_md5')
            if v:out.add(v)
        except:pass
    return out

def confidence_map(gray,model,prediction_confidence=PREDICTION_CONFIDENCE,tile_size=TILE_SIZE,tile_stride=TILE_STRIDE,model_image_size=MODEL_IMAGE_SIZE):
    colour=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);h,w=gray.shape;ph,pw=max(h,tile_size),max(w,tile_size);canvas=np.full((ph,pw,3),255,np.uint8);canvas[:h,:w]=colour;conf=np.zeros((ph,pw),np.float32)
    for y in pos(ph,tile_size,tile_stride):
        for x in pos(pw,tile_size,tile_stride):
            r=model.predict(canvas[y:y+tile_size,x:x+tile_size],conf=prediction_confidence,imgsz=model_image_size,device=0,verbose=False,retina_masks=True)[0]
            if r.masks is None:continue
            for mask,score in zip(r.masks.data.cpu().numpy(),r.boxes.conf.cpu().numpy()):
                mask=cv2.resize(mask,(tile_size,tile_size),interpolation=cv2.INTER_NEAREST)>.5;region=conf[y:y+tile_size,x:x+tile_size];region[mask&(score>region)]=score
    return conf[:h,:w]

def generate_all(source_dir,model_path,draft_mask_dir,confidence_dir,draft_audit_dir,review_audit_dir,force_rebuild_all=FORCE_REBUILD_ALL):
    source_dir=Path(source_dir);draft_mask_dir=Path(draft_mask_dir);confidence_dir=Path(confidence_dir);draft_audit_dir=Path(draft_audit_dir);review_audit_dir=Path(review_audit_dir)
    for f in [source_dir,draft_mask_dir,confidence_dir,draft_audit_dir,review_audit_dir]:f.mkdir(parents=True,exist_ok=True)
    model=load_model(model_path);done=reviewed(review_audit_dir);model_hash=md5_file(model_path);generated=skipped_reviewed=skipped_current=0
    for source in files(source_dir):
        sh=md5_file(source)
        if sh in done:skipped_reviewed+=1;continue
        mask_path=draft_mask_dir/f'{source.stem}_draft.png';conf_path=confidence_dir/f'{source.stem}_confidence.npy';audit_path=draft_audit_dir/f'{source.stem}.json';old=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        current=(not force_rebuild_all and mask_path.exists() and conf_path.exists() and old.get('source_md5')==sh and old.get('model_md5')==model_hash and old.get('prediction_confidence')==PREDICTION_CONFIDENCE)
        if current:skipped_current+=1;continue
        gray=cv2.imread(str(source),0)
        if gray is None:continue
        conf=confidence_map(gray,model);binary=(conf>=PREDICTION_CONFIDENCE).astype(np.uint8)*255;np.save(conf_path,conf.astype(np.float16));cv2.imwrite(str(mask_path),binary)
        audit_path.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'source_file':str(source),'source_md5':sh,'model_path':str(model_path),'model_md5':model_hash,'prediction_confidence':PREDICTION_CONFIDENCE,'draft_pixels':int(np.sum(binary>0)),'confidence_map':str(conf_path),'draft_mask':str(mask_path),'shape':list(gray.shape)},indent=2));generated+=1
    return {'generated':generated,'skipped_reviewed':skipped_reviewed,'skipped_current':skipped_current}
