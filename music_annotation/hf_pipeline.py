from pathlib import Path
import cv2, numpy as np
from .hitl_generator import load_model,confidence_map,draft_from_confidence
from .v78_engine import prepare_mask,telea_reconstruct,geometry_t80
_MODEL=None

def model_path(root='.'): return Path(root)/'frozen'/'yolo_detector'/'best.pt'
def get_model(root='.'):
    global _MODEL
    p=model_path(root)
    if not p.exists(): raise FileNotFoundError(f'Missing frozen detector: {p}')
    if _MODEL is None: _MODEL=load_model(p)
    return _MODEL

def analyse_page(image, root='.'):
    if image is None: raise ValueError('Upload an annotated page first.')
    arr=np.asarray(image)
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY) if arr.ndim==3 else arr.astype(np.uint8)
    model=get_model(root); conf=confidence_map(gray,model); draft=draft_from_confidence(conf); mask=prepare_mask(draft,2); telea=telea_reconstruct(gray,mask); corrected,routes=geometry_t80(gray,mask,telea)
    heat=cv2.applyColorMap(np.clip(conf*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO); overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR); colour=overlay.copy(); colour[draft>0]=(0,0,255); overlay=cv2.addWeighted(overlay,.60,colour,.40,0)
    return cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB), cv2.cvtColor(heat,cv2.COLOR_BGR2RGB), draft, corrected, f'Draft pixels: {int((draft>0).sum()):,}. Frozen V7.8 proposal generated. Human mask review is still required before treating the mask as authoritative.'
