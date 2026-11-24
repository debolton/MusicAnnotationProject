import cv2, numpy as np
PREVIEW_MAX_WIDTH = 1200

def preview_size(shape):
    h,w = shape
    scale = min(1.0, PREVIEW_MAX_WIDTH / w)
    return max(1,round(w*scale)), max(1,round(h*scale))

def resize(a, size, nearest=False):
    return cv2.resize(a, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

def overlay(gray, mask, colour=(255,105,105), alpha=.48):
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    layer=rgb.copy(); layer[mask>0]=colour
    return cv2.addWeighted(rgb,1-alpha,layer,alpha,0)

def editor_payload(gray):
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    return {'background':rgb,'layers':[],'composite':rgb}

def paint_mask(value, shape):
    if not isinstance(value,dict): return np.zeros(shape,np.uint8)
    masks=[]
    for layer in value.get('layers') or []:
        a=np.asarray(layer)
        if a.ndim==3 and a.shape[2]>=4: m=a[:,:,3]>0
        elif a.ndim==3: m=np.any(a[:,:,:3]<250,axis=2)
        else: m=a>0
        masks.append(m)
    if not masks: return np.zeros(shape,np.uint8)
    m=np.logical_or.reduce(masks).astype(np.uint8)*255
    if m.shape!=shape: m=cv2.resize(m,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return m
