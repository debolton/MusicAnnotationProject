import base64, cv2, numpy as np
PREVIEW_MAX_WIDTH=1180

def preview_size(shape):
    h,w=shape; scale=min(1.0,PREVIEW_MAX_WIDTH/w)
    return max(1,round(w*scale)),max(1,round(h*scale))

def resize(a,size,nearest=False):
    return cv2.resize(a,size,interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

def data_url(a):
    ok,b=cv2.imencode('.jpg',a,[cv2.IMWRITE_JPEG_QUALITY,82])
    if not ok: raise RuntimeError('Preview encoding failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(b).decode()

def png_url(a):
    ok,b=cv2.imencode('.png',a)
    if not ok: raise RuntimeError('Preview encoding failed')
    return 'data:image/png;base64,'+base64.b64encode(b).decode()

def overlay(gray,mask,alpha=.48):
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR); colour=rgb.copy(); colour[mask>0]=(90,90,255)
    return cv2.addWeighted(rgb,1-alpha,colour,alpha,0)

def heatmap(conf): return cv2.applyColorMap(np.clip(conf*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
