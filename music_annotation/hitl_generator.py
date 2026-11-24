import cv2,numpy as np
CONF=.20;TILE=640;STRIDE=320;IMGSZ=640
def positions(n):
 if n<=TILE:return [0]
 p=list(range(0,n-TILE+1,STRIDE))
 if p[-1]!=n-TILE:p.append(n-TILE)
 return p
def load_model(path):
 from ultralytics import YOLO
 return YOLO(str(path))
def confidence_map(gray,model,device='cpu'):
 rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);h,w=gray.shape;ph,pw=max(h,TILE),max(w,TILE);canvas=np.full((ph,pw,3),255,np.uint8);canvas[:h,:w]=rgb;out=np.zeros((ph,pw),np.float32)
 for y in positions(ph):
  for x in positions(pw):
   r=model.predict(canvas[y:y+TILE,x:x+TILE],conf=CONF,imgsz=IMGSZ,device=device,verbose=False,retina_masks=True)[0]
   if r.masks is None:continue
   for m,s in zip(r.masks.data.cpu().numpy(),r.boxes.conf.cpu().numpy()):
    m=cv2.resize(m,(TILE,TILE),interpolation=cv2.INTER_NEAREST)>.5;q=out[y:y+TILE,x:x+TILE];q[m&(s>q)]=s
 return out[:h,:w]
def draft_from_confidence(c):return (c>=CONF).astype(np.uint8)*255
