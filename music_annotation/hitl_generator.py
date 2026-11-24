import cv2,numpy as np
CONFIDENCE_THRESHOLD=.20; TILE_SIZE=640; TILE_STRIDE=320; MODEL_IMAGE_SIZE=640
def positions(n):
    if n<=TILE_SIZE:return [0]
    p=list(range(0,n-TILE_SIZE+1,TILE_STRIDE))
    if p[-1]!=n-TILE_SIZE:p.append(n-TILE_SIZE)
    return p
def load_model(path):
    from ultralytics import YOLO
    return YOLO(str(path))
def confidence_map(gray,model,device='cpu'):
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);h,w=gray.shape;ph,pw=max(h,TILE_SIZE),max(w,TILE_SIZE)
    canvas=np.full((ph,pw,3),255,np.uint8);canvas[:h,:w]=rgb;out=np.zeros((ph,pw),np.float32)
    for y in positions(ph):
        for x in positions(pw):
            result=model.predict(canvas[y:y+TILE_SIZE,x:x+TILE_SIZE],conf=CONFIDENCE_THRESHOLD,imgsz=MODEL_IMAGE_SIZE,device=device,verbose=False,retina_masks=True)[0]
            if result.masks is None:continue
            for mask,score in zip(result.masks.data.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                mask=cv2.resize(mask,(TILE_SIZE,TILE_SIZE),interpolation=cv2.INTER_NEAREST)>.5
                region=out[y:y+TILE_SIZE,x:x+TILE_SIZE]; region[mask&(score>region)]=score
    return out[:h,:w]
def draft_from_confidence(confidence):return (confidence>=CONFIDENCE_THRESHOLD).astype(np.uint8)*255
