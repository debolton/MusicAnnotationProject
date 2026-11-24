import cv2, numpy as np, torch
from torch import nn
CHANNELS=['obs', 'norm', 'mask', 'depth', 'horizontal', 'vertical', 'compact', 'preserve', 'remove', 'ambiguous', 'edge']
PATCH=33
HALF=16
class TopologyCNN(nn.Module):
 def __init__(self,nc=len(CHANNELS)):
  super().__init__(); self.net=nn.Sequential(nn.Conv2d(nc,24,3,padding=1),nn.ReLU(),nn.BatchNorm2d(24),nn.MaxPool2d(2),nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.MaxPool2d(2),nn.Conv2d(48,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(64,2)
 def forward(self,x): return self.head(self.net(x).flatten(1))
def background_surface(obs,mask,sigma=45.0):
 vis=(~mask)&(obs>=200); num=cv2.GaussianBlur(obs.astype(np.float32)*vis,(0,0),sigma); den=cv2.GaussianBlur(vis.astype(np.float32),(0,0),sigma); return np.clip(np.where(den>.02,num/np.maximum(den,1e-4),cv2.GaussianBlur(obs,(0,0),sigma)),0,255).astype(np.uint8)
def build_channels(obs,mask,routes):
 paper=background_surface(obs,mask); arrays={'obs':obs,'norm':cv2.divide(obs,np.maximum(paper,1),scale=255),'mask':mask,'depth':cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5),'horizontal':routes['horizontal'],'vertical':routes['vertical'],'compact':routes['compact'],'preserve':routes['preserve'],'remove':routes['remove'],'ambiguous':routes['ambiguous'],'edge':cv2.Canny(obs,70,170)>0}
 return {k:np.pad((v.astype(np.float32)/255.0 if k in ('obs','norm') else v.astype(np.float32)),HALF,mode='reflect') for k,v in arrays.items()}
def extract(padded,ys,xs): return np.stack([np.stack([padded[k][int(y):int(y)+PATCH,int(x):int(x)+PATCH] for k in CHANNELS]) for y,x in zip(ys,xs)]).astype(np.float32)
def score(model,calibrator,padded,ys,xs,device,batch=512):
 model.eval(); out=[]
 with torch.no_grad():
  for s in range(0,len(ys),batch): out.extend(torch.softmax(model(torch.from_numpy(extract(padded,ys[s:s+batch],xs[s:s+batch])).to(device)),1)[:,1].cpu().numpy())
 raw=np.asarray(out,dtype=np.float64); return raw,np.asarray(calibrator.predict(raw),dtype=np.float64)
