import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from .constants import CHANNELS,PATCH,HALF,SEED

def printed_ink(clean): return cv2.adaptiveThreshold(clean,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,15)>0

def truth_maps(clean):
    ink=printed_ink(clean); k=np.ones((3,3),np.uint8); near=cv2.dilate(ink.astype(np.uint8),k)>0; eroded=cv2.erode(ink.astype(np.uint8),k)>0; return ink,near,np.logical_xor(near,eroded)

def build_padded_channels(runtime,channels=CHANNELS,half=HALF):
    padded={}
    for page,z in runtime.items():
        padded[page]={}
        for key in channels:
            a=z[key].astype(np.float32)
            if key in ['obs','norm']: a/=255.0
            padded[page][key]=np.pad(a,half,mode='reflect')
    return padded

def get_patch(padded,page,y,x,channels=CHANNELS,patch=PATCH):
    return np.stack([padded[page][k][y:y+patch,x:x+patch] for k in channels]).astype(np.float32)

class TopologyCNN(nn.Module):
    def __init__(self,nc=len(CHANNELS)):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(nc,24,3,padding=1),nn.ReLU(),nn.BatchNorm2d(24),nn.MaxPool2d(2),nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.MaxPool2d(2),nn.Conv2d(48,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(64,2)
    def forward(self,x): return self.head(self.net(x).flatten(1))

class ArrayPatchDataset(Dataset):
    def __init__(self,idx,pids,ys,xs,truth,page_list,padded,channels=CHANNELS,patch=PATCH): self.idx=np.asarray(idx); self.pids=pids; self.ys=ys; self.xs=xs; self.truth=truth; self.page_list=page_list; self.padded=padded; self.channels=channels; self.patch=patch
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        j=int(self.idx[i]); page=self.page_list[int(self.pids[j])]; patch=get_patch(self.padded,page,self.ys[j],self.xs[j],self.channels,self.patch); return torch.from_numpy(patch),torch.tensor(int(self.truth[j]),dtype=torch.long)

def worker_init(worker_id,seed=SEED): random.seed(seed+worker_id); np.random.seed(seed+worker_id)

def make_loader(idx,pids,ys,xs,truth,page_list,padded,batch,shuffle,seed=SEED,channels=CHANNELS,patch=PATCH):
    g=torch.Generator(); g.manual_seed(seed); ds=ArrayPatchDataset(idx,pids,ys,xs,truth,page_list,padded,channels,patch); return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=2,pin_memory=True,generator=g,worker_init_fn=lambda w:worker_init(w,seed),persistent_workers=False)
