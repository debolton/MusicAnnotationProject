import json, hashlib
from pathlib import Path
import cv2, joblib, numpy as np, torch
from torch import nn

class TopologyCNN(nn.Module):
    def __init__(self,nc=11):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(nc,24,3,padding=1),nn.ReLU(),nn.BatchNorm2d(24),nn.MaxPool2d(2),nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.MaxPool2d(2),nn.Conv2d(48,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(64,2)
    def forward(self,x): return self.head(self.net(x).flatten(1))

def load_bundle(bundle_dir,device=None):
    bundle=Path(bundle_dir); policy=json.loads((bundle/'policy.json').read_text()); schema=json.loads((bundle/'channel_schema.json').read_text()); config=json.loads((bundle/'model_config.json').read_text()); device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); model=TopologyCNN(len(schema['channels'])).to(device); model.load_state_dict(torch.load(bundle/'model.pt',map_location=device)); model.eval(); calibrator=joblib.load(bundle/'isotonic_calibrator.joblib'); return model,calibrator,policy,schema,device

def make_padded(channels,schema):
    half=int(schema['patch'])//2; out={}
    for name in schema['channels']:
        if name not in channels: raise RuntimeError(f'Missing channel: {name}')
        a=np.asarray(channels[name]);
        if a.ndim!=2: raise RuntimeError(f'{name}: expected 2D array')
        v=a.astype(np.float32)
        if name in ('obs','norm'): v/=255.0
        if not np.isfinite(v).all(): raise RuntimeError(f'{name}: non-finite values')
        out[name]=np.pad(v,half,mode='reflect')
    shapes={np.asarray(channels[n]).shape for n in schema['channels']}
    if len(shapes)!=1: raise RuntimeError(f'Channel shape mismatch: {shapes}')
    return out

def score_coordinates(model,calibrator,padded,schema,ys,xs,device,batch=512):
    ys=np.asarray(ys,dtype=np.int32); xs=np.asarray(xs,dtype=np.int32); patch=int(schema['patch']); probs=[]
    with torch.no_grad():
        for start in range(0,len(ys),batch):
            stop=min(len(ys),start+batch); tensors=[]
            for y,x in zip(ys[start:stop],xs[start:stop]): tensors.append(np.stack([padded[n][y:y+patch,x:x+patch] for n in schema['channels']]))
            xb=torch.from_numpy(np.asarray(tensors,dtype=np.float32)).to(device); raw=torch.softmax(model(xb),1)[:,1].cpu().numpy(); probs.extend(calibrator.predict(raw))
    return np.asarray(probs,dtype=np.float64)

def decisions(prob,policy,shape,ys,xs):
    p=np.asarray(prob,dtype=np.float64); q=1-p; out=np.full(len(p),'UNCERTAIN',object); out[q>=policy['annotation_threshold']]='ANNOTATION'; out[(p>=policy['print_threshold'])&((p-q)>=policy['print_margin'])]='PRINT'
    provisional=out.copy(); pm=np.zeros(shape,np.uint8); idx=np.where(provisional=='PRINT')[0]; pm[np.asarray(ys)[idx],np.asarray(xs)[idx]]=1; n,lab,stats,_=cv2.connectedComponentsWithStats(pm,8)
    for cid in range(1,n):
        if stats[cid,cv2.CC_STAT_AREA] < int(policy['minimum_print_run']):
            yy,xx=np.where(lab==cid); keys=set(zip(yy.tolist(),xx.tolist())); drop=[i for i in idx if (int(ys[i]),int(xs[i])) in keys]; out[drop]='UNCERTAIN'
    return provisional,out

def assemble(observed,a1,authoritative_mask,ys,xs,final_decisions):
    observed=np.asarray(observed,dtype=np.uint8); a1=np.asarray(a1,dtype=np.uint8); mask=np.asarray(authoritative_mask)>0
    if not (observed.shape==a1.shape==mask.shape): raise RuntimeError('Observed/A1/mask shape mismatch')
    out=a1.copy(); idx=np.where(np.asarray(final_decisions)=='PRINT')[0]; py=np.asarray(ys)[idx]; px=np.asarray(xs)[idx]; out[py,px]=observed[py,px]
    print_mask=np.zeros(mask.shape,bool); print_mask[py,px]=True
    if np.any(out[~mask]!=observed[~mask]): raise RuntimeError('Outside-mask drift')
    if np.any(out[~print_mask]!=a1[~print_mask]): raise RuntimeError('Non-PRINT A1 regression')
    if np.any((out!=a1)&(out!=observed)): raise RuntimeError('Changed byte not equal observed')
    return out,print_mask
