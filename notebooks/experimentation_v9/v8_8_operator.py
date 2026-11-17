"""Unified V8.8 observed-print / SAFE_REMOVE operator.

Runtime inputs are an observed uint8 grayscale page and an authoritative undilated
seed mask. Clean/truth data is prohibited. The operator verifies both frozen
bundles, prepares the canonical dilation-2 V7.8 context, obtains calibrated
p_print from the V8.2F provider, builds the approved 17 predictors, obtains
calibrated REMOVE probabilities from the 52F bundle, applies the frozen
page-global SAFE_REMOVE rule, and returns M5_REMOVE plus audited page arrays.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import hashlib, importlib.util, json, platform, sys
import cv2, joblib, numpy as np, pandas as pd, torch

MASK_DILATION=2
CROP_MARGIN=48
GEOMETRY_THRESHOLD=80
BAND_WIDTH=4
P_PRINT_BATCH=512
EXPECTED_CHANNELS=['obs','norm','mask','depth','horizontal','vertical','compact','preserve','remove','ambiguous','edge']
EXPECTED_PREDICTORS=['observed_value','normalised_value','paper_value','mask_depth','boundary_distance','distance_to_protected','edge','local_edge_fraction','local_dark_fraction','local_preserve_fraction','local_remove_fraction','local_ambiguous_fraction','local_horizontal_fraction','local_vertical_fraction','local_compact_fraction','p_print','component_id']

def _require(x: bool,msg: str)->None:
    if not bool(x): raise RuntimeError(msg)
def _sha256(path: Path)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def _read_json(path: Path)->dict: return json.loads(path.read_text())
def _verify(path: Path, expected: Optional[str], label: str)->None:
    _require(path.exists(),f'Missing {label}: {path}')
    if expected: _require(_sha256(path)==expected,f'{label} SHA-256 mismatch: {path}')
def _load_module(path: Path):
    spec=importlib.util.spec_from_file_location('v8_2f_p_print_provider',path)
    _require(spec is not None and spec.loader is not None,f'Cannot import provider source: {path}')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def binary(mask): return np.where(np.asarray(mask)>0,255,0).astype(np.uint8)
def prepare_mask(seed,dilation=MASK_DILATION):
    out=binary(seed)
    if dilation>0: out=cv2.dilate(out,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))
    return out
def immutable(original,candidate,mask):
    out=original.copy(); out[mask>0]=candidate[mask>0]
    _require(np.array_equal(out[mask==0],original[mask==0]),'Outside-mask immutability failure.')
    return out
def padded_components(mask):
    n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8); h,w=mask.shape; boxes=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area>=8: boxes.append((max(0,x-CROP_MARGIN),max(0,y-CROP_MARGIN),min(w,x+bw+CROP_MARGIN),min(h,y+bh+CROP_MARGIN)))
    changed=True
    while changed:
        changed=False; merged=[]
        while boxes:
            ax0,ay0,ax1,ay1=boxes.pop(); rest=[]
            for bx0,by0,bx1,by1 in boxes:
                if not (ax1<bx0 or bx1<ax0 or ay1<by0 or by1<ay0):
                    ax0=min(ax0,bx0); ay0=min(ay0,by0); ax1=max(ax1,bx1); ay1=max(ay1,by1); changed=True
                else: rest.append((bx0,by0,bx1,by1))
            boxes=rest; merged.append((ax0,ay0,ax1,ay1))
        boxes=merged
    return boxes
def telea_reconstruct(observed,mask):
    result=observed.copy()
    for x0,y0,x1,y1 in padded_components(mask):
        im=observed[y0:y1,x0:x1]; mk=mask[y0:y1,x0:x1]; cand=cv2.inpaint(im,mk,3,cv2.INPAINT_TELEA); local=result[y0:y1,x0:x1]; local[mk>0]=cand[mk>0]; result[y0:y1,x0:x1]=local
    return immutable(observed,result,mask)
def background_surface(observed,mask,sigma=45.0):
    m=np.asarray(mask)>0; visible=(~m)&(observed>=200); num=cv2.GaussianBlur(observed.astype(np.float32)*visible,(0,0),sigma); den=cv2.GaussianBlur(visible.astype(np.float32),(0,0),sigma); fallback=cv2.GaussianBlur(observed,(0,0),sigma)
    return np.clip(np.where(den>.02,num/np.maximum(den,1e-4),fallback),0,255).astype(np.uint8)
def support_maps(observed,mask):
    dark=(observed<190)&(mask==0); h=cv2.morphologyEx(dark.astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((1,35),np.uint8))>0; v=cv2.morphologyEx(dark.astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((15,1),np.uint8))>0
    n,lab,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8),8); c=np.zeros_like(dark)
    for i in range(1,n):
        x,y,w,hg,a=stats[i]
        if 3<=a<=650 and w<=40 and hg<=40: c[lab==i]=True
    k=np.ones((2*BAND_WIDTH+1,2*BAND_WIDTH+1),np.uint8)
    return cv2.dilate(h.astype(np.uint8),k)>0,cv2.dilate(v.astype(np.uint8),k)>0,cv2.dilate(c.astype(np.uint8),k)>0
def geometry_t80(observed,mask,telea):
    paper=background_surface(observed,mask); normalised=cv2.divide(observed,np.maximum(paper,1),scale=255); horizontal,vertical,compact=support_maps(observed,mask); m=mask>0
    preserve=m&(horizontal|vertical|compact)&(observed<200); remove=m&~preserve&(observed>=GEOMETRY_THRESHOLD)&(normalised>=GEOMETRY_THRESHOLD); ambiguous=m&~preserve&~remove
    result=observed.copy(); result[remove]=paper[remove]; result[ambiguous]=telea[ambiguous]; result[preserve]=observed[preserve]
    return immutable(observed,result,mask),{'preserve':preserve,'remove':remove,'ambiguous':ambiguous,'horizontal':horizontal,'vertical':vertical,'compact':compact,'paper':paper,'normalised':normalised}
def frac(a,k=5): return cv2.blur(np.asarray(a).astype(np.float32),(k,k))
def dist_to(mask):
    mask=np.asarray(mask,bool)
    return cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5) if mask.any() else np.full(mask.shape,float(max(mask.shape)),np.float32)

@dataclass(frozen=True)
class OperatorCapabilities:
    unseen_page_inference: bool=True
    registered_replay: bool=True
    requires_external_decisions: bool=False
    uses_clean_truth: bool=False
    requires_authoritative_seed_mask: bool=True

@dataclass
class PageResult:
    corrected: np.ndarray
    observed: np.ndarray
    authoritative_mask: np.ndarray
    baseline_v78: np.ndarray
    m3_mask: np.ndarray
    m5_remove_mask: np.ndarray
    safe_remove_mask: np.ndarray
    p_print: np.ndarray
    p_remove: np.ndarray
    reason_map: np.ndarray
    features: pd.DataFrame
    audit: Dict[str,Any]

class V88Operator:
    def __init__(self,p_print_bundle: Path,remove_bundle: Path,device: Optional[str]=None):
        self.p_root=Path(p_print_bundle); self.r_root=Path(remove_bundle); self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); self.capabilities=OperatorCapabilities()
        pm=_read_json(self.p_root/'bundle_manifest.json'); pp=_read_json(self.p_root/'provider_policy.json'); rm=_read_json(self.r_root/'bundle_manifest.json'); rp=_read_json(self.r_root/'policy.json'); rs=_read_json(self.r_root/'predictor_schema.json')
        _verify(self.p_root/pm['model'],pm.get('model_sha256'),'p_print model'); _verify(self.p_root/pm['calibrator'],pm.get('calibrator_sha256'),'p_print calibrator'); _verify(self.p_root/'channel_schema.json',pm.get('channel_schema_sha256'),'channel schema'); _verify(self.p_root/'p_print_provider.py',pm.get('provider_source_sha256'),'p_print provider source')
        _verify(self.r_root/rm['estimator'],rm.get('estimator_sha256'),'REMOVE estimator'); _verify(self.r_root/rm['calibrator'],rm.get('calibrator_sha256'),'REMOVE calibrator'); _verify(self.r_root/rm['predictor_schema'],rm.get('predictor_schema_sha256'),'REMOVE schema'); _verify(self.r_root/rm['policy'],rm.get('policy_sha256'),'REMOVE policy')
        channels=_read_json(self.p_root/'channel_schema.json').get('channels'); predictors=rs.get('ordered_predictors')
        _require(channels==EXPECTED_CHANNELS,'Unexpected p_print channel order.'); _require(predictors==EXPECTED_PREDICTORS,'Unexpected REMOVE predictor order.'); _require(pp.get('clean_truth_required_at_inference') is False,'p_print provider is not truth-free.'); _require(rp.get('clean_truth_required_at_inference') is False,'REMOVE provider is not truth-free.'); _require(rp.get('safe_protect_runtime_enabled') is False,'SAFE_PROTECT must remain disabled.')
        self.p_manifest=pm; self.p_policy=pp; self.r_manifest=rm; self.r_policy=rp; self.predictors=predictors
        module=_load_module(self.p_root/'p_print_provider.py'); self.p_model=module.TopologyCNN().to(self.device); state=torch.load(self.p_root/pm['model'],map_location=self.device); self.p_model.load_state_dict(state,strict=True); self.p_model.eval(); self.p_cal=joblib.load(self.p_root/pm['calibrator']); self.r_model=joblib.load(self.r_root/rm['estimator']); self.r_cal=joblib.load(self.r_root/rm['calibrator'])
        self.bundle_id=hashlib.sha256((pm['bundle_id']+'|'+rm['bundle_id']).encode()).hexdigest()
    def _score_p_print(self,observed,mask,routes,ys,xs):
        arrays={'obs':observed,'norm':routes['normalised'],'mask':mask>0,'depth':cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5),'horizontal':routes['horizontal'],'vertical':routes['vertical'],'compact':routes['compact'],'preserve':routes['preserve'],'remove':routes['remove'],'ambiguous':routes['ambiguous'],'edge':cv2.Canny(observed,70,170)>0}
        padded={}
        for k in EXPECTED_CHANNELS:
            a=np.asarray(arrays[k],dtype=np.float32)
            if k in ('obs','norm'): a/=255.0
            padded[k]=np.pad(a,16,mode='reflect')
        out=[]
        with torch.no_grad():
            for s in range(0,len(ys),P_PRINT_BATCH):
                batch=np.stack([np.stack([padded[k][int(y):int(y)+33,int(x):int(x)+33] for k in EXPECTED_CHANNELS]) for y,x in zip(ys[s:s+P_PRINT_BATCH],xs[s:s+P_PRINT_BATCH])]).astype(np.float32)
                out.extend(torch.softmax(self.p_model(torch.from_numpy(batch).to(self.device)),1)[:,1].cpu().numpy())
        raw=np.asarray(out,np.float64); return raw,np.asarray(self.p_cal.predict(raw),np.float64)
    def _build_features(self,page,observed,m3,protected,routes,p_print):
        ys,xs=np.where(m3); edge=cv2.Canny(observed,70,170)>0; depth=cv2.distanceTransform(m3.astype(np.uint8),cv2.DIST_L2,5); boundary=m3&~cv2.erode(m3.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool); local={k:frac(routes[k]) for k in ['preserve','remove','ambiguous','horizontal','vertical','compact']}; local['edge']=frac(edge); local['dark']=frac(observed<200); _,labels,_,_=cv2.connectedComponentsWithStats(m3.astype(np.uint8),8)
        return pd.DataFrame({'page':page,'y':ys,'x':xs,'observed_value':observed[ys,xs],'normalised_value':routes['normalised'][ys,xs],'paper_value':routes['paper'][ys,xs],'mask_depth':depth[ys,xs],'boundary_distance':dist_to(boundary)[ys,xs],'distance_to_protected':dist_to(protected)[ys,xs],'edge':edge[ys,xs].astype(np.uint8),'local_edge_fraction':local['edge'][ys,xs],'local_dark_fraction':local['dark'][ys,xs],'local_preserve_fraction':local['preserve'][ys,xs],'local_remove_fraction':local['remove'][ys,xs],'local_ambiguous_fraction':local['ambiguous'][ys,xs],'local_horizontal_fraction':local['horizontal'][ys,xs],'local_vertical_fraction':local['vertical'][ys,xs],'local_compact_fraction':local['compact'][ys,xs],'p_print':p_print[ys,xs],'component_id':labels[ys,xs].astype(int)})
    def _safe_remove(self,features):
        p=features.p_remove.to_numpy(float); candidate=p>=float(self.r_policy['remove_probability_threshold']); h=int(features.y.max())+1; w=int(features.x.max())+1; mask=np.zeros((h,w),np.uint8); mask[features.y.to_numpy(int)[candidate],features.x.to_numpy(int)[candidate]]=1; n,lab,stats,_=cv2.connectedComponentsWithStats(mask,int(self.r_policy['connectivity'])); selected=np.zeros(len(features),bool); lookup={(int(r.y),int(r.x)):i for i,r in features[['y','x']].iterrows()}
        for cid in range(1,n):
            yy,xx=np.where(lab==cid); ids=[lookup[(int(y),int(x))] for y,x in zip(yy,xx) if (int(y),int(x)) in lookup]
            if not ids: continue
            q=p[ids]
            if len(ids)>=int(self.r_policy['minimum_remove_run']) and float(q.mean())>=float(self.r_policy['remove_region_minimum_mean_probability']) and float((q>=float(self.r_policy['remove_probability_threshold'])).mean())>=float(self.r_policy['remove_region_minimum_pixel_fraction_above_threshold']): selected[ids]=True
        return selected
    def run_page(self,observed,authoritative_seed_mask,page='page')->PageResult:
        observed=np.asarray(observed); seed=np.asarray(authoritative_seed_mask); _require(observed.ndim==2 and observed.dtype==np.uint8,'Observed page must be 2D uint8 grayscale.'); _require(seed.shape==observed.shape,'Observed and seed-mask shapes differ.'); mask=prepare_mask(seed); telea=telea_reconstruct(observed,mask); baseline,routes=geometry_t80(observed,mask,telea)
        elig=(mask>0)&(observed<200)&~routes['preserve']; eys,exs=np.where(elig); p_print=np.full(observed.shape,np.nan,np.float64)
        if len(eys): _,pp=self._score_p_print(observed,mask,routes,eys,exs); p_print[eys,exs]=pp
        # Strict V8.2 PRINT protection is part of the protected evidence used by Notebook 49.
        strict_print=np.zeros(observed.shape,bool)
        if len(eys):
            provisional=pp>=.995; pm=np.zeros(observed.shape,np.uint8); pm[eys[provisional],exs[provisional]]=1; n,lab,stats,_=cv2.connectedComponentsWithStats(pm,8)
            for cid in range(1,n):
                if stats[cid,cv2.CC_STAT_AREA]>=2: strict_print|=lab==cid
        protected=routes['preserve']|strict_print
        # M3 is the incumbent pixels eligible for REMOVE reconsideration: inside authority and not already protected.
        m3=(mask>0)&~protected
        features=self._build_features(page,observed,m3,protected,routes,p_print)
        _require(len(features)>0,'No M3 candidate pixels.')
        X=features[self.predictors].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan); raw_remove=self.r_model.predict_proba(X)[:,1]; p_remove_rows=np.asarray(self.r_cal.predict(raw_remove),np.float64); features['raw_remove_score']=raw_remove; features['p_remove']=p_remove_rows; selected=self._safe_remove(features); features['safe_remove']=selected
        safe=np.zeros(observed.shape,bool); safe[features.y.to_numpy(int)[selected],features.x.to_numpy(int)[selected]]=True; m5_remove=m3&~safe
        # Historical V8.8 assembly contract (Notebook 53 lineage):
        #   1. M5_REMOVE = M3 & ~SAFE_REMOVE
        #   2. reconstruct the residual M5_REMOVE mask from the observed page with Telea
        #   3. replace SAFE_REMOVE pixels with the frozen local paper estimate
        #   4. retain observed bytes everywhere outside M3
        #
        # Do not substitute baseline[safe] here. SAFE_REMOVE has an explicit paper
        # treatment, while the complementary M5_REMOVE region must be reconstructed.
        reconstructed_m5=telea_reconstruct(observed,binary(m5_remove))
        corrected=reconstructed_m5.copy()
        corrected[safe]=routes['paper'][safe]
        _require(np.array_equal(corrected[~m3],observed[~m3]),'Outside-M3 drift.')
        _require(np.array_equal(corrected[mask==0],observed[mask==0]),'Outside-mask drift.')
        _require(np.array_equal(corrected[safe],routes['paper'][safe]),'SAFE_REMOVE paper-treatment mismatch.')
        _require(np.array_equal(corrected[m5_remove],reconstructed_m5[m5_remove]),'M5_REMOVE reconstruction mismatch.')
        p_remove=np.full(observed.shape,np.nan,np.float64); p_remove[features.y.to_numpy(int),features.x.to_numpy(int)]=p_remove_rows; reason=np.zeros(observed.shape,np.uint8); reason[routes['preserve']]=1; reason[strict_print]=2; reason[m3&~safe]=3; reason[safe]=4
        audit={'page':page,'operator_bundle_id':self.bundle_id,'p_print_bundle_id':self.p_manifest['bundle_id'],'remove_bundle_id':self.r_manifest['bundle_id'],'authoritative_mask_pixels':int((mask>0).sum()),'p_print_eligible_pixels':int(len(eys)),'strict_print_pixels':int(strict_print.sum()),'m3_pixels':int(m3.sum()),'safe_remove_pixels':int(safe.sum()),'m5_remove_pixels':int(m5_remove.sum()),'assembly_contract':'historical_m5_telea_then_safe_remove_paper','m5_reconstructed_pixels':int(np.sum(reconstructed_m5[m5_remove]!=observed[m5_remove])),'safe_remove_paper_pixels':int(np.sum(corrected[safe]!=observed[safe])),'outside_m3_drift':int(np.sum(corrected[~m3]!=observed[~m3])),'outside_mask_drift':int(np.sum(corrected[mask==0]!=observed[mask==0])),'uses_clean_truth':False,'software':{'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'platform':platform.platform()}}
        return PageResult(corrected,observed.copy(),mask>0,baseline,m3,m5_remove,safe,p_print,p_remove,reason,features,audit)

def load_operator(p_print_bundle_root,remove_bundle_root,device=None): return V88Operator(Path(p_print_bundle_root),Path(remove_bundle_root),device=device)
def run_page(observed,authoritative_seed_mask,operator:V88Operator,page='page'): return operator.run_page(observed,authoritative_seed_mask,page=page)
