import cv2
import numpy as np

def effective_print_threshold(print_threshold,print_margin): return max(float(print_threshold),(1.0+float(print_margin))/2.0)

def provisional_decision(prob,print_threshold=.995,print_margin=.990,annotation_threshold=.995):
    p=np.asarray(prob,dtype=np.float64); q=1.0-p; out=np.full(len(p),'UNCERTAIN',object); out[q>=annotation_threshold]='ANNOTATION'; out[(p>=print_threshold)&((p-q)>=print_margin)]='PRINT'; return out

def apply_run_filter(frame,provisional,enabled=True,min_run=2,shape_by_page=None):
    final=np.asarray(provisional,dtype=object).copy(); demoted=0
    if not enabled or min_run<=1:return final,demoted
    for page,g in frame.groupby('page',sort=False):
        idx=g.index.to_numpy(); h,w=shape_by_page[page] if shape_by_page is not None else (int(g.y.max())+1,int(g.x.max())+1); pm=np.zeros((h,w),np.uint8); pidx=idx[provisional[idx]=='PRINT']
        if len(pidx)==0:continue
        pm[frame.loc[pidx,'y'].to_numpy(),frame.loc[pidx,'x'].to_numpy()]=1; n,lab,stats,_=cv2.connectedComponentsWithStats(pm,8)
        for cid in range(1,n):
            if int(stats[cid,cv2.CC_STAT_AREA])<min_run:
                ys,xs=np.where(lab==cid); keys={(int(y),int(x)) for y,x in zip(ys,xs)}; drop=[i for i in pidx if (int(frame.at[i,'y']),int(frame.at[i,'x'])) in keys]; final[drop]='UNCERTAIN'; demoted+=len(drop)
    return final,demoted

def assemble_page(observed,baseline,authoritative_mask,ys,xs,decisions):
    out=baseline.copy(); pm=np.zeros(authoritative_mask.shape,bool); chosen=np.asarray(decisions)=='PRINT'; pm[np.asarray(ys)[chosen],np.asarray(xs)[chosen]]=True; out[pm]=observed[pm]
    outside=int(np.sum(out[~authoritative_mask]!=observed[~authoritative_mask])); nonprint=int(np.sum(out[~pm]!=baseline[~pm])); wrong=int(np.sum((out!=baseline)&(out!=observed)))
    if outside or nonprint or wrong: raise RuntimeError(f'page contract failed outside={outside}, nonprint={nonprint}, changed_not_observed={wrong}')
    return out,pm,{'outside_mask_drift':outside,'non_print_v78_regression':nonprint,'changed_not_equal_observed':wrong}
