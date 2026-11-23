import math
import numpy as np
import pandas as pd

def wilson_upper(errors,n,z=1.96):
    if n==0:return np.nan
    p=errors/n; d=1+z*z/n; c=p+z*z/(2*n); r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c+r)/d

def page_clustered_bootstrap(page_df,resamples=10000,seed=42):
    rng=np.random.default_rng(seed); unique=page_df.page.to_numpy(); indexed=page_df.set_index('page'); boot=[]
    for _ in range(resamples):
        q=indexed.loc[rng.choice(unique,size=len(unique),replace=True)]; bt=int(q.true_print.sum()); bf=int(q.false_annotation.sum()); bn=bt+bf; bc=int(q.catastrophic_components.sum()); boot.append((bt-5*bf-500*bc,bf/max(bt,1),bf/max(bn,1)))
    b=np.asarray(boot,float); return {'bootstrap_utility_95_low':float(np.quantile(b[:,0],.025)),'bootstrap_utility_95_high':float(np.quantile(b[:,0],.975)),'bootstrap_false_true_95_high':float(np.quantile(b[:,1],.975)),'bootstrap_false_preservation_95_high':float(np.quantile(b[:,2],.975))}

def catastrophic(false_annotation,true_print): return bool(false_annotation>=50 or (false_annotation/max(true_print,1)>1 and false_annotation>=20))
