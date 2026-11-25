from pathlib import Path
import importlib.util, numpy as np
from .canonical_v78 import build_v78
ROOT=Path(__file__).resolve().parents[1]; BUNDLE=ROOT/'frozen/a5_v1'
class A5Operator:
    def __init__(self,device=None):
        p=BUNDLE/'inference_runtime.py'; spec=importlib.util.spec_from_file_location('a5_frozen_runtime',p); self.rt=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.rt)
        required=['load_bundle','make_padded','score_coordinates','decisions','assemble']; missing=[n for n in required if not callable(getattr(self.rt,n,None))]
        if missing: raise RuntimeError(f'A5 frozen runtime API incomplete: {missing}')
        self.model,self.calibrator,self.policy,self.schema,self.device=self.rt.load_bundle(BUNDLE,device=device)
        from .hf043_replay import run_ardross_replay
        self.startup_replay=run_ardross_replay(self)
    def run(self,observed,reviewed_seed,page=None,batch=512):
        v=build_v78(observed,reviewed_seed); padded=self.rt.make_padded(v['channels'],self.schema)
        prob=self.rt.score_coordinates(self.model,self.calibrator,padded,self.schema,v['ys'],v['xs'],self.device,batch=batch)
        provisional,final=self.rt.decisions(prob,self.policy,observed.shape,v['ys'],v['xs'])
        a5,print_mask=self.rt.assemble(observed,v['a1'],v['mask'],v['ys'],v['xs'],final)
        if np.any(a5[~(v['mask']>0)]!=observed[~(v['mask']>0)]): raise RuntimeError('A5 outside-mask drift')
        if np.any(a5[~print_mask]!=v['a1'][~print_mask]): raise RuntimeError('A5 non-PRINT A1 regression')
        if np.any((a5!=v['a1'])&(a5!=observed)): raise RuntimeError('A5 changed byte not observed')
        return {**v,'a5':a5,'p_print':prob,'provisional':provisional,'final':final,'print_mask':print_mask}
