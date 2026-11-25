import importlib.util,numpy as np
from startup_gate import BUNDLE
class A5Adapter:
 def __init__(self,device):
  s=importlib.util.spec_from_file_location('a5rt',BUNDLE/'inference_runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);self.rt=None
  for n in ('load_runtime','load_bundle','load_operator','A5Runtime','FrozenA5Runtime'):
   o=getattr(m,n,None)
   if o is None:continue
   for kw in ({'device':device},{'bundle_root':BUNDLE,'device':device},{}):
    try:self.rt=o(BUNDLE,**kw) if 'bundle_root' not in kw else o(**kw);break
    except TypeError:pass
   if self.rt is not None:break
  if self.rt is None:raise RuntimeError('Unsupported frozen A5 loader; refusing to invent A1')
 def run(self,obs,seed,page):
  f=getattr(self.rt,'run_page',None) or getattr(self.rt,'infer_page',None)
  if f is None:raise RuntimeError('A5 runtime lacks page inference')
  try:r=f(observed=obs,authoritative_seed_mask=seed,page=page)
  except TypeError:raise RuntimeError('A5 runtime cannot construct its own frozen A1; refusing a new A1 implementation')
  def g(*ns):
   for n in ns:
    if isinstance(r,dict) and n in r:return r[n]
    if hasattr(r,n):return getattr(r,n)
  a5,a1,mask=g('a5','corrected','output','a5_page'),g('a1','baseline','a1_page','fallback_page'),g('authoritative_mask','prepared_mask','mask')
  dec,prob=g('decision','decision_map','final_decision'),g('p_print','probability','probability_map')
  if any(x is None for x in (a5,a1,mask)):raise RuntimeError('A5 runtime must return A5, A1 and authority')
  a5,a1,mask=np.asarray(a5),np.asarray(a1),np.asarray(mask)>0
  if a5.shape!=obs.shape or a1.shape!=obs.shape or mask.shape!=obs.shape:raise RuntimeError('A5 shape mismatch')
  if np.any((a5!=a1)&(a5!=obs)) or np.any(a5[~mask]!=obs[~mask]):raise RuntimeError('A5 byte/authority invariant failed')
  if dec is not None:
   d=np.asarray(dec);pm=(d=='PRINT') if d.dtype.kind in 'OUS' else d.astype(bool)
   if np.any(a5[~pm]!=a1[~pm]):raise RuntimeError('A5 non-PRINT differs from A1')
  return a5.astype('uint8'),a1.astype('uint8'),mask,dec,prob,g('audit') or {}
