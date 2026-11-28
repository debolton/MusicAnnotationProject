from datetime import datetime,timezone
from pathlib import Path
import json,traceback,cv2,numpy as np,pandas as pd
from . import v78,policy

STAGES=[
    'VALIDATE_INPUT',
    'SAVE_AUTHORITY',
    'PREPARE_MASK',
    'TELEA',
    'GEOMETRY_T80',
    'SAVE_ROUTING',
    'COMPONENTS',
    'SAVE_PREDICTORS',
    'VERIFY_V77F',
    'SCORE_V77F',
    'SERIALISE_RESPONSE',
    'COMPLETE'
]

def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,list):return [plain(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,)):return None if not np.isfinite(v) else float(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def status_path(d):
    return d/'authorise_status.json'

def write_status(d,stage,state='running',**extra):
    value={
        'version':'HF-03.1',
        'stage':stage,
        'state':state,
        'utc':datetime.now(timezone.utc).isoformat(),
        **plain(extra)
    }
    
    status_path(d).write_text(json.dumps(value,indent=2))
    
    return value

def fail(d,stage,error):
    value=write_status(d,stage,'failed',exception_type=type(error).__name__,exception=str(error),traceback=traceback.format_exc())
    (d/'authorise_error.txt').write_text(value['traceback'])
    return value

def run(d,state,policy_root):
    stage='VALIDATE_INPUT'
    try:
        write_status(d,stage)
        obs=state['gray']
        seed=state['working']
        
        if obs.ndim!=2 or seed.ndim!=2:
            raise RuntimeError(f'Expected 2-D greyscale page and mask, got {obs.shape} and {seed.shape}')
        if obs.shape!=seed.shape:
            raise RuntimeError(f'Observed/mask shape mismatch: {obs.shape} vs {seed.shape}')
        if obs.dtype!=np.uint8:
            obs=np.clip(obs,0,255).astype(np.uint8)
        
        seed=np.where(seed>0,255,0).astype(np.uint8)
        
        stage='SAVE_AUTHORITY'
        write_status(d,stage)
        cv2.imwrite(str(d/'reviewed_authoritative_mask.png'),seed)
        
        stage='PREPARE_MASK'
        write_status(d,stage)
        mask=v78.prepare_mask(seed)
        cv2.imwrite(str(d/'prepared_dilation2_mask.png'),mask)
        
        stage='TELEA'
        write_status(d,stage)
        telea=v78.telea_reconstruct(obs,mask)
        cv2.imwrite(str(d/'telea_reconstruction.png'),telea)
        assert np.array_equal(telea[mask==0],obs[mask==0])
        
        stage='GEOMETRY_T80'
        write_status(d,stage)
        proposal,routes=v78.geometry_t80(obs,mask,telea)
        cv2.imwrite(str(d/'geometry_t80_proposal.png'),proposal)
        assert np.array_equal(proposal[mask==0],obs[mask==0])
        
        stage='SAVE_ROUTING'
        write_status(d,stage)
        
        route_arrays={
            k:routes[k].astype(np.uint8) for k in ['preserve','remove','ambiguous','horizontal','vertical','compact']
        }
        np.savez_compressed(d/'routing_maps.npz',**route_arrays)
        [cv2.imwrite(str(d/f'route_{k}.png'),v*255) for k,v in route_arrays.items()]
        
        stage='COMPONENTS'
        write_status(d,stage)
        labels,frame=v78.components(obs,mask,routes)
        
        stage='SAVE_PREDICTORS'
        write_status(d,stage,components=len(frame))
        frame.to_csv(d/'component_predictors.csv',index=False)
        np.save(d/'component_labels.npy',labels)
        
        stage='VERIFY_V77F'
        write_status(d,stage)
        bundle,integrity=policy.verify_and_load(policy_root)
        (d/'v77f_integrity.json').write_text(json.dumps(plain(integrity),indent=2))
        
        stage='SCORE_V77F'
        write_status(d,stage,policy_available=bundle is not None)
        scored=policy.score(frame,bundle)
        scored.to_csv(d/'components.csv',index=False)
        
        stage='SERIALISE_RESPONSE'
        write_status(d,stage)
        records=plain(scored.to_dict(orient='records'))
        json.dumps(records,allow_nan=False)
        dec={str(int(c)):'unreviewed' for c in scored.component.tolist()}
        (d/'decisions.json').write_text(json.dumps(dec,indent=2))
        np.savez_compressed(d/'build.npz',mask=mask,telea=telea,proposal=proposal,labels=labels,**route_arrays)
        
        write_status(
            d,
            'COMPLETE',
            'passed',
            components=len(records),
            policy_loaded=bundle is not None,
            intermediate_files=[
                'reviewed_authoritative_mask.png',
                'prepared_dilation2_mask.png',
                'telea_reconstruction.png',
                'geometry_t80_proposal.png',
                'routing_maps.npz',
                'component_predictors.csv',
                'components.csv'
            ]
        )
        
        return {'ok':True,'stage':'COMPLETE','components':records,'policy_loaded':bundle is not None,'proposal_url':None}
    
    except Exception as e:
        return {'ok':False,**fail(d,stage,e)}
