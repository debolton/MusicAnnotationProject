from pathlib import Path
import json,cv2,numpy as np,pandas as pd,gradio as gr
from music_annotation.hitl_generator import load_model,confidence_map,draft_from_confidence
from music_annotation.v77_policy import load_policy
from music_annotation.session_store import create,load,save,write_json,folder
from music_annotation.backend import package,decode,build,current,psize,resize,archive
from music_annotation.ipad_canvas import HTML,JS,CSS
ROOT=Path(__file__).parent;YOLO=ROOT/'frozen/yolo_detector/best.pt';POLICY=ROOT/'frozen/structural_policy_v7_7f/artefacts';_m=_b=None
def models():
 global _m,_b
 if _m is None:_m=load_model(YOLO)
 if _b is None:_b=load_policy(POLICY)
 return _m,_b
def generate(image):
 if image is None:raise gr.Error('Upload a page.')
 a=np.asarray(image).astype(np.uint8);g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a;c=confidence_map(g,models()[0],'cpu');d=draft_from_confidence(c);sid=create(g,c,d);return json.dumps(package(g,d,c,'uploaded-page')),sid,'YOLO draft complete. Open Mask review.'
def auth(command,sid):
 q=json.loads(command);z=load(sid);g=z['gray'];c=z['conf'].astype(np.float32);d=z['draft'];sz=psize(g.shape);a=decode(q.get('add'),(sz[1],sz[0]));e=decode(q.get('erase'),(sz[1],sz[0]));a=cv2.resize(a,(g.shape[1],g.shape[0]),interpolation=cv2.INTER_NEAREST);e=cv2.resize(e,(g.shape[1],g.shape[0]),interpolation=cv2.INTER_NEAREST);seed=(((d>0)|(a>0))&~(e>0)).astype(np.uint8)*255;m,p,l,f=build(g,seed,c,models()[1]);save(sid,'build',seed=seed,mask=m,proposal=p,labels=l);f.to_csv(folder(sid)/'components.csv',index=False);dec={str(int(x)):'unreviewed' for x in f.component};write_json(sid,'decisions',dec);art=archive(folder(sid),g,d,c,seed,m,p,dec);ids=f.component.astype(int).tolist();tab=f[['component','pilot_category','prob_structural_safe','annotation_confidence_mean','area','preserve_fraction','remove_fraction','ambiguous_fraction']];return resize(p,psize(g.shape)),tab,gr.update(choices=ids,value=ids[0] if ids else None),dec,f'Authorised. ADD {int((a>0).sum()):,}; ERASE {int((e>0).sum()):,}.',str(art),''
def pair(sid,cid):
 if cid is None:return None,''
 z=load(sid);b=load(sid,'build');f=pd.read_csv(folder(sid)/'components.csv');r=f[f.component.astype(int)==int(cid)].iloc[0];M=100;y0=max(0,int(r.y)-M);x0=max(0,int(r.x)-M);y1=min(z['gray'].shape[0],int(r.y+r.bbox_h)+M);x1=min(z['gray'].shape[1],int(r.x+r.bbox_w)+M);a=cv2.cvtColor(z['gray'][y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);p=cv2.cvtColor(b['proposal'][y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);return np.hstack([a,np.full((a.shape[0],8,3),235,np.uint8),p]),f'C{cid} | {r.pilot_category.upper()} | p_safe={r.prob_structural_safe:.4f}'
def decide(sid,cid,status,dec):
 dec=dict(dec or {});dec[str(int(cid))]=status;write_json(sid,'decisions',dec);z=load(sid);b=load(sid,'build');return resize(current(z['gray'],b['proposal'],b['labels'],dec),psize(z['gray'].shape)),dec,f'C{cid}: {status}'
def finalise(sid,dec):
 z=load(sid);b=load(sid,'build');f=pd.read_csv(folder(sid)/'components.csv');missing=[int(x) for x in f.component if dec.get(str(int(x))) not in ('accepted','left_unchanged')]
 if missing:raise gr.Error(f'{len(missing)} component(s) remain unreviewed/deferred.')
 out=current(z['gray'],b['proposal'],b['labels'],dec);assert np.array_equal(out[b['mask']==0],z['gray'][b['mask']==0]);cv2.imwrite(str(folder(sid)/'approved_final_page.png'),out);return out,'Approved. Outside-mask immutability passed.'
with gr.Blocks(title='Music Annotation HF-02.7') as demo:
 gr.Markdown('# Music Annotation Removal Assistant - HF-02.7')
 sid=gr.State('');dec=gr.State({})
 with gr.Tab('1 Upload'):
  inp=gr.Image(type='numpy');go=gr.Button('Generate YOLO draft',variant='primary');status=gr.Textbox()
 with gr.Tab('2 Mask review'):
  canvas=gr.HTML(HTML,js_on_load=JS);pkg=gr.Textbox(elem_id='page-package',elem_classes=['bridge-offscreen']);bridge=gr.Textbox(elem_id='action-bridge',elem_classes=['bridge-offscreen']);mask_status=gr.Textbox();assets=gr.File(label='Training and audit artefacts')
 with gr.Tab('3 Component review'):
  proposal=gr.Image(height=500);table=gr.Dataframe(interactive=False);comp=gr.Dropdown();crop=gr.Image(height=420);info=gr.Textbox();cur=gr.Image(height=500);ds=gr.Textbox()
  with gr.Row():accept=gr.Button('Accept correction');leave=gr.Button('Leave unchanged');defer=gr.Button('Defer')
 with gr.Tab('4 Final approval'):
  approve=gr.Button('Approve final page');final=gr.Image();fs=gr.Textbox()
 ev=go.click(generate,inp,[pkg,sid,status]);ev.then(None,pkg,None,js='(raw)=>{if(raw&&window.hitlIngest)window.hitlIngest(raw);return []}')
 bridge.change(auth,[bridge,sid],[proposal,table,comp,dec,mask_status,assets,bridge]);comp.change(pair,[sid,comp],[crop,info]);accept.click(lambda s,c,d:decide(s,c,'accepted',d),[sid,comp,dec],[cur,dec,ds]);leave.click(lambda s,c,d:decide(s,c,'left_unchanged',d),[sid,comp,dec],[cur,dec,ds]);defer.click(lambda s,c,d:decide(s,c,'deferred',d),[sid,comp,dec],[cur,dec,ds]);approve.click(finalise,[sid,dec],[final,fs])
if __name__=='__main__':demo.launch(css=CSS)
