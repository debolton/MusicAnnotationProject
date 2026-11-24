from pathlib import Path
import json,cv2,numpy as np,pandas as pd,gradio as gr
from music_annotation.hitl_generator import load_model,confidence_map,draft_from_confidence
from music_annotation.v77_policy import load_policy
from music_annotation.session_store import create,load,save,write_json
from music_annotation.review_backend import package,decode_url,build,current,component_pair,preview_size,resize
from music_annotation.notebook07_canvas import CANVAS_HTML,CANVAS_JS,CSS
ROOT=Path(__file__).parent;YOLO=ROOT/'frozen/yolo_detector/best.pt';POLICY=ROOT/'frozen/structural_policy_v7_7f/artefacts';_model=_bundle=None

def models():
 global _model,_bundle
 if _model is None:
  if not YOLO.exists():raise RuntimeError(f'Missing YOLO checkpoint: {YOLO}')
  _model=load_model(YOLO)
 if _bundle is None:_bundle=load_policy(POLICY)
 return _model,_bundle

def generate(image):
 if image is None:raise gr.Error('Upload an annotated page.')
 a=np.asarray(image).astype(np.uint8);gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a
 conf=confidence_map(gray,models()[0],device='cpu');draft=draft_from_confidence(conf);sid=create(gray,conf,draft)
 return json.dumps(package(gray,draft,conf,'uploaded-page')),sid,'YOLO draft complete. Open Mask review.'
def authorise(add_url,erase_url,sid):
 z=load(sid);gray=z['gray'];conf=z['conf'].astype(np.float32);draft=z['draft'];sz=preview_size(gray.shape)
 add_s=decode_url(add_url,(sz[1],sz[0]));erase_s=decode_url(erase_url,(sz[1],sz[0]))
 add=cv2.resize(add_s,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST);erase=cv2.resize(erase_s,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
 seed=(((draft>0)|(add>0))&~(erase>0)).astype(np.uint8)*255;mask,proposal,labels,frame=build(gray,seed,conf,models()[1])
 save(sid,'build',seed=seed,mask=mask,proposal=proposal,labels=labels);frame.to_csv(ROOT/'runtime_sessions'/sid/'components.csv',index=False)
 decisions={str(int(c)):'unreviewed' for c in frame.component.tolist()};write_json(sid,'decisions',decisions);ids=frame.component.astype(int).tolist()
 table=frame[['component','pilot_category','prob_structural_safe','annotation_confidence_mean','area','preserve_fraction','remove_fraction','ambiguous_fraction']]
 return resize(proposal,preview_size(gray.shape)),table,gr.update(choices=ids,value=ids[0] if ids else None),decisions,f'Authorised. ADD {int((add>0).sum()):,} px; ERASE {int((erase>0).sum()):,} px.'
def load_component(sid,cid):
 if cid is None:return None,''
 z=load(sid);b=load(sid,'build');frame=pd.read_csv(ROOT/'runtime_sessions'/sid/'components.csv');return component_pair(z['gray'],b['proposal'],b['labels'],frame,int(cid))
def decide(sid,cid,status,decisions):
 if cid is None:raise gr.Error('Select a component.')
 decisions=dict(decisions or {});decisions[str(int(cid))]=status;write_json(sid,'decisions',decisions);z=load(sid);b=load(sid,'build');out=current(z['gray'],b['proposal'],b['labels'],decisions);return resize(out,preview_size(z['gray'].shape)),decisions,f'C{int(cid)}: {status}'
def finalise(sid,decisions):
 z=load(sid);b=load(sid,'build');frame=pd.read_csv(ROOT/'runtime_sessions'/sid/'components.csv');ids=frame.component.astype(int).tolist();missing=[c for c in ids if decisions.get(str(c)) not in ('accepted','left_unchanged')]
 if missing:raise gr.Error(f'{len(missing)} component(s) remain unreviewed/deferred.')
 out=current(z['gray'],b['proposal'],b['labels'],decisions);assert np.array_equal(out[b['mask']==0],z['gray'][b['mask']==0]);return out,'Final page approved. Outside-mask immutability passed.'
with gr.Blocks(title='Music Annotation HF-02.5') as demo:
 gr.Markdown('# Music Annotation Removal Assistant - HF-02.5')
 sid=gr.State('');decisions=gr.State({})
 with gr.Tab('1 Upload',id='upload'):
  inp=gr.Image(type='numpy',label='Annotated page');go=gr.Button('Generate YOLO draft',variant='primary');status=gr.Textbox(interactive=False)
 with gr.Tab('2 Mask review',id='mask'):
  canvas=gr.HTML(CANVAS_HTML,elem_id='canvas-shell',js_on_load=CANVAS_JS)
  page_package=gr.Textbox(value='',elem_id='page-package',elem_classes=['bridge-offscreen'])
  action_bridge=gr.Textbox(value='',elem_id='action-bridge',elem_classes=['bridge-offscreen'])
  add_mask=gr.Textbox(value='',elem_id='add-mask-out',elem_classes=['bridge-offscreen']);erase_mask=gr.Textbox(value='',elem_id='erase-mask-out',elem_classes=['bridge-offscreen'])
  mask_status=gr.Textbox(interactive=False)
 with gr.Tab('3 Component review',id='components'):
  proposal=gr.Image(label='V7.8 proposal',height=600);table=gr.Dataframe(interactive=False);component=gr.Dropdown(label='Component');pair=gr.Image(label='ORIGINAL | PROPOSED',height=520);info=gr.Textbox(interactive=False);current_img=gr.Image(label='Current reviewed result',height=600);decision_status=gr.Textbox(interactive=False)
  with gr.Row():accept=gr.Button('Accept correction',variant='primary');leave=gr.Button('Leave unchanged');defer=gr.Button('Defer')
 with gr.Tab('4 Final approval',id='final'):
  approve=gr.Button('Approve final page',variant='primary');final=gr.Image(label='Approved full-resolution page');final_status=gr.Textbox(interactive=False)
 event=go.click(generate,inp,[page_package,sid,status])
 event.then(None,page_package,None,js='(raw)=>{if(raw&&window.hitlIngest){window.hitlIngest(raw)};return []}')
 def handle_action(command_text,sid):
  if not command_text:return gr.update(),gr.update(),gr.update(),gr.update(),gr.update(),gr.update()
  command=json.loads(command_text)
  if command.get('action')!='save':raise gr.Error('Unknown reviewer action.')
  return (*authorise(command.get('add',''),command.get('erase',''),sid), '')
 action_bridge.change(handle_action,[action_bridge,sid],[proposal,table,component,decisions,mask_status,action_bridge],show_progress='minimal')
 component.change(load_component,[sid,component],[pair,info]);accept.click(lambda s,c,d:decide(s,c,'accepted',d),[sid,component,decisions],[current_img,decisions,decision_status]);leave.click(lambda s,c,d:decide(s,c,'left_unchanged',d),[sid,component,decisions],[current_img,decisions,decision_status]);defer.click(lambda s,c,d:decide(s,c,'deferred',d),[sid,component,decisions],[current_img,decisions,decision_status]);approve.click(finalise,[sid,decisions],[final,final_status])
if __name__=='__main__':demo.launch(css=CSS, share=True)
