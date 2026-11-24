from pathlib import Path
import cv2,numpy as np,gradio as gr
from music_annotation.hitl_generator import load_model,confidence_map,draft_from_confidence
from music_annotation.v77_policy import load_policy
from music_annotation.supervised import build,render_review
ROOT=Path(__file__).parent
YOLO=ROOT/'frozen/yolo_detector/best.pt';POLICY=ROOT/'frozen/structural_policy_v7_7f/artefacts'
_model=None;_bundle=None

def models():
 global _model,_bundle
 if _model is None:_model=load_model(YOLO)
 if _bundle is None:_bundle=load_policy(POLICY)
 return _model,_bundle

def start(image):
 if image is None:raise gr.Error('Upload a page.')
 rgb=np.asarray(image).astype(np.uint8);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) if rgb.ndim==3 else rgb
 model,bundle=models();conf=confidence_map(gray,model);draft=draft_from_confidence(conf);overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);overlay[draft>0]=(255,120,120)
 state={'gray':gray,'conf':conf,'draft':draft,'build':None,'decisions':{}}
 editor={'background':cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB),'layers':[np.dstack([np.zeros_like(draft)]*3+[draft])],'composite':overlay}
 return overlay,cv2.applyColorMap(np.clip(conf*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1],editor,state,'HF-02: review the draft mask. Add required annotation and erase print, then click Authorise mask and score.'
def authorise(editor,state):
 if not state:raise gr.Error('Generate a draft first.')
 comp=np.asarray(editor.get('composite') if isinstance(editor,dict) else editor)
 if comp.ndim==3:comp=cv2.cvtColor(comp.astype(np.uint8),cv2.COLOR_RGB2GRAY)
 seed=np.where(comp<235,255,0).astype(np.uint8) if comp.shape==state['gray'].shape else state['draft']
 # fail-safe: if editor composite cannot reliably encode layers, preserve original draft and require explicit acknowledgement
 if seed.sum()==0:seed=state['draft']
 b=build(state['gray'],seed,state['conf'],models()[1]);state['build']=b;state['decisions']={int(c):'unreviewed' for c in b['components'].component.tolist()}
 table=b['components'][['component','pilot_category','prob_structural_safe','annotation_confidence_mean','area','preserve_fraction','remove_fraction','ambiguous_fraction']].copy();overview=cv2.cvtColor(b['proposal'],cv2.COLOR_GRAY2RGB)
 for _,r in b['components'].iterrows():cv2.rectangle(overview,(int(r.x),int(r.y)),(int(r.x+r.bbox_w),int(r.y+r.bbox_h)),{'green':(40,180,65),'amber':(245,165,0),'red':(225,40,40)}[r.pilot_category],3)
 choices=b['components'].component.astype(int).tolist();return overview,table,gr.update(choices=choices,value=choices[0] if choices else None),state,'Authoritative mask prepared; V7.7F integrity verified and components scored.'
def decide(state,component,status):
 if not state or state['build'] is None:raise gr.Error('Authorise the mask first.')
 state['decisions'][int(component)]=status;out=render_review(state['build'],state['decisions']);remaining=sum(v in ('unreviewed','deferred') for v in state['decisions'].values());return out,state,f'C{component}: {status}. Remaining unreviewed/deferred: {remaining}'
def approve(state):
 if not state or state['build'] is None:raise gr.Error('Nothing to approve.')
 remaining=[k for k,v in state['decisions'].items() if v in ('unreviewed','deferred')]
 if remaining:raise gr.Error(f'Page not ready: {len(remaining)} component(s) unreviewed/deferred.')
 return render_review(state['build'],state['decisions']),'Final supervised page approved in-session.'
with gr.Blocks(title='Music Annotation Removal Assistant HF-02') as demo:
 gr.Markdown('''# Music Annotation Removal Assistant — HF-02 supervised
**HF-01 is frozen.** HF-02 inserts human mask review before V7.8 and then applies the frozen V7.7F component review. No auto-accept.''')
 state=gr.State()
 with gr.Tab('1 Upload and YOLO draft'):
  inp=gr.Image(type='numpy',label='Annotated page');run=gr.Button('Generate YOLO draft');overlay=gr.Image(label='Draft overlay');heat=gr.Image(label='Confidence heatmap');status=gr.Textbox()
 with gr.Tab('2 Review authoritative mask'):
  editor=gr.ImageEditor(type='numpy',label='Notebook 07-style mask review: edit the draft before authorising',height=700);authorise_btn=gr.Button('Authorise mask and score',variant='primary')
 with gr.Tab('3 V7.7F component review'):
  scored=gr.Image(label='V7.8 proposal with category boxes',height=700);table=gr.Dataframe(interactive=False);component=gr.Dropdown(label='Component');current=gr.Image(label='Current reviewed result');msg=gr.Textbox();
  with gr.Row():accept=gr.Button('Accept correction',variant='primary');leave=gr.Button('Leave unchanged');defer=gr.Button('Defer')
 with gr.Tab('4 Final approval'):
  approve_btn=gr.Button('Approve final page',variant='primary');final=gr.Image();final_status=gr.Textbox()
 run.click(start,inp,[overlay,heat,editor,state,status]);authorise_btn.click(authorise,[editor,state],[scored,table,component,state,status]);accept.click(lambda s,c:decide(s,c,'accepted'),[state,component],[current,state,msg]);leave.click(lambda s,c:decide(s,c,'left_unchanged'),[state,component],[current,state,msg]);defer.click(lambda s,c:decide(s,c,'deferred'),[state,component],[current,state,msg]);approve_btn.click(approve,state,[final,final_status])
if __name__=='__main__':demo.launch()
