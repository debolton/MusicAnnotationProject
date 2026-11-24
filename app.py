from pathlib import Path
import json, cv2, numpy as np, gradio as gr
from music_annotation.hitl_generator import load_model,confidence_map,draft_from_confidence
from music_annotation.v77_policy import load_policy
from music_annotation.supervised import build
from music_annotation.session_store import create,load,save
from music_annotation.light_canvas import preview_size,resize,data_url,png_url,overlay,heatmap
ROOT=Path(__file__).parent; YOLO=ROOT/'frozen/yolo_detector/best.pt'; POLICY=ROOT/'frozen/structural_policy_v7_7f/artefacts'; _model=_bundle=None

def models():
 global _model,_bundle
 if _model is None:_model=load_model(YOLO)
 if _bundle is None:_bundle=load_policy(POLICY)
 return _model,_bundle

def make_canvas_html(page_url,mask_url,conf_url,w,h):
 return f'''<div id="reviewRoot"><div class="toolbar"><button onclick="setView('mask')">Mask</button><button onclick="setView('confidence')">Confidence</button><button onclick="setView('original')">Original</button><span></span><button onclick="setMode('pan')">Pan</button><button onclick="setMode('add')">ADD</button><button onclick="setMode('erase')">ERASE</button><button onclick="undoStroke()">Undo</button><button onclick="redoStroke()">Redo</button><button onclick="resetStrokes()">Reset</button><label>Brush <input id="brush" type="range" min="2" max="60" value="14"></label></div><canvas id="reviewCanvas" width="{w}" height="{h}"></canvas><textarea id="strokeJSON" style="display:none"></textarea></div><script>window.HF022={{page:{json.dumps(page_url)},mask:{json.dumps(mask_url)},confidence:{json.dumps(conf_url)},mode:'pan',view:'mask',strokes:[],redo:[]}};</script>'''

def generate(image):
 if image is None:raise gr.Error('Upload a page.')
 a=np.asarray(image).astype(np.uint8);gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a;conf=confidence_map(gray,models()[0]);draft=draft_from_confidence(conf);sid=create(gray,conf,draft);size=preview_size(gray.shape);small=resize(gray,size);dm=resize(draft,size,True);cm=resize(conf,size);html=make_canvas_html(data_url(small),png_url(overlay(small,dm)),data_url(heatmap(cm)),size[0],size[1])
 return html,sid,'Mask review ready. Full-resolution arrays remain server-side.'

def authorise(strokes_json,sid):
 z=load(sid);gray=z['gray'];conf=z['conf'].astype(np.float32);draft=z['draft'];size=preview_size(gray.shape);add=np.zeros((size[1],size[0]),np.uint8);erase=np.zeros_like(add)
 try:strokes=json.loads(strokes_json or '[]')
 except:strokes=[]
 for s in strokes:
  pts=s.get('points',[]);mode=s.get('mode');th=max(1,int(s.get('brush',14)))
  target=add if mode=='add' else erase if mode=='erase' else None
  if target is None or not pts:continue
  for a,b in zip(pts,pts[1:]):cv2.line(target,(round(a[0]),round(a[1])),(round(b[0]),round(b[1])),255,th,cv2.LINE_AA)
  if len(pts)==1:cv2.circle(target,(round(pts[0][0]),round(pts[0][1])),max(1,th//2),255,-1)
 add=cv2.resize(add,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST);erase=cv2.resize(erase,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST);seed=(((draft>0)|(add>0))&~(erase>0)).astype(np.uint8)*255;b=build(gray,seed,conf,models()[1]);save(sid,'build',seed=seed,mask=b['mask'],proposal=b['proposal'],labels=b['labels']);table=b['components'][['component','pilot_category','prob_structural_safe','annotation_confidence_mean','area','preserve_fraction','remove_fraction','ambiguous_fraction']];ids=b['components'].component.astype(int).tolist();return resize(b['proposal'],size),table,gr.update(choices=ids,value=ids[0] if ids else None),{},f'Authorised mask: {int((seed>0).sum()):,} px; ADD {int((add>0).sum()):,}; ERASE {int((erase>0).sum()):,}.'

HEAD='''<style>#reviewRoot{width:100%;overflow:auto}.toolbar{position:sticky;top:0;background:white;padding:8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;z-index:2}.toolbar span{flex:1}#reviewCanvas{max-width:none;touch-action:none;border:1px solid #bbb}</style><script>
function canvas(){return document.getElementById('reviewCanvas')}function st(){return window.HF022}function loadImg(src){return new Promise(r=>{let i=new Image();i.onload=()=>r(i);i.src=src})}async function draw(){let c=canvas();if(!c||!st())return;let x=c.getContext('2d');let src=st().view==='original'?st().page:st().view==='confidence'?st().confidence:st().mask;let i=await loadImg(src);x.clearRect(0,0,c.width,c.height);x.drawImage(i,0,0,c.width,c.height);for(let s of st().strokes){x.strokeStyle=s.mode==='add'?'rgba(0,80,255,.72)':'rgba(0,190,70,.72)';x.lineWidth=s.brush;x.lineCap='round';x.beginPath();s.points.forEach((p,k)=>k?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]));x.stroke()}sync()}function sync(){let e=document.getElementById('strokeJSON');if(e){e.value=JSON.stringify(st().strokes);e.dispatchEvent(new Event('input',{bubbles:true}))}}function setView(v){st().view=v;draw()}function setMode(v){st().mode=v}function undoStroke(){if(st().strokes.length)st().redo.push(st().strokes.pop());draw()}function redoStroke(){if(st().redo.length)st().strokes.push(st().redo.pop());draw()}function resetStrokes(){st().strokes=[];st().redo=[];draw()}document.addEventListener('pointerdown',e=>{let c=canvas();if(e.target!==c||!st()||st().mode==='pan')return;let r=c.getBoundingClientRect();st().active={mode:st().mode,brush:+document.getElementById('brush').value,points:[[(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height]]};c.setPointerCapture(e.pointerId)});document.addEventListener('pointermove',e=>{let c=canvas();if(!st()?.active||e.target!==c)return;let r=c.getBoundingClientRect();st().active.points.push([(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height]);draw()});document.addEventListener('pointerup',e=>{if(st()?.active){st().strokes.push(st().active);st().active=null;st().redo=[];draw()}});new MutationObserver(()=>{if(canvas())draw()}).observe(document.documentElement,{subtree:true,childList:true});</script>'''
with gr.Blocks(title='Music Annotation HF-02.2',head=HEAD) as demo:
 gr.Markdown('# Music Annotation Removal Assistant - HF-02.2')
 sid=gr.State('');decisions=gr.State({});tabs=gr.Tabs(selected='upload');
 with tabs:
  with gr.Tab('1 Upload',id='upload'):
   inp=gr.Image(type='numpy',label='Annotated page');go=gr.Button('Generate YOLO draft',variant='primary');status=gr.Textbox()
  with gr.Tab('2 Mask review',id='mask'):
   canvas_html=gr.HTML();stroke_data=gr.Textbox(visible=False,elem_id='strokeJSON');auth=gr.Button('Authorise reviewed mask',variant='primary')
  with gr.Tab('3 Component review',id='components'):
   proposal=gr.Image();table=gr.Dataframe(interactive=False);component=gr.Dropdown();component_status=gr.Textbox()
  with gr.Tab('4 Final approval',id='final'):
   gr.Markdown('Component-review actions and final approval continue from the existing HF-02 supervised backend.')
 go.click(generate,inp,[canvas_html,sid,status]).then(lambda:gr.Tabs(selected='mask'),None,tabs)
 auth.click(authorise,[stroke_data,sid],[proposal,table,component,decisions,component_status]).then(lambda:gr.Tabs(selected='components'),None,tabs)
if __name__=='__main__':demo.launch()
