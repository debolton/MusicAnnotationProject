from pathlib import Path
import json,cv2,numpy as np,pandas as pd,gradio as gr
from music_annotation.hitl_generator import load_model,confidence_map,draft_from_confidence
from music_annotation.v77f import load_policy
from music_annotation.session_store import create,load,save,write_json,folder
from music_annotation.reviewer_backend import package,rasterise_strokes,build,current,archive,preview_size,resize
from music_annotation.image_overlay_ui import HTML,JS,CSS
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
    a=np.asarray(image).astype(np.uint8);gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a;confidence=confidence_map(gray,models()[0],'cpu');draft=draft_from_confidence(confidence);sid=create(gray,confidence,draft);return json.dumps(package(gray,draft,confidence,'uploaded-page')),sid,'YOLO draft complete. Open Mask review.'
def authorise(command,sid):
    if not command:raise gr.Error('No reviewer data received.')
    q=json.loads(command);z=load(sid);gray=z['gray'];confidence=z['confidence'].astype(np.float32);draft=z['draft'];size=preview_size(gray.shape);strokes=q.get('strokes',[]);add,erase=rasterise_strokes(strokes,(size[1],size[0]),gray.shape);seed=(((draft>0)|(add>0))&~(erase>0)).astype(np.uint8)*255;mask,telea,proposal,routes,labels,frame=build(gray,seed,confidence,models()[1]);save(sid,'build',seed=seed,mask=mask,proposal=proposal,labels=labels);decisions={str(int(c)):'unreviewed' for c in frame.component.tolist()};write_json(sid,'decisions',decisions);art=archive(folder(sid),gray,confidence,draft,add,erase,seed,mask,telea,proposal,routes,frame,decisions);frame.to_csv(folder(sid)/'components.csv',index=False);ids=frame.component.astype(int).tolist();table=frame[['component','pilot_category','prob_structural_safe','annotation_confidence_mean','area','preserve_fraction','remove_fraction','ambiguous_fraction']];return resize(proposal,preview_size(gray.shape)),table,gr.update(choices=ids,value=ids[0] if ids else None),decisions,f'Authorised. ADD {int((add>0).sum()):,} px; ERASE {int((erase>0).sum()):,} px.',str(art),''
def crop(sid,cid):
    if cid is None:return None,''
    z=load(sid);b=load(sid,'build');f=pd.read_csv(folder(sid)/'components.csv');r=f[f.component.astype(int)==int(cid)].iloc[0];m=110;x0=max(0,int(r.x)-m);y0=max(0,int(r.y)-m);x1=min(z['gray'].shape[1],int(r.x+r.bbox_w)+m);y1=min(z['gray'].shape[0],int(r.y+r.bbox_h)+m);a=cv2.cvtColor(z['gray'][y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);p=cv2.cvtColor(b['proposal'][y0:y1,x0:x1],cv2.COLOR_GRAY2RGB);return np.hstack([a,np.full((a.shape[0],8,3),235,np.uint8),p]),f'C{cid} | {r.pilot_category.upper()} | p_safe={r.prob_structural_safe:.4f}'
def decide(sid,cid,status,decisions):
    if cid is None:raise gr.Error('Select a component.')
    decisions=dict(decisions or {});decisions[str(int(cid))]=status;write_json(sid,'decisions',decisions);z=load(sid);b=load(sid,'build');out=current(z['gray'],b['proposal'],b['labels'],decisions);return resize(out,preview_size(z['gray'].shape)),decisions,f'C{cid}: {status}'
def finalise(sid,decisions):
    z=load(sid);b=load(sid,'build');f=pd.read_csv(folder(sid)/'components.csv');missing=[int(c) for c in f.component if decisions.get(str(int(c))) not in ('accepted','left_unchanged')]
    if missing:raise gr.Error(f'{len(missing)} component(s) remain unreviewed/deferred.')
    out=current(z['gray'],b['proposal'],b['labels'],decisions);assert np.array_equal(out[b['mask']==0],z['gray'][b['mask']==0]);cv2.imwrite(str(folder(sid)/'approved_final_page.png'),out);return out,'Final page approved. Outside-mask immutability passed.'
with gr.Blocks(title='Music Annotation HF-02.8',css=CSS) as demo:
    gr.Markdown('''# Music Annotation Removal Assistant - HF-02.8
The page is rendered as a normal browser image. Only the transparent ADD/ERASE stroke layer uses canvas.''')
    sid=gr.State('');decisions=gr.State({})
    with gr.Tab('1 Upload'):
        inp=gr.Image(type='numpy',label='Annotated page');go=gr.Button('Generate YOLO draft',variant='primary');status=gr.Textbox(interactive=False)
    with gr.Tab('2 Mask review'):
        reviewer=gr.HTML(HTML,js_on_load=JS);pkg=gr.Textbox(elem_classes=['bridge-offscreen']);bridge=gr.Textbox(elem_id='action-bridge',elem_classes=['bridge-offscreen']);mask_status=gr.Textbox(interactive=False);assets=gr.File(label='Training and audit artefacts')
    with gr.Tab('3 Component review'):
        proposal=gr.Image(label='V7.8 proposal',height=540);table=gr.Dataframe(interactive=False);component=gr.Dropdown(label='Component');pair=gr.Image(label='ORIGINAL | PROPOSED',height=440);info=gr.Textbox(interactive=False);current_image=gr.Image(label='Current reviewed result',height=540);decision_status=gr.Textbox(interactive=False)
        with gr.Row():accept=gr.Button('Accept correction',variant='primary');leave=gr.Button('Leave unchanged');defer=gr.Button('Defer')
    with gr.Tab('4 Final approval'):
        approve=gr.Button('Approve final page',variant='primary');final=gr.Image(label='Approved full-resolution page');final_status=gr.Textbox(interactive=False)
    event=go.click(generate,inp,[pkg,sid,status]);event.then(None,pkg,None,js='(raw)=>{if(raw&&window.hitlIngest)window.hitlIngest(raw);return []}')
    bridge.change(authorise,[bridge,sid],[proposal,table,component,decisions,mask_status,assets,bridge]);component.change(crop,[sid,component],[pair,info]);accept.click(lambda s,c,d:decide(s,c,'accepted',d),[sid,component,decisions],[current_image,decisions,decision_status]);leave.click(lambda s,c,d:decide(s,c,'left_unchanged',d),[sid,component,decisions],[current_image,decisions,decision_status]);defer.click(lambda s,c,d:decide(s,c,'deferred',d),[sid,component,decisions],[current_image,decisions,decision_status]);approve.click(finalise,[sid,decisions],[final,final_status])
if __name__=='__main__':demo.launch()
