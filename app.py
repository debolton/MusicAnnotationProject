from pathlib import Path
import cv2, numpy as np, gradio as gr
from music_annotation.hitl_generator import load_model, confidence_map, draft_from_confidence
from music_annotation.v77_policy import load_policy
from music_annotation.supervised import build, render_review
from music_annotation.session_store import create, load, save
from music_annotation.ipad_review import preview_size, resize, overlay, editor_payload, paint_mask
ROOT=Path(__file__).parent
YOLO=ROOT/'frozen/yolo_detector/best.pt'
POLICY=ROOT/'frozen/structural_policy_v7_7f/artefacts'
_model=_bundle=None

def models():
    global _model,_bundle
    if _model is None:_model=load_model(YOLO)
    if _bundle is None:_bundle=load_policy(POLICY)
    return _model,_bundle

def generate(image):
    if image is None: raise gr.Error('Upload a page.')
    a=np.asarray(image).astype(np.uint8)
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a
    conf=confidence_map(gray,models()[0]); draft=draft_from_confidence(conf)
    sid=create(gray,conf,draft); size=preview_size(gray.shape)
    pg=resize(gray,size); pd=resize(draft,size,True)
    heat=cv2.applyColorMap(np.clip(resize(conf,size)*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1]
    return overlay(pg,pd),heat,editor_payload(pg),editor_payload(pg),sid,f'Draft ready at {size[0]} x {size[1]}. Full-resolution arrays remain server-side.'

def authorise(add_editor,erase_editor,sid):
    z=load(sid); gray=z['gray']; conf=z['conf'].astype(np.float32); draft=z['draft']; size=preview_size(gray.shape)
    add_small=paint_mask(add_editor,(size[1],size[0])); erase_small=paint_mask(erase_editor,(size[1],size[0]))
    add=cv2.resize(add_small,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
    erase=cv2.resize(erase_small,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
    seed=(((draft>0)|(add>0))&~(erase>0)).astype(np.uint8)*255
    b=build(gray,seed,conf,models()[1])
    save(sid,'build',seed=seed,mask=b['mask'],proposal=b['proposal'],labels=b['labels'])
    table=b['components'][['component','pilot_category','prob_structural_safe','annotation_confidence_mean','area','preserve_fraction','remove_fraction','ambiguous_fraction']]
    ids=b['components'].component.astype(int).tolist()
    return overlay(resize(gray,size),resize(seed,size,True)),resize(b['proposal'],size),table,gr.update(choices=ids,value=ids[0] if ids else None),{},f'Authorised mask: {int((seed>0).sum()):,} px; ADD {int((add>0).sum()):,}; ERASE {int((erase>0).sum()):,}.'

def decide(component,status,decisions,sid):
    if component is None: raise gr.Error('Choose a component.')
    decisions=dict(decisions or {}); decisions[str(int(component))]=status
    page=load(sid)['gray']; b=load(sid,'build'); out=page.copy()
    for key,val in decisions.items():
        if val=='accepted':
            m=b['labels']==int(key); out[m]=b['proposal'][m]
    return resize(out,preview_size(page.shape)),decisions,f'C{int(component)}: {status}'

def finalise(decisions,sid):
    page=load(sid)['gray']; b=load(sid,'build'); labels=b['labels']
    ids=[int(x) for x in np.unique(labels) if x>0 and int((labels==x).sum())>=20]
    missing=[x for x in ids if decisions.get(str(x)) not in ('accepted','left_unchanged')]
    if missing: raise gr.Error(f'{len(missing)} components remain unreviewed/deferred.')
    out=page.copy()
    for cid in ids:
        if decisions[str(cid)]=='accepted':
            m=labels==cid; out[m]=b['proposal'][m]
    assert np.array_equal(out[b['mask']==0],page[b['mask']==0])
    return out,'Full-resolution supervised page approved.'

with gr.Blocks(title='Music Annotation HF-02.1',css='.gradio-container{max-width:1250px!important}') as demo:
    gr.Markdown('# Music Annotation Removal Assistant - HF-02.1\nFull-resolution arrays remain server-side. The iPad receives previews no wider than 1200 pixels. Authoritative mask = (draft OR ADD) AND NOT ERASE.')
    sid=gr.State(''); decisions=gr.State({})
    with gr.Tab('1 Upload and YOLO'):
        inp=gr.Image(type='numpy',label='Annotated page'); go=gr.Button('Generate YOLO draft',variant='primary')
        with gr.Row(): draft_view=gr.Image(label='Draft overlay'); heat=gr.Image(label='Confidence heatmap')
        msg=gr.Textbox()
    with gr.Tab('2 ADD / ERASE mask review'):
        gr.Markdown('Paint missing annotation in ADD. Paint false-positive mask regions in ERASE.')
        with gr.Row():
            add=gr.ImageEditor(type='numpy',label='ADD missing annotation',height=700)
            erase=gr.ImageEditor(type='numpy',label='ERASE masked print',height=700)
        auth=gr.Button('Authorise mask and run V7.8/V7.7F',variant='primary')
        with gr.Row(): auth_overlay=gr.Image(label='Authoritative mask'); proposal=gr.Image(label='V7.8 proposal')
    with gr.Tab('3 Component review'):
        table=gr.Dataframe(interactive=False); component=gr.Dropdown(label='Component'); current=gr.Image(); status=gr.Textbox()
        with gr.Row(): accept=gr.Button('Accept correction',variant='primary'); leave=gr.Button('Leave unchanged'); defer=gr.Button('Defer')
    with gr.Tab('4 Final approval'):
        approve=gr.Button('Approve final page',variant='primary'); final=gr.Image(); final_status=gr.Textbox()
    go.click(generate,inp,[draft_view,heat,add,erase,sid,msg])
    auth.click(authorise,[add,erase,sid],[auth_overlay,proposal,table,component,decisions,msg])
    accept.click(lambda c,d,s:decide(c,'accepted',d,s),[component,decisions,sid],[current,decisions,status])
    leave.click(lambda c,d,s:decide(c,'left_unchanged',d,s),[component,decisions,sid],[current,decisions,status])
    defer.click(lambda c,d,s:decide(c,'deferred',d,s),[component,decisions,sid],[current,decisions,status])
    approve.click(finalise,[decisions,sid],[final,final_status])
if __name__=='__main__': demo.launch()
