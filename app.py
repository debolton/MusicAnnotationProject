from pathlib import Path
from datetime import datetime,timezone
import cv2,gradio as gr,numpy as np,zipfile,json,uuid
ROOT=Path(__file__).parent;MODEL=ROOT/'frozen/yolo_detector/best.pt';SESS=ROOT/'runtime_sessions';SESS.mkdir(exist_ok=True);_model=None
CONF=.20;TILE=640;STRIDE=320;MAXW=600

def get_model():
 global _model
 if _model is None:
  if not MODEL.exists():raise RuntimeError(f'Missing YOLO checkpoint: {MODEL}')
  from ultralytics import YOLO
  _model=YOLO(str(MODEL))
 return _model
def positions(n):
 if n<=TILE:return [0]
 p=list(range(0,n-TILE+1,STRIDE))
 if p[-1]!=n-TILE:p.append(n-TILE)
 return p
def infer(gray):
 h,w=gray.shape;ph,pw=max(h,TILE),max(w,TILE);canvas=np.full((ph,pw,3),255,np.uint8);canvas[:h,:w]=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);out=np.zeros((ph,pw),np.float32)
 for y in positions(ph):
  for x in positions(pw):
   r=get_model().predict(canvas[y:y+TILE,x:x+TILE],conf=CONF,imgsz=TILE,device='cpu',verbose=False,retina_masks=True)[0]
   if r.masks is None:continue
   for m,s in zip(r.masks.data.cpu().numpy(),r.boxes.conf.cpu().numpy()):
    m=cv2.resize(m,(TILE,TILE),interpolation=cv2.INTER_NEAREST)>.5;q=out[y:y+TILE,x:x+TILE];q[m&(s>q)]=s
 return out[:h,:w]
def load(sid):
 with np.load(SESS/sid/'state.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(sid,z):np.savez_compressed(SESS/sid/'state.npz',**z)
def editor_background(gray,mask):
 rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);colour=rgb.copy();colour[mask>0]=(255,60,60);return cv2.addWeighted(rgb,.62,colour,.38,0)
def painted(v,shape):
 out=np.zeros(shape,np.uint8)
 if not isinstance(v,dict):return out
 for layer in v.get('layers') or []:
  a=np.asarray(layer)
  if a.size==0:continue
  m=(a[:,:,3]>0) if a.ndim==3 and a.shape[2]>=4 else (np.any(a[:,:,:3]>8,axis=2) if a.ndim==3 else a>8)
  if m.shape!=shape:m=cv2.resize(m.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
  out[m]=255
 return out
def generate(image):
 if image is None:raise gr.Error('Upload an annotated page.')
 a=np.asarray(image).astype(np.uint8);gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a;conf=infer(gray);draft=(conf>=CONF).astype(np.uint8)*255;h,w=gray.shape;s=min(1.,MAXW/w);sz=(max(1,round(w*s)),max(1,round(h*s)));pg=cv2.resize(gray,sz,interpolation=cv2.INTER_AREA);pm=cv2.resize(draft,sz,interpolation=cv2.INTER_NEAREST);sid=uuid.uuid4().hex;d=SESS/sid;d.mkdir();np.savez_compressed(d/'state.npz',gray=gray,confidence=conf.astype(np.float16),draft=draft,working=draft,preview_gray=pg,preview_mask=pm);cv2.imwrite(str(d/'annotated_page.png'),gray);cv2.imwrite(str(d/'yolo_draft_mask.png'),draft);np.save(d/'yolo_confidence.npy',conf.astype(np.float16));cv2.imwrite(str(d/'mask_revision_000.png'),draft);(d/'audit.json').write_text(json.dumps({'session_id':sid,'created_utc':datetime.now(timezone.utc).isoformat(),'review_width':sz[0],'revisions':[]},indent=2));bg=editor_background(pg,pm);return sid,bg,0,f'YOLO draft ready at {sz[0]} x {sz[1]}. Make 1-3 strokes, then commit. Full-resolution mask remains server-side.'
def commit(sid,editor,mode,revision,note):
 if not sid:raise gr.Error('Generate the YOLO draft first.')
 z=load(sid);pm=z['preview_mask'];marks=painted(editor,pm.shape);painted_count=int((marks>0).sum())
 if not painted_count:raise gr.Error('No newly painted pixels found.')
 before=pm.copy();pm[marks>0]=255 if mode=='ADD' else 0;full=cv2.resize(pm,(z['gray'].shape[1],z['gray'].shape[0]),interpolation=cv2.INTER_NEAREST);z['preview_mask']=pm;z['working']=full;save(sid,z);revision=int(revision)+1;d=SESS/sid;cv2.imwrite(str(d/f'mask_revision_{revision:03d}.png'),full);cv2.imwrite(str(d/'reviewed_authoritative_mask.png'),full);audit=json.loads((d/'audit.json').read_text());audit['revisions'].append({'revision':revision,'utc':datetime.now(timezone.utc).isoformat(),'operation':mode.lower(),'painted_preview_pixels':painted_count,'preview_mask_delta_pixels':int(np.sum(before!=pm)),'full_mask_pixels':int((full>0).sum()),'note':note or ''});(d/'audit.json').write_text(json.dumps(audit,indent=2));bg=editor_background(z['preview_gray'],pm);return bg,revision,f'Revision {revision} saved. {mode}: {painted_count:,} painted preview pixels; full-resolution mask: {int((full>0).sum()):,} pixels. Editor reset.'
def undo(sid,revision):
 if not sid:raise gr.Error('Generate the YOLO draft first.')
 revision=int(revision)
 if revision<=0:raise gr.Error('Already at the YOLO draft.')
 z=load(sid);newrev=revision-1;full=z['draft'].copy() if newrev==0 else cv2.imread(str(SESS/sid/f'mask_revision_{newrev:03d}.png'),0)
 if full is None:raise gr.Error(f'Missing revision {newrev}.')
 pm=cv2.resize(full,(z['preview_gray'].shape[1],z['preview_gray'].shape[0]),interpolation=cv2.INTER_NEAREST);z['working']=full;z['preview_mask']=pm;save(sid,z);cv2.imwrite(str(SESS/sid/'reviewed_authoritative_mask.png'),full);return editor_background(z['preview_gray'],pm),newrev,f'Restored revision {newrev}. Editor reset.'
def finish(sid,revision):
 if not sid:raise gr.Error('Generate the YOLO draft first.')
 z=load(sid);d=SESS/sid;cv2.imwrite(str(d/'reviewed_authoritative_mask.png'),z['working']);manifest={'version':'HF-02.11','revision':int(revision),'training_pair':['annotated_page.png','reviewed_authoritative_mask.png'],'raw_yolo_is_training_truth':False,'review_width':int(z['preview_gray'].shape[1]),'files':[]}
 for p in sorted(d.iterdir()):
  if p.is_file() and p.suffix!='.zip':manifest['files'].append({'name':p.name,'bytes':p.stat().st_size})
 (d/'package_manifest.json').write_text(json.dumps(manifest,indent=2));out=d/'hf_02_11_training_and_audit.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as q:
  for p in sorted(d.iterdir()):
   if p.is_file() and p!=out:q.write(p,p.name)
 final=np.where(z['working']>0,255,0).astype(np.uint8)
 return final,str(out),f'Review finished at revision {revision}. Full-resolution authoritative mask and all revisions packaged.'
with gr.Blocks(title='HF-02.11 Minimal Single ImageEditor') as demo:
 gr.Markdown('# HF-02.11 Minimal Single-ImageEditor Reviewer\n600px review image. After each commit, only one fresh editor image and text are returned. Full-resolution pages, confidence and mask revisions remain server-side until Finish.')
 sid=gr.State('');revision=gr.State(0)
 with gr.Tab('1 Upload and YOLO'):
  upload=gr.Image(type='numpy',label='Annotated page');generate_button=gr.Button('Generate YOLO draft',variant='primary');generation_status=gr.Textbox(interactive=False)
 with gr.Tab('2 Mask review'):
  mode=gr.Radio(['ADD','ERASE'],value='ADD',label='Operation for next short batch');editor=gr.ImageEditor(type='numpy',label='Make 1-3 strokes, then commit',height=650,brush=gr.Brush(colors=['#0066ff'],default_color='#0066ff',color_mode='fixed'));note=gr.Textbox(label='Optional revision note')
  with gr.Row():commit_button=gr.Button('Commit and reset editor',variant='primary');undo_button=gr.Button('Undo previous committed revision')
  review_status=gr.Textbox(interactive=False,label='Server-side mask status')
 with gr.Tab('3 Finish'):
  finish_button=gr.Button('Finish mask review and package evidence',variant='primary');final_mask=gr.Image(type='numpy',label='Final full-resolution authoritative mask',height=600);download=gr.File(label='Training and audit package');finish_status=gr.Textbox(interactive=False)
 generate_button.click(generate,upload,[sid,editor,revision,generation_status],show_progress='full')
 commit_button.click(commit,[sid,editor,mode,revision,note],[editor,revision,review_status],show_progress='minimal')
 undo_button.click(undo,[sid,revision],[editor,revision,review_status],show_progress='minimal')
 finish_button.click(finish,[sid,revision],[final_mask,download,finish_status],show_progress='minimal')
if __name__=='__main__':demo.launch()
