from pathlib import Path
from datetime import datetime,timezone
import cv2,gradio as gr,numpy as np,zipfile,json,uuid
ROOT=Path(__file__).parent;MODEL_PATH=ROOT/'frozen/yolo_detector/best.pt';SESS=ROOT/'runtime_sessions';SESS.mkdir(exist_ok=True);_model=None
CONF=.20;TILE=640;STRIDE=320;MAXW=720

def model():
 global _model
 if _model is None:
  if not MODEL_PATH.exists():raise RuntimeError(f'Missing YOLO checkpoint: {MODEL_PATH}')
  from ultralytics import YOLO
  _model=YOLO(str(MODEL_PATH))
 return _model
def pos(n):
 if n<=TILE:return [0]
 p=list(range(0,n-TILE+1,STRIDE))
 if p[-1]!=n-TILE:p.append(n-TILE)
 return p
def yolo_conf(gray):
 h,w=gray.shape;ph,pw=max(h,TILE),max(w,TILE);c=np.full((ph,pw,3),255,np.uint8);c[:h,:w]=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);out=np.zeros((ph,pw),np.float32)
 for y in pos(ph):
  for x in pos(pw):
   r=model().predict(c[y:y+TILE,x:x+TILE],conf=CONF,imgsz=TILE,device='cpu',verbose=False,retina_masks=True)[0]
   if r.masks is None:continue
   for m,s in zip(r.masks.data.cpu().numpy(),r.boxes.conf.cpu().numpy()):
    m=cv2.resize(m,(TILE,TILE),interpolation=cv2.INTER_NEAREST)>.5;q=out[y:y+TILE,x:x+TILE];q[m&(s>q)]=s
 return out[:h,:w]
def load(sid):
 with np.load(SESS/sid/'state.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(sid,z):np.savez_compressed(SESS/sid/'state.npz',**z)
def ov(g,m):
 r=cv2.cvtColor(g,cv2.COLOR_GRAY2RGB);c=r.copy();c[m>0]=(255,60,60);return cv2.addWeighted(r,.62,c,.38,0)
def ev(bg):return {'background':bg,'layers':[],'composite':bg.copy()}
def marks(v,shape):
 o=np.zeros(shape,np.uint8)
 if not isinstance(v,dict):return o
 for layer in v.get('layers') or []:
  a=np.asarray(layer)
  if a.ndim==3 and a.shape[2]>=4:m=a[:,:,3]>0
  elif a.ndim==3:m=np.any(a[:,:,:3]>8,axis=2)
  else:m=a>8
  if m.shape!=shape:m=cv2.resize(m.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
  o[m]=255
 return o
def generate(image):
 if image is None:raise gr.Error('Upload an annotated page.')
 a=np.asarray(image).astype(np.uint8);g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a;conf=yolo_conf(g);draft=(conf>=CONF).astype(np.uint8)*255;h,w=g.shape;s=min(1.,MAXW/w);sz=(max(1,round(w*s)),max(1,round(h*s)));pg=cv2.resize(g,sz,interpolation=cv2.INTER_AREA);pm=cv2.resize(draft,sz,interpolation=cv2.INTER_NEAREST);sid=uuid.uuid4().hex;d=SESS/sid;d.mkdir();np.savez_compressed(d/'state.npz',gray=g,confidence=conf.astype(np.float16),draft=draft,working=draft,preview_gray=pg,preview_mask=pm);(d/'audit.json').write_text(json.dumps({'session_id':sid,'revisions':[]},indent=2));return sid,ov(pg,pm),ev(ov(pg,pm)),draft,0,f'YOLO draft generated. Review image {sz[0]} x {sz[1]}. Commit after 1-3 short strokes.'
def commit(sid,editor,mode,rev,note):
 z=load(sid);pm=z['preview_mask'];mk=marks(editor,pm.shape);n=int((mk>0).sum())
 if not n:raise gr.Error('No new painted pixels found.')
 before=pm.copy();pm[mk>0]=255 if mode=='ADD' else 0;full=cv2.resize(pm,(z['gray'].shape[1],z['gray'].shape[0]),interpolation=cv2.INTER_NEAREST);z['preview_mask']=pm;z['working']=full;save(sid,z);rev=int(rev)+1;d=SESS/sid;cv2.imwrite(str(d/f'mask_revision_{rev:03d}.png'),full);a=json.loads((d/'audit.json').read_text());a['revisions'].append({'revision':rev,'utc':datetime.now(timezone.utc).isoformat(),'operation':mode.lower(),'painted_preview_pixels':n,'mask_delta_pixels':int(np.sum(before!=pm)),'note':note or ''});(d/'audit.json').write_text(json.dumps(a,indent=2));return ov(z['preview_gray'],pm),ev(ov(z['preview_gray'],pm)),full,rev,f'Revision {rev} committed. Editor reset.'
def undo(sid,rev):
 z=load(sid);rev=int(rev)
 if rev<=0:raise gr.Error('Already at YOLO draft.')
 nr=rev-1;full=z['draft'] if nr==0 else cv2.imread(str(SESS/sid/f'mask_revision_{nr:03d}.png'),0);pm=cv2.resize(full,(z['preview_gray'].shape[1],z['preview_gray'].shape[0]),interpolation=cv2.INTER_NEAREST);z['working']=full;z['preview_mask']=pm;save(sid,z);return ov(z['preview_gray'],pm),ev(ov(z['preview_gray'],pm)),full,nr,f'Restored revision {nr}.'
def package(sid,rev):
 z=load(sid);d=SESS/sid;cv2.imwrite(str(d/'annotated_page.png'),z['gray']);cv2.imwrite(str(d/'yolo_draft_mask.png'),z['draft']);np.save(d/'yolo_confidence.npy',z['confidence']);cv2.imwrite(str(d/'reviewed_authoritative_mask.png'),z['working']);(d/'package_manifest.json').write_text(json.dumps({'version':'HF-02.10','revision':int(rev),'training_pair':['annotated_page.png','reviewed_authoritative_mask.png'],'draft_is_training_truth':False},indent=2));out=d/'hf_02_10_training_and_audit.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as q:
  for p in d.iterdir():
   if p.is_file() and p!=out:q.write(p,p.name)
 return str(out),f'Revision {rev} packaged.'
with gr.Blocks(title='HF-02.10 Incremental Mask Reviewer') as demo:
 gr.Markdown('# HF-02.10 Incremental iPad Mask Reviewer\nReal YOLO followed by short standard ImageEditor commits. Make 1-3 strokes, commit, and continue in the freshly reset editor. No custom HTML or JavaScript. No V7.8/V7.7F in this stability build.')
 sid=gr.State('');rev=gr.State(0)
 with gr.Tab('1 Upload and YOLO'):
  up=gr.Image(type='numpy');go=gr.Button('Generate YOLO draft',variant='primary');gs=gr.Textbox(interactive=False)
 with gr.Tab('2 Incremental mask review'):
  over=gr.Image(type='numpy',label='Current mask overlay',height=580);mode=gr.Radio(['ADD','ERASE'],value='ADD');editor=gr.ImageEditor(type='numpy',height=650,label='Paint 1-3 strokes, then commit',brush=gr.Brush(colors=['#0066ff'],default_color='#0066ff',color_mode='fixed'));note=gr.Textbox(label='Optional note')
  with gr.Row():apply=gr.Button('Commit short edit batch and reset editor',variant='primary');back=gr.Button('Undo previous committed revision')
  mask=gr.Image(type='numpy',label='Current authoritative mask',height=480);rs=gr.Textbox(interactive=False)
 with gr.Tab('3 Save training evidence'):
  pack=gr.Button('Package annotated page, reviewed mask and revisions',variant='primary');file=gr.File();ps=gr.Textbox(interactive=False)
 go.click(generate,up,[sid,over,editor,mask,rev,gs],show_progress='full');apply.click(commit,[sid,editor,mode,rev,note],[over,editor,mask,rev,rs],show_progress='minimal');back.click(undo,[sid,rev],[over,editor,mask,rev,rs]);pack.click(package,[sid,rev],[file,ps])
if __name__=='__main__':demo.launch()
