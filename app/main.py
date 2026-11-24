from pathlib import Path
from datetime import datetime,timezone
import io,json,zipfile,cv2,numpy as np,pandas as pd
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import FileResponse,HTMLResponse,Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from . import store,yolo,v78,policy
ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'frozen/yolo_detector/best.pt';POLICY=ROOT/'frozen/structural_policy_v7_7f/artefacts';_model=None;_policy=None
app=FastAPI(title='Music Annotation HF-03.0');app.mount('/static',StaticFiles(directory=ROOT/'app/static'),name='static')
class Stroke(BaseModel):mode:str;brush:float;points:list[list[float]];view_width:int;view_height:int
class Decision(BaseModel):component:int;status:str

def get_model():
 global _model
 if _model is None:
  if not MODEL.exists():raise RuntimeError('Missing frozen/yolo_detector/best.pt')
  _model=yolo.load(MODEL)
 return _model
def get_policy():
 global _policy
 if _policy is None:_policy=policy.load(POLICY)
 return _policy
def png(a):
 ok,b=cv2.imencode('.png',a);return Response(content=b.tobytes(),media_type='image/png',headers={'Cache-Control':'no-store'})
def overlay(gray,mask):
 rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);c=rgb.copy();c[mask>0]=(60,60,255);return cv2.addWeighted(rgb,.62,c,.38,0)
@app.get('/',response_class=HTMLResponse)
def home():return (ROOT/'app/templates/index.html').read_text()
@app.post('/api/session')
async def create_session(file:UploadFile=File(...)):
 try:a=np.array(Image.open(io.BytesIO(await file.read())).convert('L'))
 except Exception as e:raise HTTPException(400,f'Unreadable image: {e}')
 conf=yolo.confidence_map(a,get_model());draft=(conf>=yolo.CONF).astype(np.uint8)*255;sid=store.create(a,conf,draft,file.filename or 'upload');d=store.folder(sid);cv2.imwrite(str(d/'annotated_page.png'),a);cv2.imwrite(str(d/'yolo_draft_mask.png'),draft);np.save(d/'yolo_confidence.npy',conf.astype(np.float16));cv2.imwrite(str(d/'mask_revision_000.png'),draft);return {'session_id':sid,'width':a.shape[1],'height':a.shape[0],'revision':0,'image_url':f'/api/session/{sid}/image','overlay_url':f'/api/session/{sid}/overlay?rev=0'}
@app.get('/api/session/{sid}/image')
def image(sid:str):return png(store.load(sid)['gray'])
@app.get('/api/session/{sid}/overlay')
def mask_overlay(sid:str,rev:int=0):
 z=store.load(sid);return png(overlay(z['gray'],z['working']))
@app.get('/api/session/{sid}/mask')
def mask(sid:str):return png(store.load(sid)['working'])
@app.post('/api/session/{sid}/stroke')
def stroke(sid:str,s:Stroke):
 if s.mode not in {'add','erase'} or not s.points:raise HTTPException(400,'Invalid stroke')
 z=store.load(sid);h,w=z['working'].shape;sx=w/max(s.view_width,1);sy=h/max(s.view_height,1);pts=[(round(p[0]*sx),round(p[1]*sy)) for p in s.points];brush=max(1,round(s.brush*(sx+sy)/2));m=np.zeros((h,w),np.uint8)
 if len(pts)==1:cv2.circle(m,pts[0],max(1,brush//2),255,-1)
 else:
  for a,b in zip(pts,pts[1:]):cv2.line(m,a,b,255,brush,cv2.LINE_AA)
 before=z['working'].copy();z['working'][m>0]=255 if s.mode=='add' else 0;store.save(sid,z);a=store.read_audit(sid);a['revision']+=1;rev=a['revision'];a['events'].append({'utc':datetime.now(timezone.utc).isoformat(),'revision':rev,'operation':s.mode,'brush':brush,'points':len(pts),'changed_pixels':int(np.sum(before!=z['working']))});store.write_audit(sid,a);cv2.imwrite(str(store.folder(sid)/f'mask_revision_{rev:03d}.png'),z['working']);return {'revision':rev,'changed_pixels':int(np.sum(before!=z['working'])),'mask_pixels':int((z['working']>0).sum()),'overlay_url':f'/api/session/{sid}/overlay?rev={rev}'}
@app.post('/api/session/{sid}/undo')
def undo(sid:str):
 a=store.read_audit(sid);rev=max(0,int(a['revision'])-1);p=store.folder(sid)/f'mask_revision_{rev:03d}.png';m=cv2.imread(str(p),0)
 if m is None:raise HTTPException(409,'Previous revision unavailable')
 z=store.load(sid);z['working']=m;store.save(sid,z);a['revision']=rev;a['events'].append({'utc':datetime.now(timezone.utc).isoformat(),'operation':'undo','revision':rev});store.write_audit(sid,a);return {'revision':rev,'overlay_url':f'/api/session/{sid}/overlay?rev={rev}'}
@app.post('/api/session/{sid}/authorise')
def authorise(sid:str):
 z=store.load(sid);d=store.folder(sid);cv2.imwrite(str(d/'reviewed_authoritative_mask.png'),z['working']);mask,telea,proposal,routes,labels,frame=v78.build(z['gray'],z['working']);frame=policy.score(frame,get_policy());np.savez_compressed(d/'build.npz',mask=mask,telea=telea,proposal=proposal,labels=labels,**{k:v.astype(np.uint8) for k,v in routes.items()});frame.to_csv(d/'components.csv',index=False);cv2.imwrite(str(d/'prepared_dilation2_mask.png'),mask);cv2.imwrite(str(d/'geometry_t80_proposal.png'),proposal);dec={str(int(c)):'unreviewed' for c in frame.component.tolist()};(d/'decisions.json').write_text(json.dumps(dec,indent=2));return {'components':frame.replace({np.nan:None}).to_dict(orient='records'),'proposal_url':f'/api/session/{sid}/proposal','policy_loaded':get_policy() is not None}
@app.get('/api/session/{sid}/proposal')
def proposal(sid:str):
 with np.load(store.folder(sid)/'build.npz') as z:return png(z['proposal'])
@app.post('/api/session/{sid}/decision')
def decision(sid:str,d:Decision):
 p=store.folder(sid)/'decisions.json';dec=json.loads(p.read_text());dec[str(d.component)]=d.status;p.write_text(json.dumps(dec,indent=2));return {'ok':True}
@app.post('/api/session/{sid}/finish')
def finish(sid:str):
 d=store.folder(sid);z=store.load(sid);cv2.imwrite(str(d/'reviewed_authoritative_mask.png'),z['working']);out=d/'hf_03_0_training_and_audit.zip';manifest={'version':'HF-03.0','training_pair':['annotated_page.png','reviewed_authoritative_mask.png'],'raw_yolo_is_training_truth':False,'files':[]}
 for p in sorted(d.iterdir()):
  if p.is_file() and p!=out:manifest['files'].append({'name':p.name,'bytes':p.stat().st_size})
 (d/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as q:
  for p in sorted(d.iterdir()):
   if p.is_file() and p!=out:q.write(p,p.name)
 return {'download_url':f'/api/session/{sid}/download'}
@app.get('/api/session/{sid}/download')
def download(sid:str):return FileResponse(store.folder(sid)/'hf_03_0_training_and_audit.zip',filename='hf_03_0_training_and_audit.zip')
@app.get('/health')
def health():return {'ok':True,'version':'3.0.0'}
