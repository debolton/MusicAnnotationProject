from pathlib import Path
from datetime import datetime,timezone
import io,json,zipfile,cv2,numpy as np
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import HTMLResponse,Response,FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from . import store,yolo
from frozen.v8_9.v8_9_operator import load_v89_operator
ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'frozen/yolo_detector/best.pt';PP=ROOT/'frozen/p_print_provider_v8_2f';RB=ROOT/'frozen/notebook52f_remove_bundle';_model=None;_op=None
app=FastAPI(title='HF-04.0 Music Annotation');app.mount('/static',StaticFiles(directory=ROOT/'app/static'),name='static')
class Stroke(BaseModel):mode:str;brush:float;points:list[list[float]];view_width:int;view_height:int
class Approval(BaseModel):status:str;note:str=''
def get_model():
 global _model
 if _model is None:
  if not MODEL.exists():raise RuntimeError('Missing frozen/yolo_detector/best.pt')
  _model=yolo.load(MODEL)
 return _model
def get_op():
 global _op
 if _op is None:_op=load_v89_operator(PP,RB,device='cuda' if __import__('torch').cuda.is_available() else 'cpu')
 return _op
def png(a):
 ok,b=cv2.imencode('.png',a);return Response(content=b.tobytes(),media_type='image/png',headers={'Cache-Control':'no-store'})
def overlay(gray,mask):
 rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);c=rgb.copy();c[mask>0]=(60,60,255);return cv2.addWeighted(rgb,.62,c,.38,0)
@app.get('/',response_class=HTMLResponse)
def home():return (ROOT/'app/templates/index.html').read_text()
@app.post('/api/session')
async def create(file:UploadFile=File(...)):
 a=np.array(Image.open(io.BytesIO(await file.read())).convert('L'));conf=yolo.confidence_map(a,get_model());draft=(conf>=yolo.CONF).astype(np.uint8)*255;sid=store.create(a,conf,draft,file.filename or 'upload');d=store.folder(sid);cv2.imwrite(str(d/'annotated_page.png'),a);cv2.imwrite(str(d/'yolo_draft_mask.png'),draft);np.save(d/'yolo_confidence.npy',conf.astype(np.float16));cv2.imwrite(str(d/'mask_revision_000.png'),draft);return {'session_id':sid,'image_url':f'/api/session/{sid}/image','overlay_url':f'/api/session/{sid}/overlay?rev=0'}
@app.get('/api/session/{sid}/image')
def image(sid:str):return png(store.load(sid)['gray'])
@app.get('/api/session/{sid}/overlay')
def ov(sid:str,rev:int=0):
 z=store.load(sid);return png(overlay(z['gray'],z['working']))
@app.post('/api/session/{sid}/stroke')
def stroke(sid:str,s:Stroke):
 z=store.load(sid);h,w=z['working'].shape;sx=w/max(s.view_width,1);sy=h/max(s.view_height,1);pts=[(round(p[0]*sx),round(p[1]*sy)) for p in s.points];brush=max(1,round(s.brush*(sx+sy)/2));m=np.zeros((h,w),np.uint8)
 if len(pts)==1:cv2.circle(m,pts[0],max(1,brush//2),255,-1)
 else:
  for a,b in zip(pts,pts[1:]):cv2.line(m,a,b,255,brush,cv2.LINE_AA)
 z['working'][m>0]=255 if s.mode=='add' else 0;store.save(sid,z);a=store.audit(sid);a['revision']+=1;rev=a['revision'];a['events'].append({'utc':datetime.now(timezone.utc).isoformat(),'revision':rev,'operation':s.mode,'points':len(pts)});store.put_audit(sid,a);cv2.imwrite(str(store.folder(sid)/f'mask_revision_{rev:03d}.png'),z['working']);return {'revision':rev,'overlay_url':f'/api/session/{sid}/overlay?rev={rev}'}
@app.post('/api/session/{sid}/undo')
def undo(sid:str):
 a=store.audit(sid);rev=max(0,int(a['revision'])-1);m=cv2.imread(str(store.folder(sid)/f'mask_revision_{rev:03d}.png'),0);z=store.load(sid);z['working']=m;store.save(sid,z);a['revision']=rev;store.put_audit(sid,a);return {'revision':rev,'overlay_url':f'/api/session/{sid}/overlay?rev={rev}'}
@app.post('/api/session/{sid}/authorise')
def authorise(sid:str):
 z=store.load(sid);d=store.folder(sid);cv2.imwrite(str(d/'reviewed_authoritative_seed_mask.png'),z['working']);res=get_op().run_page(z['gray'],z['working'],page=sid);cv2.imwrite(str(d/'v8_9_corrected.png'),res.corrected);cv2.imwrite(str(d/'v8_8_parent.png'),res.parent_v88_corrected);cv2.imwrite(str(d/'v7_8_baseline.png'),res.baseline_v78);cv2.imwrite(str(d/'prepared_authoritative_mask.png'),res.authoritative_mask.astype(np.uint8)*255);cv2.imwrite(str(d/'ring2_mask.png'),res.ring2_mask.astype(np.uint8)*255);cv2.imwrite(str(d/'v8_9_delta.png'),res.v89_delta_mask.astype(np.uint8)*255);cv2.imwrite(str(d/'safe_remove_mask.png'),res.safe_remove_mask.astype(np.uint8)*255);cv2.imwrite(str(d/'m5_remove_mask.png'),res.m5_remove_mask.astype(np.uint8)*255);np.savez_compressed(d/'operator_diagnostics.npz',p_print=res.p_print,p_remove=res.p_remove,reason_map=res.reason_map,m3=res.m3_mask,ring2=res.ring2_mask,safe_remove=res.safe_remove_mask,m5_remove=res.m5_remove_mask);res.features.to_csv(d/'remove_features.csv',index=False);(d/'operator_audit.json').write_text(json.dumps(res.audit,indent=2,default=str));return {'ok':True,'corrected_url':f'/api/session/{sid}/result','original_url':f'/api/session/{sid}/image','diagnostics':{'ring2_pixels':int(res.ring2_mask.sum()),'v89_delta_pixels':int(res.v89_delta_mask.sum()),'safe_remove_pixels':int(res.safe_remove_mask.sum()),'outside_mask_drift':int(res.audit.get('outside_mask_drift',0))}}
@app.get('/api/session/{sid}/result')
def result(sid:str):return png(cv2.imread(str(store.folder(sid)/'v8_9_corrected.png'),0))
@app.get('/api/session/{sid}/diagnostic/{name}')
def diag(sid:str,name:str):
 names={'mask':'prepared_authoritative_mask.png','ring2':'ring2_mask.png','delta':'v8_9_delta.png','safe_remove':'safe_remove_mask.png','m5':'m5_remove_mask.png','v78':'v7_8_baseline.png','v88':'v8_8_parent.png'}
 if name not in names:raise HTTPException(404,'Unknown diagnostic')
 return png(cv2.imread(str(store.folder(sid)/names[name]),0))
@app.post('/api/session/{sid}/approval')
def approval(sid:str,x:Approval):
 if x.status not in {'approved','return_to_mask','flag_failure'}:raise HTTPException(400,'Invalid status')
 d=store.folder(sid);(d/'whole_page_review.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'status':x.status,'note':x.note},indent=2));return {'ok':True,'status':x.status}
@app.post('/api/session/{sid}/finish')
def finish(sid:str):
 d=store.folder(sid);z=store.load(sid);cv2.imwrite(str(d/'reviewed_authoritative_seed_mask.png'),z['working']);out=d/'hf_04_0_training_and_audit.zip';manifest={'version':'HF-04.0','training_pair':['annotated_page.png','reviewed_authoritative_seed_mask.png'],'raw_yolo_is_training_truth':False,'correction_operator':'frozen V8.9','files':[]}
 for p in sorted(d.iterdir()):
  if p.is_file() and p!=out:manifest['files'].append({'name':p.name,'bytes':p.stat().st_size})
 (d/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as q:
  for p in sorted(d.iterdir()):
   if p.is_file() and p!=out:q.write(p,p.name)
 return {'download_url':f'/api/session/{sid}/download'}
@app.get('/api/session/{sid}/download')
def download(sid:str):return FileResponse(store.folder(sid)/'hf_04_0_training_and_audit.zip',filename='hf_04_0_training_and_audit.zip')
@app.get('/health')
def health():return {'ok':True,'version':'4.0.0'}
