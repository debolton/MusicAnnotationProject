# Merge these imports/models/routes into app/main.py. They do not alter upload,
# mask editing, Authorise, V7.8 or V7.7F.
from . import component_review
class ReviewDecision(BaseModel): component:int; status:str; note:str=''
class FinalApproval(BaseModel): note:str=''
@app.get('/api/session/{sid}/review/queue')
def review_queue(sid:str):
 d=store.folder(sid);items=component_review.queue(d);return {'components':items,'summary':component_review.summary(d)}
@app.get('/api/session/{sid}/review/component/{component}/{view}')
def review_crop(sid:str,component:int,view:str):
 if view not in {'original','proposed'}:raise HTTPException(404,'Unknown review view')
 p=component_review.component_payload(store.folder(sid),component);a=p[view].copy();m=p['mask'];rgb=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR);colour=rgb.copy();colour[m]=(30,190,255);rgb=cv2.addWeighted(rgb,.82,colour,.18,0);return png(rgb)
@app.post('/api/session/{sid}/review/decision')
def review_decision(sid:str,v:ReviewDecision):return component_review.decide(store.folder(sid),v.component,v.status,v.note)
@app.post('/api/session/{sid}/review/approve')
def review_approve(sid:str,v:FinalApproval):
 out,record=component_review.assemble(store.folder(sid),v.note);return {'ok':True,'record':record,'image_url':f'/api/session/{sid}/review/final'}
@app.get('/api/session/{sid}/review/final')
def review_final(sid:str):
 a=cv2.imread(str(store.folder(sid)/'final_approved_page.png'),0)
 if a is None:raise HTTPException(404,'Final page not approved')
 return png(a)
