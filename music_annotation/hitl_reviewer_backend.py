"""Exact extraction of Notebook 07 HITL B V12 Python backend semantics."""
from pathlib import Path
from datetime import datetime,timezone
import base64,cv2,json,re
import numpy as np
from .hashing import md5_file
SUPPORTED={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
DEFAULT_CONFIDENCE=.20

def files(folder):return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
def reviewed_hashes(audit_dir):
    result=set()
    for path in Path(audit_dir).glob('Page*.json'):
        try:
            value=json.loads(path.read_text()).get('source_md5')
            if value:result.add(value)
        except Exception as error:print('WARNING audit',path,error)
    return result
def next_page_number(aligned_dir,corrected_dir):
    values=[]
    for folder in [Path(aligned_dir),Path(corrected_dir)]:
        for path in files(folder):
            match=re.fullmatch(r'Page(\d+)',path.stem,re.I)
            if match:values.append(int(match.group(1)))
    return max(values,default=0)+1
def build_queue(source_dir,draft_mask_dir,audit_dir):
    done=reviewed_hashes(audit_dir)
    return [p for p in files(source_dir) if (Path(draft_mask_dir)/f'{p.stem}_draft.png').exists() and md5_file(p) not in done]
def data_url(array,extension='.png'):
    ok,encoded=cv2.imencode(extension,array)
    if not ok:raise RuntimeError('Image encoding failed')
    mime='image/png' if extension=='.png' else 'image/jpeg'
    return f'data:{mime};base64,'+base64.b64encode(encoded.tobytes()).decode()
def confidence_to_png(confidence):return np.clip(confidence*255,0,255).astype(np.uint8)
def decode_mask_data_url(value,shape):
    if not value:return np.zeros(shape,np.uint8)
    payload=value.split(',',1)[1] if ',' in value else value;raw=base64.b64decode(payload);array=np.frombuffer(raw,np.uint8);image=cv2.imdecode(array,cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError('Correction mask could not be decoded')
    if image.shape!=shape:image=cv2.resize(image,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return np.where(image>0,255,0).astype(np.uint8)
def package_page(source,draft_mask_dir,confidence_dir,aligned_dir,corrected_dir):
    source=Path(source);bgr=cv2.imread(str(source));draft=cv2.imread(str(Path(draft_mask_dir)/f'{source.stem}_draft.png'),0);confidence_path=Path(confidence_dir)/f'{source.stem}_confidence.npy';confidence=np.load(confidence_path).astype(np.float32) if confidence_path.exists() else (draft>0).astype(np.float32)*DEFAULT_CONFIDENCE
    if draft is None or draft.shape!=bgr.shape[:2] or confidence.shape!=draft.shape:raise RuntimeError('Source, draft or confidence dimensions do not match.')
    return {'name':source.name,'source_path':str(source),'source_md5':md5_file(source),'width':int(bgr.shape[1]),'height':int(bgr.shape[0]),'image':data_url(bgr,'.png'),'draft':data_url(draft,'.png'),'confidence':data_url(confidence_to_png(confidence),'.png'),'next_page':f'Page{next_page_number(aligned_dir,corrected_dir):03d}'}
def save_page(source,add_data,erase_data,notes,draft_mask_dir,aligned_dir,corrected_dir,audit_dir,preview_dir):
    source=Path(source);bgr=cv2.imread(str(source));draft=cv2.imread(str(Path(draft_mask_dir)/f'{source.stem}_draft.png'),0);additions=decode_mask_data_url(add_data,draft.shape);erasures=decode_mask_data_url(erase_data,draft.shape);final=(((draft>0)|(additions>0))&~(erasures>0)).astype(np.uint8)*255
    output_name=f'Page{next_page_number(aligned_dir,corrected_dir):03d}';image_output=Path(aligned_dir)/f'{output_name}.png';mask_output=Path(corrected_dir)/f'{output_name}.png'
    for d in [Path(aligned_dir),Path(corrected_dir),Path(audit_dir),Path(preview_dir)]:d.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(image_output),bgr);cv2.imwrite(str(mask_output),final);overlay=bgr.copy();colour=overlay.copy();colour[final>0]=(0,0,255);overlay=cv2.addWeighted(overlay,.60,colour,.40,0);preview=Path(preview_dir)/f'{output_name}_overlay.png';cv2.imwrite(str(preview),overlay)
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),'page':output_name,'source':str(source),'source_md5':md5_file(source),'draft_pixels':int(np.sum(draft>0)),'added_pixels':int(np.sum(additions>0)),'erased_pixels':int(np.sum(erasures>0)),'final_pixels':int(np.sum(final>0)),'notes':notes,'aligned_output':str(image_output),'corrected_mask_output':str(mask_output)};(Path(audit_dir)/f'{output_name}.json').write_text(json.dumps(audit,indent=2));return audit
