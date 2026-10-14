# V8.0 selected-pixel crop renderer
# Run in Google Colab. Read-only with respect to source/project artefacts.

from google.colab import drive
drive.mount('/content/drive')

import cv2, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/MusicAnnotationProject')
V80_RUN = None  # Optional: Path('...') to an exact V8.0 run directory
CROP_RADIUS = 55
OVERVIEW_RADIUS = 95

# V8.0 audit rows from the run being inspected.
TRUE_POINTS = {
    'Page015': [(308,149),(308,150),(308,151),(662,885),(662,886),(663,885),(663,886),(664,885),(664,886),(665,886),(666,886)],
    'Page020': [(1122,751),(1122,752)],
    'Page036': [(493,432),(493,433),(500,379),(500,380),(500,381),(501,380),(501,381),(502,380),(502,381),(503,380),(503,381),(504,381),(505,381),(506,381),(507,382),(515,423),(515,424),(515,425),(515,426),(515,427),(515,428),(604,786),(604,787),(626,778),(626,779),(626,780),(626,781),(626,782)],
}
FALSE_POINTS = {'Page036': [(611,783),(626,786)]}
UNCERTAIN_POINTS = {'Page036': [(603,783)]}

# Requested centres ensure each distinct neighbourhood is rendered even where points cluster.
CENTRES = [
    ('Page015',308,150,'benefit_A'),
    ('Page015',664,886,'benefit_B'),
    ('Page020',1122,752,'benefit_C'),
    ('Page036',500,381,'benefit_D'),
    ('Page036',515,426,'benefit_E'),
    ('Page036',607,784,'mixed_neighbourhood_A'),
    ('Page036',626,783,'mixed_neighbourhood_B'),
]

SUPPORTED = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}

def newest_complete_run(root):
    candidates=[]
    if root.exists():
        for p in root.rglob('*'):
            if not p.is_dir(): continue
            names={x.name for x in p.rglob('*') if x.is_file()}
            if 'visual_audit_beneficial.csv' in names and 'visual_audit_false_heavy.csv' in names:
                candidates.append(p)
    if not candidates:
        raise RuntimeError(f'No V8.0 run containing the visual-audit CSVs found below {root}')
    return max(candidates,key=lambda p:p.stat().st_mtime)

def image_map(folder):
    folder=Path(folder)
    return {p.stem:p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED} if folder.exists() else {}

def find_file(run,name):
    hits=list(Path(run).rglob(name))
    if len(hits)!=1:
        raise RuntimeError(f'Expected exactly one {name} below {run}; found {len(hits)}: {hits[:10]}')
    return hits[0]

def latest_v75():
    root=ROOT/'reconstruction_geometry_benchmark_v7_5'
    hits=[]
    if root.exists():
        for p in root.rglob('*'):
            if p.is_dir() and (p/'tables'/'cache_status.csv').exists(): hits.append(p)
    return max(hits,key=lambda p:p.stat().st_mtime) if hits else None

def crop(a,cy,cx,r):
    h,w=a.shape[:2]; y0=max(0,cy-r);y1=min(h,cy+r+1);x0=max(0,cx-r);x1=min(w,cx+r+1)
    return a[y0:y1,x0:x1],(y0,x0,y1,x1)

def colour_mask(gray, masks):
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    out=rgb.copy().astype(np.float32)
    colours={'true':(0,180,0),'false':(220,0,0),'uncertain':(255,165,0),'mask':(255,215,0),'preserve':(30,110,255),'remove':(0,190,80),'ambiguous':(255,140,0)}
    for name,m,alpha in masks:
        if m is None: continue
        q=m.astype(bool)
        c=np.array(colours[name],np.float32)
        out[q]=out[q]*(1-alpha)+c*alpha
    return np.clip(out,0,255).astype(np.uint8)

def point_maps(page,shape):
    d={k:np.zeros(shape,bool) for k in ['true','false','uncertain']}
    for y,x in TRUE_POINTS.get(page,[]):
        if 0<=y<shape[0] and 0<=x<shape[1]: d['true'][y,x]=True
    for y,x in FALSE_POINTS.get(page,[]):
        if 0<=y<shape[0] and 0<=x<shape[1]: d['false'][y,x]=True
    for y,x in UNCERTAIN_POINTS.get(page,[]):
        if 0<=y<shape[0] and 0<=x<shape[1]: d['uncertain'][y,x]=True
    # Enlarge dots only for visibility, never for measurement.
    k=np.ones((5,5),np.uint8)
    return {name:cv2.dilate(m.astype(np.uint8),k)>0 for name,m in d.items()}

V80_ROOT=ROOT/'observed_print_classifier_v8_0'
RUN=Path(V80_RUN) if V80_RUN else newest_complete_run(V80_ROOT)
OUT=RUN/'visual_pixel_crops'
OUT.mkdir(parents=True,exist_ok=True)

beneficial=pd.read_csv(find_file(RUN,'visual_audit_beneficial.csv'))
false_heavy=pd.read_csv(find_file(RUN,'visual_audit_false_heavy.csv'))
uncertain_path=list(RUN.rglob('visual_audit_uncertain.csv'))
uncertain=pd.read_csv(uncertain_path[0]) if len(uncertain_path)==1 else pd.DataFrame()

# Fail closed if the hard-coded audit coordinates no longer match the run.
def coordinate_set(df):
    if df.empty:return set()
    page_col=next((c for c in ['page','Page'] if c in df.columns),None)
    y_col=next((c for c in ['y','row','pixel_y'] if c in df.columns),None)
    x_col=next((c for c in ['x','col','pixel_x'] if c in df.columns),None)
    if not all([page_col,y_col,x_col]): return None
    return {(str(r[page_col]),int(r[y_col]),int(r[x_col])) for _,r in df.iterrows()}
expected_true={(p,y,x) for p,pts in TRUE_POINTS.items() for y,x in pts}
expected_false={(p,y,x) for p,pts in FALSE_POINTS.items() for y,x in pts}
actual_true=coordinate_set(beneficial); actual_false=coordinate_set(false_heavy)
if actual_true is not None and actual_true != expected_true:
    raise RuntimeError('Beneficial audit coordinates differ from the expected V8.0 run. Update TRUE_POINTS from the current CSV before rendering.')
if actual_false is not None and actual_false != expected_false:
    raise RuntimeError('False-heavy audit coordinates differ from the expected V8.0 run. Update FALSE_POINTS from the current CSV before rendering.')

observed_map=image_map(ROOT/'aligned')
clean_map=image_map(ROOT/'source'/'clean')
mask_map=image_map(ROOT/'masks'/'corrected')

# Prefer the frozen V7.5 cache for exact geometry_t80 and support/routing arrays.
v75=latest_v75(); cache_lookup={}
if v75:
    status=pd.read_csv(v75/'tables'/'cache_status.csv')
    if {'page','cache'}.issubset(status.columns):
        cache_lookup={str(r.page):Path(r.cache) for _,r in status.iterrows()}

loaded={}
for page in sorted({x[0] for x in CENTRES}):
    if page not in observed_map or page not in clean_map:
        raise RuntimeError(f'{page}: missing aligned observed or clean image')
    obs=cv2.imread(str(observed_map[page]),cv2.IMREAD_GRAYSCALE)
    clean=cv2.imread(str(clean_map[page]),cv2.IMREAD_GRAYSCALE)
    if obs is None or clean is None or obs.shape!=clean.shape:
        raise RuntimeError(f'{page}: observed/clean unreadable or shape mismatch')
    z={}
    zp=cache_lookup.get(page)
    if zp and zp.exists():
        with np.load(zp) as zz:z={k:zz[k] for k in zz.files}
    seed=cv2.imread(str(mask_map[page]),cv2.IMREAD_GRAYSCALE) if page in mask_map else None
    loaded[page]={'observed':obs,'clean':clean,'seed':seed,'cache':z}

for page,cy,cx,label in CENTRES:
    q=loaded[page];obs=q['observed'];clean=q['clean'];z=q['cache'];shape=obs.shape
    pm=point_maps(page,shape)
    mask=(z.get('mask')>0) if 'mask' in z else ((q['seed']>0) if q['seed'] is not None else None)
    baseline=z.get('geometry_t80')
    preserve=(z.get('g80_preserve')>0) if 'g80_preserve' in z else None
    remove=(z.get('g80_remove')>0) if 'g80_remove' in z else None
    ambiguous=(z.get('g80_ambiguous')>0) if 'g80_ambiguous' in z else None
    horizontal=(z.get('g80_horizontal')>0) if 'g80_horizontal' in z else None
    vertical=(z.get('g80_vertical')>0) if 'g80_vertical' in z else None
    compact=(z.get('g80_compact')>0) if 'g80_compact' in z else None

    panels=[];titles=[]
    def add(arr,title,cmap='gray',vmin=0,vmax=255):
        if arr is None:return
        c,_=crop(arr,cy,cx,CROP_RADIUS);panels.append((c,cmap,vmin,vmax));titles.append(title)
    add(obs,'Observed')
    add(clean,'Clean truth (audit only)')
    if mask is not None:add(mask.astype(np.uint8)*255,'Authorised removal mask')
    if baseline is not None:add(baseline,'Frozen V7.8 / geometry_t80')

    overlay=colour_mask(obs,[('true',pm['true'],.85),('false',pm['false'],.95),('uncertain',pm['uncertain'],.95)])
    add(overlay,'V8 selected pixels: green=true, red=false, orange=uncertain',None,None,None)
    if horizontal is not None:add(horizontal.astype(np.uint8)*255,'Horizontal support')
    if vertical is not None:add(vertical.astype(np.uint8)*255,'Vertical support')
    if compact is not None:add(compact.astype(np.uint8)*255,'Compact support')
    if preserve is not None:add(preserve.astype(np.uint8)*255,'V7.8 preserve route')
    if remove is not None:add(remove.astype(np.uint8)*255,'V7.8 remove route')
    if ambiguous is not None:add(ambiguous.astype(np.uint8)*255,'V7.8 ambiguous route')

    cols=4;rows=int(np.ceil(len(panels)/cols));fig,axes=plt.subplots(rows,cols,figsize=(4.6*cols,4.6*rows),squeeze=False)
    for ax in axes.flat:ax.axis('off')
    for ax,(panel,title) in zip(axes.flat,zip(panels,titles)):
        im,cmap,vmin,vmax=panel
        if cmap is None:ax.imshow(im)
        else:ax.imshow(im,cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_title(title,fontsize=10);ax.axis('off')
        # Centre crosshair in crop coordinates.
        ax.axhline(min(CROP_RADIUS,im.shape[0]-1),color='cyan',lw=.6,alpha=.65)
        ax.axvline(min(CROP_RADIUS,im.shape[1]-1),color='cyan',lw=.6,alpha=.65)
    fig.suptitle(f'{page} {label} | centre (y={cy}, x={cx}) | radius={CROP_RADIUS}px',fontsize=14)
    plt.tight_layout()
    out=OUT/f'{page}_{label}_y{cy}_x{cx}.png';plt.savefig(out,dpi=180,bbox_inches='tight');plt.close(fig)
    print('Saved',out)

# Two wider Page036 views for the adjacent true/false structures.
for cy,cx,label in [(607,784,'Page036_wide_A'),(626,783,'Page036_wide_B')]:
    obs=loaded['Page036']['observed'];pm=point_maps('Page036',obs.shape)
    rgb=colour_mask(obs,[('true',pm['true'],.85),('false',pm['false'],.95),('uncertain',pm['uncertain'],.95)])
    c,b=crop(rgb,cy,cx,OVERVIEW_RADIUS)
    fig,ax=plt.subplots(figsize=(9,9));ax.imshow(c);ax.axis('off');ax.set_title(f'{label}: green true PRINT; red false PRINT; orange UNCERTAIN | crop bounds y{x if False else b[0]}:{b[2]}, x{b[1]}:{b[3]}')
    out=OUT/f'{label}.png';plt.savefig(out,dpi=200,bbox_inches='tight');plt.close(fig);print('Saved',out)

manifest={
    'v8_0_run':str(RUN),'output':str(OUT),'crop_radius':CROP_RADIUS,'overview_radius':OVERVIEW_RADIUS,
    'observed_source':'MusicAnnotationProject/aligned','clean_truth_source':'MusicAnnotationProject/source/clean',
    'v7_5_cache_run':str(v75) if v75 else None,
    'note':'Clean truth is rendered for post-hoc audit only; this script performs no inference or training.'
}
(OUT/'crop_render_manifest.json').write_text(json.dumps(manifest,indent=2))
print('DONE:',OUT)
