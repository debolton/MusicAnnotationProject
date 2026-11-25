from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

SUPPORTED = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
CHANNELS = ['obs','norm','mask','depth','horizontal','vertical','compact','preserve','remove','ambiguous','edge']

class BatchFailure(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _safe_extract(upload: bytes, target: Path, max_files: int = 1000, max_uncompressed: int = 2_000_000_000) -> None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(upload))
    except Exception as e:
        raise BatchFailure(f'Uploaded object is not a readable ZIP: {e}')
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > max_files:
        raise BatchFailure(f'ZIP contains {len(infos)} files; limit is {max_files}')
    total = sum(i.file_size for i in infos)
    if total > max_uncompressed:
        raise BatchFailure(f'ZIP expands to {total} bytes; limit is {max_uncompressed}')
    for info in infos:
        p = PurePosixPath(info.filename)
        if p.is_absolute() or '..' in p.parts:
            raise BatchFailure(f'Unsafe ZIP member: {info.filename}')
        if info.external_attr >> 16 & 0o170000 == 0o120000:
            raise BatchFailure(f'Symlinks are not permitted: {info.filename}')
    zf.extractall(target)


def _image_map(folder: Path) -> dict[str, Path]:
    if not folder.exists() or not folder.is_dir():
        raise BatchFailure(f'Missing required directory: {folder.name}/')
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            if p.stem in out:
                raise BatchFailure(f'Duplicate filename stem in {folder.name}: {p.stem}')
            out[p.stem] = p
    return out


def _find_layout(extracted: Path) -> tuple[Path, Path]:
    candidates = [extracted] + [p for p in extracted.iterdir() if p.is_dir()]
    hits = [(p/'annotated', p/'reviewed_masks') for p in candidates if (p/'annotated').is_dir() and (p/'reviewed_masks').is_dir()]
    if len(hits) != 1:
        raise BatchFailure('ZIP must contain exactly one annotated/ + reviewed_masks/ directory pair, optionally beneath one top-level folder')
    return hits[0]


def validate_batch(extracted: Path) -> list[dict[str, Any]]:
    annotated_dir, masks_dir = _find_layout(extracted)
    annotated = _image_map(annotated_dir)
    masks = _image_map(masks_dir)
    if not annotated:
        raise BatchFailure('annotated/ contains no supported images')
    if set(annotated) != set(masks):
        raise BatchFailure(f'Filename stems differ. Missing masks={sorted(set(annotated)-set(masks))}; orphan masks={sorted(set(masks)-set(annotated))}')
    rows = []
    for stem in sorted(annotated):
        obs = cv2.imread(str(annotated[stem]), cv2.IMREAD_GRAYSCALE)
        seed = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
        if obs is None or seed is None:
            raise BatchFailure(f'{stem}: unreadable annotated page or mask')
        if obs.shape != seed.shape:
            raise BatchFailure(f'{stem}: shape mismatch annotated={obs.shape}, mask={seed.shape}')
        binary = np.where(seed > 0, 255, 0).astype(np.uint8)
        if int((binary > 0).sum()) == 0:
            raise BatchFailure(f'{stem}: reviewed mask is empty')
        rows.append({'page': stem, 'annotated_path': annotated[stem], 'mask_path': masks[stem], 'shape': obs.shape,
                     'annotated_sha256': _sha256(annotated[stem]), 'reviewed_mask_sha256': _sha256(masks[stem])})
    return rows


def _save(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), a):
        raise BatchFailure(f'Failed to write {path}')


def _contact_sheet(page_dirs: list[Path], output: Path, thumb_w: int = 350) -> None:
    # One row per page: annotated | reviewed mask | A1 | A5 | A5 minus A1.
    labels = ['ANNOTATED', 'REVIEWED MASK', 'A1', 'A5', 'A5 - A1']
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in page_dirs:
        paths = [d/'annotated.png', d/'reviewed_seed_mask.png', d/'a1_geometry_t80.png', d/'a5_v1_corrected.png', d/'a5_delta_from_a1.png']
        ims = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
        if any(x is None for x in ims):
            raise BatchFailure(f'{d.name}: contact-sheet source missing')
        panels = []
        for im, label in zip(ims, labels):
            scale = thumb_w / im.shape[1]
            resized = cv2.resize(im, (thumb_w, max(1, int(round(im.shape[0]*scale)))), interpolation=cv2.INTER_AREA)
            banner = np.full((34, thumb_w), 255, np.uint8)
            cv2.putText(banner, label, (8, 23), font, .55, 0, 1, cv2.LINE_AA)
            panels.append(np.vstack([banner, resized]))
        h = max(x.shape[0] for x in panels)
        panels = [np.pad(x, ((0,h-x.shape[0]),(0,0)), constant_values=255) for x in panels]
        name = np.full((30, thumb_w*5), 255, np.uint8)
        cv2.putText(name, d.name, (8,21), font, .55, 0, 1, cv2.LINE_AA)
        rows.append(np.vstack([name, np.hstack(panels)]))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0,0),(0,width-r.shape[1])), constant_values=255) for r in rows]
    _save(output, np.vstack(rows))


def run_batch(upload: bytes, operator, canonical_v78, output_root: Path) -> Path:
    """Run frozen A1/A5. `operator` is the already commissioned A5Operator instance.

    Required canonical_v78 callable:
        result = canonical_v78(observed_uint8, reviewed_seed_mask_uint8)
    Required result keys:
        prepared_mask, telea, a1, norm, depth, horizontal, vertical,
        compact, preserve, remove, ambiguous, edge

    Required A5 operator method:
        result = operator.process_arrays(arrays, a1)
    Required A5 result keys:
        probability, provisional, final, y, x, print_mask, corrected
    """
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ') + '_hf045_batch'
    run_dir = Path(output_root) / run_id
    extracted = run_dir / '_input'
    results = run_dir / 'results'
    extracted.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    try:
        _safe_extract(upload, extracted)
        roster = validate_batch(extracted)
        summary_rows = []
        page_dirs = []
        input_manifest = []
        for item in roster:
            page = item['page']
            t0 = time.time()
            obs = cv2.imread(str(item['annotated_path']), cv2.IMREAD_GRAYSCALE)
            seed = np.where(cv2.imread(str(item['mask_path']), cv2.IMREAD_GRAYSCALE) > 0, 255, 0).astype(np.uint8)
            v = canonical_v78(obs, seed)
            required = {'prepared_mask','telea','a1','norm','depth','horizontal','vertical','compact','preserve','remove','ambiguous','edge'}
            missing = sorted(required - set(v))
            if missing:
                raise BatchFailure(f'{page}: canonical V7.8 result missing {missing}')
            for k in required:
                if np.asarray(v[k]).shape != obs.shape:
                    raise BatchFailure(f'{page}: {k} shape {np.asarray(v[k]).shape} != observed {obs.shape}')
            arrays = {'obs': obs, 'norm': v['norm'], 'mask': v['prepared_mask'] > 0, 'depth': v['depth'],
                      'horizontal': v['horizontal'] > 0, 'vertical': v['vertical'] > 0, 'compact': v['compact'] > 0,
                      'preserve': v['preserve'] > 0, 'remove': v['remove'] > 0, 'ambiguous': v['ambiguous'] > 0,
                      'edge': v['edge'] > 0}
            a5 = operator.process_arrays(arrays=arrays, a1=np.asarray(v['a1'], dtype=np.uint8))
            pdir = results / page
            pdir.mkdir(parents=True, exist_ok=False)
            _save(pdir/'annotated.png', obs)
            _save(pdir/'reviewed_seed_mask.png', seed)
            _save(pdir/'prepared_dilation2_mask.png', np.where(v['prepared_mask'] > 0,255,0).astype(np.uint8))
            _save(pdir/'telea.png', np.asarray(v['telea'],np.uint8))
            _save(pdir/'a1_geometry_t80.png', np.asarray(v['a1'],np.uint8))
            _save(pdir/'a5_print_mask.png', np.where(a5['print_mask'],255,0).astype(np.uint8))
            _save(pdir/'a5_delta_from_a1.png', np.where(np.asarray(a5['corrected']) != np.asarray(v['a1']),255,0).astype(np.uint8))
            _save(pdir/'a5_v1_corrected.png', np.asarray(a5['corrected'],np.uint8))
            np.save(pdir/'a5_probability.npy', np.asarray(a5['probability']))
            np.save(pdir/'a5_provisional_decision.npy', np.asarray(a5['provisional']))
            np.save(pdir/'a5_final_decision.npy', np.asarray(a5['final']))
            np.save(pdir/'a5_y.npy', np.asarray(a5['y']))
            np.save(pdir/'a5_x.npy', np.asarray(a5['x']))
            prepared = np.asarray(v['prepared_mask']) > 0
            print_mask = np.asarray(a5['print_mask'], bool)
            changed = np.asarray(a5['corrected']) != np.asarray(v['a1'])
            outside = int(np.sum(np.asarray(a5['corrected'])[~prepared] != obs[~prepared]))
            nonprint = int(np.sum(np.asarray(a5['corrected'])[~print_mask] != np.asarray(v['a1'])[~print_mask]))
            wrong = int(np.sum(changed & (np.asarray(a5['corrected']) != obs)))
            if outside or nonprint or wrong:
                raise BatchFailure(f'{page}: integrity failure outside={outside}, nonprint={nonprint}, wrong_byte={wrong}')
            provisional = np.asarray(a5['provisional']).astype(str)
            final = np.asarray(a5['final']).astype(str)
            audit = {'page':page,'policy_id':operator.policy.get('policy_id'),'shape':list(obs.shape),
                     'reviewed_seed_mask_pixels':int((seed>0).sum()),'prepared_mask_pixels':int(prepared.sum()),
                     'eligible_pixels':int(len(a5['x'])),'provisional_print':int((provisional=='PRINT').sum()),
                     'final_print':int((final=='PRINT').sum()),'run_demotions':int(((provisional=='PRINT')&(final!='PRINT')).sum()),
                     'a5_changed_from_a1':int(changed.sum()),'outside_mask_drift':outside,
                     'non_print_a1_regression':nonprint,'changed_bytes_not_equal_observed':wrong,
                     'probability_min':float(np.min(a5['probability'])) if len(a5['probability']) else None,
                     'probability_max':float(np.max(a5['probability'])) if len(a5['probability']) else None,
                     'seconds':time.time()-t0}
            (pdir/'page_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True))
            summary_rows.append(audit)
            input_manifest.append({k:v for k,v in item.items() if k not in {'annotated_path','mask_path','shape'}} | {'shape':list(item['shape'])})
            page_dirs.append(pdir)
        fields = list(summary_rows[0])
        with (run_dir/'batch_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary_rows)
        _contact_sheet(page_dirs, run_dir/'batch_contact_sheet.png')
        manifest = {'version':'HF-04.5','created_utc':datetime.now(timezone.utc).isoformat(),'mode':'reviewed-mask-batch-A1-A5',
                    'yolo_bypassed':True,'reviewed_seed_masks_authoritative':True,'mask_dilation':2,
                    'policy_id':operator.policy.get('policy_id'),'pages':len(roster),'inputs':input_manifest,
                    'startup_a5_replay_passed':bool(getattr(operator,'startup_replay',{}).get('passed'))}
        (run_dir/'batch_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str))
        shutil.rmtree(extracted)
        zip_base = run_dir.parent / run_id
        zip_path = Path(shutil.make_archive(str(zip_base),'zip',root_dir=run_dir))
        return zip_path
    except Exception:
        # Preserve a failure record, but never publish a partial success ZIP.
        (run_dir/'FAILED.json').write_text(json.dumps({'failed_utc':datetime.now(timezone.utc).isoformat()},indent=2))
        raise
