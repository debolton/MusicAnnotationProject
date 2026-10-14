# modules/pilot/regression_gate.py
import json, hashlib, cv2, numpy as np, pandas as pd
from pathlib import Path
from modules.pilot.structural import geometry_t80, surrounding_features

class RegressionGate:
    def __init__(self, root, run_dir, canonical_structural_block_sha256,
                 min_component_area=20, atol=1e-9, rtol=1e-7):
        self.root=Path(root)
        self.run_dir=Path(run_dir)
        self.sha=canonical_structural_block_sha256
        self.min_component_area=min_component_area
        self.atol=atol
        self.rtol=rtol
        self.regression_dir=self.run_dir/'development_regression_gate'
        self.tables=self.regression_dir/'tables'
        self.diffs=self.regression_dir/'difference_images'
        self.tables.mkdir(parents=True,exist_ok=True)
        self.diffs.mkdir(parents=True,exist_ok=True)

    @staticmethod
    def regression_latest(root,required):
        hits=[]
        root=Path(root)
        if root.exists():
            for p in root.rglob('*'):
                if p.is_dir() and all((p/x).exists() for x in required):
                    hits.append(p)
        return max(hits,key=lambda p:p.stat().st_mtime) if hits else None

    @staticmethod
    def regression_imap(folder):
        supported={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
        folder=Path(folder)
        return {p.stem:p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported} if folder.exists() else {}

    @staticmethod
    def regression_loadz(path):
        with np.load(path) as z:
            return {k:z[k] for k in z.files}

    @staticmethod
    def regression_digest(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()+str(a.shape).encode()+str(a.dtype).encode()).hexdigest()

    def execute(self):
        v75=self.regression_latest(self.root/'reconstruction_geometry_benchmark_v7_5',['tables/cache_status.csv','tables/surrounding_area_evidence_by_component.csv'])
        v77=self.regression_latest(self.root/'structural_risk_calibration_v7_7',['tables/asymmetric_component_dataset.csv','v7_7_manifest.json'])
        if v75 is None or v77 is None:
            raise RuntimeError('Missing regression references')

        reg_clean=self.regression_imap(self.root/'source'/'clean')
        reg_observed=self.regression_imap(self.root/'aligned')
        reg_status=pd.read_csv(v75/'tables'/'cache_status.csv')
        reg_cache={str(r.page):Path(r.cache) for _,r in reg_status.iterrows()}
        reg_evidence=pd.read_csv(v75/'tables'/'surrounding_area_evidence_by_component.csv')
        reg_v77=pd.read_csv(v77/'tables'/'asymmetric_component_dataset.csv')

        method_col=next((x for x in ['method','candidate','geometry_candidate'] if x in reg_evidence.columns),None)
        if method_col is not None:
            reg_evidence=reg_evidence[reg_evidence[method_col].astype(str).str.lower().str.contains('t80')].copy()

        pages=sorted(set(reg_clean)&set(reg_observed)&set(reg_cache))
        reg_stage_a=[];reg_missing=[];reg_geometry=[];reg_routes=[];reg_components=[];reg_predictor_rows=[];reg_labels=[]
        predictors=['w','h','area','aspect_ratio','fill_ratio','touches_page_border','context_pixels','context_edge_density','context_dark_density','context_mean_intensity','context_intensity_std','maximum_mask_depth','horizontal_fraction','vertical_fraction','compact_fraction','preserve_fraction','remove_fraction','ambiguous_fraction','reconstruction_dependency_fraction']

        for page in pages:
            z=self.regression_loadz(reg_cache[page])
            clean=cv2.imread(str(reg_clean[page]),0)
            obs=cv2.imread(str(reg_observed[page]),0)
            absent=[x for x in ['mask','telea','geometry_t80'] if x not in z]
            if absent:
                reg_missing.append({'page':page,'missing':'|'.join(absent)})
                continue
            mask=z['mask']; telea=z['telea']
            reg_stage_a.append({'page':page,'observed_shape_ok':obs is not None and clean is not None and obs.shape==clean.shape,'mask_shape_ok':mask.shape==obs.shape,'telea_shape_ok':telea.shape==obs.shape,'telea_outside_mask_immutable':bool(np.array_equal(telea[mask==0],obs[mask==0]))})
            built,br=geometry_t80(obs,mask,telea)
            ref=z['geometry_t80']
            reg_geometry.append({'page':page,'exact':bool(np.array_equal(ref,built)),'different_pixels':int(np.sum(ref!=built))})
            diff=np.abs(ref.astype(np.int16)-built.astype(np.int16)).astype(np.uint8)
            if diff.any():
                cv2.imwrite(str(self.diffs/f'{page}_geometry_absdiff.png'),diff)
            for key in ['preserve','remove','ambiguous','horizontal','vertical','compact']:
                rk=f'g80_{key}'
                if rk in z:
                    a=z[rk]>0; b=br[key]
                    reg_routes.append({'page':page,'route':key,'tested':True,'exact':bool(np.array_equal(a,b))})
            n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
            ev=reg_evidence[reg_evidence.page.astype(str)==page]
            vref=reg_v77[reg_v77.page.astype(str)==page]
            for cid in range(1,n):
                if stats[cid,cv2.CC_STAT_AREA] < self.min_component_area: continue
                comp=lab==cid
                fv=surrounding_features(obs,comp,br)
                refrow=ev[ev.component.astype(int)==cid]
                reg_components.append({'page':page,'component':cid,'reference_row_count':len(refrow)})
                for field in predictors:
                    rv=refrow.iloc[0][field] if len(refrow)==1 and field in refrow.columns else np.nan
                    bv=fv[field]
                    match=pd.notna(rv) and bool(np.isclose(float(rv),float(bv),atol=self.atol,rtol=self.rtol,equal_nan=True)) if not isinstance(bv,(bool,np.bool_)) else (pd.notna(rv) and bool(rv)==bool(bv))
                    reg_predictor_rows.append({'page':page,'component':cid,'field':field,'match':match})
                lr=vref[vref.component.astype(int)==cid]
                reg_labels.append({'page':page,'component':cid,'v7_7_rows':len(lr)})

        reg_a=pd.DataFrame(reg_stage_a)
        reg_geometry_df=pd.DataFrame(reg_geometry)
        reg_route_df=pd.DataFrame(reg_routes)
        reg_component_df=pd.DataFrame(reg_components)
        reg_pred_df=pd.DataFrame(reg_predictor_rows)
        reg_label_df=pd.DataFrame(reg_labels)

        reg_a.to_csv(self.tables/'stage_a_cache_integrity.csv',index=False)
        pd.DataFrame(reg_missing).to_csv(self.tables/'missing_arrays.csv',index=False)
        reg_geometry_df.to_csv(self.tables/'stage_b_geometry.csv',index=False)
        reg_route_df.to_csv(self.tables/'stage_b_routing.csv',index=False)
        reg_component_df.to_csv(self.tables/'stage_c_components.csv',index=False)
        reg_pred_df.to_csv(self.tables/'stage_c_predictors.csv',index=False)
        reg_label_df.to_csv(self.tables/'stage_d_v7_7_alignment.csv',index=False)

        all_route_tested=bool(len(reg_route_df) and reg_route_df.tested.all())
        checks={
            'authoritative_arrays_complete': len(reg_missing)==0,
            'stage_a_shapes_and_telea': bool(len(reg_a) and reg_a.all().all()),
            'stage_b_geometry_exact': bool(len(reg_geometry_df) and reg_geometry_df.exact.all()),
            'stage_b_routing_arrays_available': all_route_tested,
            'stage_b_routing_exact': bool(all_route_tested and reg_route_df.exact.all()),
            'stage_c_component_reference_unique': bool(len(reg_component_df) and (reg_component_df.reference_row_count==1).all()),
            'stage_c_predictors_exact': bool(len(reg_pred_df) and reg_pred_df.match.all()),
            'stage_d_v7_7_component_alignment': bool(len(reg_label_df) and (reg_label_df.v7_7_rows==1).all())
        }
        first_failure=next((k for k,v in checks.items() if not v),None)
        result={'verdict':'PASS' if first_failure is None else 'FAIL','first_failed_stage':first_failure,**checks,'v7_5_reference':str(v75),'v7_7_reference':str(v77),'canonical_structural_block_sha256':self.sha}
        (self.regression_dir/'regression_manifest.json').write_text(json.dumps(result,indent=2))
        pd.DataFrame([result]).to_csv(self.tables/'audit_verdict.csv',index=False)
        if result['verdict']!='PASS':
            raise RuntimeError(f"PILOT REVIEWER BLOCKED: {first_failure}")
        return result
