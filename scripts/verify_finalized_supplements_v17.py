"""Verify pinned supplements and frozen Stage1 compatibility on MPS, without training."""
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.finalize_supplements_v17 import OUT,digest,feature_check
from active.v17.model_v17 import SLTStage1V17,Stage1V17Config

def run():
    assert torch.backends.mps.is_available(),'MPS required'
    torch.set_num_threads(2)
    checkpoint='artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/best_model.pth'
    base=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
    model=SLTStage1V17(Stage1V17Config(**base['model_config']))
    model.load_state_dict(base['model_state_dict']);model.eval().requires_grad_(False).to('mps')
    summary=json.loads((OUT/'summary.json').read_text())
    baseline=json.loads((ROOT/'active/v17/approved_phrase_manifest_20260921_v2.json').read_text())
    baseline_hashes={r['video_sha256'] for r in baseline['admitted']}
    o5s5=json.loads((ROOT/'artifacts/reports/o5s5_citizen100_v17/apple_vision_supervision.json').read_text())
    o5s5_hashes={r['source_item_id']:r['video_sha256'] for r in o5s5['rows']}
    baseline_parents={}
    for name in ['asllrp_contiguous_phrases_v17','asllrp_other_ctc_v17']:
        manifest=json.loads((ROOT/f'data/local/{name}/manifest.json').read_text())
        for span in manifest['spans']:
            for row in baseline['admitted']:
                if row['video_sha256']==span['sha256']:
                    baseline_parents.setdefault(span['utterance_video_filename'],set()).add(row['role'])
    parent_overlap=[]
    pending=[];windows=0;records=0;duplicates=[];seen={};overlap=Counter();recovered=[]
    def forward():
        nonlocal windows
        if not pending:return
        with torch.inference_mode():
            y=model(torch.from_numpy(np.stack(pending).astype(np.float32)).to('mps')).cpu()
        assert y.shape==(len(pending),100) and torch.isfinite(y).all()
        windows+=len(pending);pending.clear()
    for name,sha in summary['manifests'].items():
        path=OUT/name
        assert digest(str(path.relative_to(ROOT)))==sha
        data=json.loads(path.read_text())
        for evidence,expected in data['evidence_sha256'].items():assert digest(evidence)==expected
        assert dict(Counter(r['role'] for r in data['records']))==data['counts']
        for r in data['records']:
            assert digest(r['feature_path'])==r['feature_sha256']
            assert digest(r['video_path'])==r['video_sha256']
            if r['source']=='o5s5':assert r['video_sha256']==o5s5_hashes[r['source_item_id']]
            assert r['class_index']==base['label_to_index'][r['canonical_label']]
            assert r['ctc_target']==r['class_index']+1
            key=(r['video_sha256'],r.get('start_seconds'),r.get('end_seconds'),json.dumps(r.get('interval_sampling'),sort_keys=True))
            assert key not in seen,(seen.get(key),r['feature_path'])
            seen[key]=r['feature_path']
            if r['video_sha256'] in baseline_hashes:duplicates.append(r['feature_path'])
            if r.get('parent_video') in baseline_parents:
                parent_overlap.append(dict(feature_path=r['feature_path'],role=r['role'],
                                           baseline_roles=sorted(baseline_parents[r['parent_video']]),parent_video=r['parent_video']))
            if r['signer_overlap_with_source_train']:overlap[r['source']]+=1
            if 'original_feature_path' in r:recovered.append(r['feature_path'])
            with np.load(ROOT/r['feature_path'],allow_pickle=False) as d:
                x=d['landmarks'] if r['representation']=='windowed_landmarks_v17' else d['features']
                feature_check(x)
                for window in x.reshape(-1,32,61,5):
                    pending.append(window)
                    if len(pending)==32:forward()
            records+=1
        print(name,len(data['records']),'verified',flush=True)
    forward()
    assert records==summary['accepted']
    result=dict(records_verified=records,windows_mps_forward_verified=windows,device='mps',optimizer_steps=0,
                finite_100_class_outputs=True,checkpoint_sha256=digest(checkpoint),
                exact_video_hash_overlap_with_494=duplicates,source_local_validation_signer_overlap=dict(overlap),
                recovered_files=recovered,learnability_or_accuracy_measured=False)
    result['asllrp_shared_parent_with_baseline']=parent_overlap
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
