"""Compare isolated and connected decoding using the same 24 annotated sign cores."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from active.v17.model_stage2_v17 import load_frozen_unified_stage1, load_stage2_other_preserving
from active.v17.model_unified_multimodal_v17 import UnifiedMultimodalStage1V17
from active.v17.train_stage_2_v17 import collapse_ctc, edit_distance

OUT=Path(__file__).resolve().parent

def one(root, filename, suffix):
    paths=list((ROOT/'data/local'/root/'validation').glob('*/*'+filename+'_*'+suffix))
    assert len(paths)==1, (filename, paths)
    return paths[0]

def decode(model, features, labels):
    tensor=torch.from_numpy(features.astype(np.float32))[None]
    logits,lengths=model(tensor,torch.ones((1,len(features)),dtype=torch.bool))
    ids=collapse_ctc(logits[0,:int(lengths[0])].argmax(-1).numpy())
    return [labels[i] for i in ids if i<100]

def main():
    torch.set_num_threads(2)
    items=json.loads((OUT/'matched_manifest.json').read_text())
    manifest=json.loads((ROOT/'active/v17/citizen100_manifest.json').read_text())
    labels=[r['canonical_label'] for r in sorted(manifest['classes'],key=lambda r:r['class_index'])]
    landmark,hand,fusion,_=load_frozen_unified_stage1(ROOT/'artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth')
    stage1=UnifiedMultimodalStage1V17(landmark,hand,fusion).eval()
    repaired,payload=load_stage2_other_preserving(ROOT/'artifacts/models/stage2_v17_transition_repair_v3/seed_1702.pth')
    results=[]
    with torch.inference_mode():
        for item in items:
            cores=[]; features=[]
            for core in item['cores']:
                fn=core['sign_video_filename']
                annotated_frames=int(core['sign_end_frame'])-int(core['sign_start_frame'])+1
                if annotated_frames < 4:
                    cores.append(dict(reference=core['canonical_label'], excluded='fewer_than_four_annotated_frames', annotated_frames=annotated_frames, source_video=fn))
                    continue
                rgb=one('stage2_v17_asllrp_segmented_validation_multimodal',fn,'.stage2_rgb_v17.npz')
                hp=one('stage2_v17_asllrp_segmented_validation_hand_mobileclip2',fn,'.stage2_hand_mobileclip2_v17.npz')
                fp=one('stage2_v17_asllrp_segmented_validation_frozen_features',fn,'.stage2_frozen_v17.npz')
                with np.load(rgb,allow_pickle=False) as z: lm=z['landmarks'].astype(np.float32)
                with np.load(hp,allow_pickle=False) as z: he=z['embeddings'].astype(np.float32); hv=z['valid']; hb=z['boxes_normalized'].astype(np.float32)
                with np.load(fp,allow_pickle=False) as z:
                    ff=z['frozen_features'].astype(np.float32); md=json.loads(str(z['metadata_json']))
                assert md['role']=='validation' and md['target_sequence']==[core['canonical_label']]
                assert md['stage1_checkpoint_sha256']==payload['encoder_sha256']
                assert len(lm)==len(ff)==1, fn
                score=stage1(torch.from_numpy(lm),torch.from_numpy(he),torch.from_numpy(hv),torch.from_numpy(hb))[0]
                top=score.topk(3).indices.tolist()
                cores.append(dict(reference=core['canonical_label'],stage1=labels[top[0]],stage1_top3=[labels[i] for i in top],stage2_accepted=decode(repaired.accepted,ff,labels),stage2_repaired=decode(repaired,ff,labels),rgb_archive=str(rgb.relative_to(ROOT)),frozen_archive=str(fp.relative_to(ROOT))))
                features.append(ff)
            complete=all('stage1' in c for c in cores)
            combined=np.concatenate(features,axis=0)
            cached=[labels[i] for i in item['cached_prediction']]
            row=dict(item_id=item['item_id'],reference=item['reference'],cached_connected=cached,cores=cores,
                     oracle_core_windows=decode(repaired,combined,labels) if complete else None,
                     stitched_stage1=[c['stage1'] for c in cores] if complete else None,
                     stitched_stage2=[g for c in cores for g in c['stage2_repaired']] if complete else None)
            results.append(row)
            print(row['item_id'],row['reference'],'stage1 cores',row['stitched_stage1'],'ctc cores',row['stitched_stage2'],'whole',cached,'oracle windows',row['oracle_core_windows'],flush=True)
    metrics={}
    for key in ['cached_connected','stitched_stage1','stitched_stage2','oracle_core_windows']:
        cohort=[r for r in results if r['stitched_stage1'] is not None]
        tokens=sum(len(r['reference']) for r in cohort)
        edits=sum(edit_distance(r['reference'],r[key]) for r in cohort)
        metrics[key]=dict(edits=edits,tokens=tokens,wer=edits/tokens,exact=sum(r['reference']==r[key] for r in cohort),samples=len(cohort))
    evaluable=[c for r in results for c in r['cores'] if 'stage1' in c]
    metrics['individual_cores']=dict(samples=len(evaluable),excluded=24-len(evaluable),stage1_correct=sum(c['stage1']==c['reference'] for c in evaluable),stage2_correct=sum(c['stage2_repaired']==[c['reference']] for c in evaluable))
    (OUT/'manual_core_probe.json').write_text(json.dumps(dict(rows=results,metrics=metrics,disclosure='Development-only oracle boundaries. Segmented cores have independent extraction; cropping and duration resampling differ from full phrase. No deployment or generalization claim.'),indent=2)+'\n')
    print(metrics)

if __name__=='__main__': main()
