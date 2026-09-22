"""Controlled BIO swaps; existing split/poses/Reel, no fitting or promotion."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_boundary_expanded_v17 as ev
from active.v17.pretrained_boundary_v17 import Pose

OUT = ROOT / 'artifacts/reports/boundary_adaptation_diagnostic_v17_20260922'


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    if (OUT/'evaluation.json').exists():
        raise FileExistsError('preserve diagnostic results')
    supplied = json.loads((ev.REPORT/'evaluation.json').read_text())
    rows = ev.evaluation_rows(report=OUT)
    for r in rows: r['subset'] = 'asllrp12'
    local = ev.local_rows()
    for r in local: r['subset'] = 'local60'
    rows += local
    ev.atomic(OUT/'evaluation_membership.json', dict(videos=[{k:r[k] for k in
        ('source_item_id','video_path','video_sha256','target_sequence','subset')} for r in rows]))
    prepared = {r['item']:r for r in json.loads(ev.PREPARED.read_text())['records']}
    fit = {r['video_sha256'] for r in prepared.values() if r['split'] in ('train','calibration')}
    assert not fit & {r['video_sha256'] for r in rows}
    sys.path.insert(0,str(ev.UPSTREAM))
    from probe import upstream_helpers
    decode = upstream_helpers()
    args = ev.arguments(rows[0]['video_path'],OUT/'sessions'); args.no_motion_trim=True
    reel = ev.build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    assert reel.provenance() == supplied['reel']
    detector = ev.AppleVisionDetector(args.minimum_point_confidence)
    models = [ev.PretrainedBoundary().to('mps').eval() for _ in ev.ARMS]
    original = models[0].state_dict()
    weight_audit = []
    for spec, model in zip(ev.ARMS[1:],models[1:]):
        state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        assert state['recipe_sha256']==ev.digest(spec['recipe'])
        model.load_state_dict(state['model_state_dict'],strict=True)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),original[k].cpu())]
        assert all(k.startswith(('edge.','backbone.encoder_attn.')) for k in changed),changed
        weight_audit.append(dict(arm=spec['name'],changed_tensors=len(changed),
            frozen_cnn_norm_bio_exact=True,checkpoint_sha256=ev.digest(spec['checkpoint'])))
    ev.atomic(OUT/'weight_audit.json',weight_audit)
    times=torch.tensor((np.arange(ev.FRAMES)-ev.TARGET)[None]/ev.FPS,dtype=torch.float32,device='mps')
    runs=[dict(arm=spec['name']+'_original_bio',records=[]) for spec in ev.ARMS]
    baseline={r['id']:r for r in supplied['runs'][0]['records']}
    latencies=[]; started=time.perf_counter()
    for number,row in enumerate(rows):
        item=row['source_item_id']; record=prepared.get(item)
        if record is None:
            path=ev.REPORT/'evaluation_poses'/(hashlib.sha256(item.encode()).hexdigest()[:20]+'.pose')
            assert path.exists(),path
            with path.open('rb') as f: pose=Pose.read(f)
            record=dict(pose_path=str(path.relative_to(ROOT)),pose_fps=pose.body.fps,
                        pose_sha256=baseline[item]['pose_sha256'])
        assert ev.digest(ROOT/record['pose_path'])==baseline[item]['pose_sha256']
        pose=ev.load_pose(record)
        tick=time.perf_counter(); obs=ev.observations(row,args,detector); frontend=time.perf_counter()-tick
        projected=[]; tick=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(pose.body.data),16):
                x=np.stack([ev.window_features(pose,j)[0] for j in range(start,min(start+16,len(pose.body.data)))])
                projected.append(models[0].project(torch.from_numpy(x).to('mps')).cpu())
        z=torch.cat(projected); projection=time.perf_counter()-tick
        for index,(model,run) in enumerate(zip(models,runs)):
            values=[]; tick=time.perf_counter()
            with torch.inference_mode():
                for start in range(0,len(z),128):
                    h=model.encode_projected(z[start:start+128].to('mps'),times)
                    values.append(model.backbone.sign_bio_head(h).log_softmax(-1).cpu())
            values=torch.cat(values); head=time.perf_counter()-tick
            events=[dict(start_seconds=s['start']/ev.FPS,end_seconds=s['end']/ev.FPS)
                for s in decode['filter_segments'](decode['likeliest_probs_to_segments'](values))]
            tick=time.perf_counter()
            predictions=[ev.classify_interval(reel,obs,e,args,.1) for e in events]
            classify=time.perf_counter()-tick
            hyp=[p['committed_gloss'] for p in predictions if p['committed_gloss']]
            metrics=ev.edit_counts(row['target_sequence'],hyp)
            if index==0:
                assert hyp==baseline[item]['hypothesis'],('frozen reproduction',item,hyp,baseline[item]['hypothesis'])
                assert [(p['start_seconds'],p['end_seconds']) for p in predictions]==[(p['start_seconds'],p['end_seconds']) for p in baseline[item]['predictions']]
            run['records'].append(dict(id=item,subset=row['subset'],reference=row['target_sequence'],hypothesis=hyp,
                metrics=metrics,predictions=predictions,head_seconds=head,classifier_seconds=classify))
        latencies.append(dict(item=item,frames=len(z),apple_frontend_seconds=frontend,projection_seconds=projection))
        print(json.dumps(dict(completed=number+1,total=len(rows),correct=[sum(r['metrics']['correct'] for r in a['records']) for a in runs])),flush=True)
    for run in runs:
        run['summaries']={subset:ev.summarize([r for r in run['records'] if subset=='combined' or r['subset']==subset],list(baseline.values()) if subset=='combined' else None) for subset in ('asllrp12','local60','combined')}
    assert all(ev.digest(ROOT/r['video_path'])==r['video_sha256'] for r in rows)
    ev.atomic(OUT/'evaluation.json',dict(status='complete',runs=runs,reel=reel.provenance(),timings=latencies,
        elapsed_seconds=time.perf_counter()-started,frozen_reference_reproduced=True,promoted=False,
        limitation='Cached MediaPipe pose, batched bounded-window offline composition; not live or iPhone latency.'))
    print(json.dumps([dict(arm=r['arm'],**r['summaries']['combined']) for r in runs]),flush=True)

if __name__=='__main__':main()
