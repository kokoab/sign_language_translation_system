"""Paired full-video bounded-context evaluation with frozen Reel; cached poses, not live latency."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from active.v17.pretrained_boundary_v17 import UPSTREAM,PretrainedBoundary,load_pose,window_features,FPS,FRAMES,TARGET
from active.v17.temporal_boundary_v17 import BoundaryDecoder
from active.v17.approved_phrase_data_v17 import digest
from scripts.train_pretrained_boundary_v17 import REPORT,RECIPE,PREPARED,verified_recipe
from scripts.train_temporal_boundary_v17 import atomic
from scripts.evaluate_temporal_boundary_v17 import evaluation_rows,arguments,observations,edit_counts,summarize,transition_commits
from scripts.live_boundary_v17 import classify_interval
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector


def evaluate(report=REPORT,recipe_path=RECIPE):
    REPORT=report;RECIPE=recipe_path
    recipe=verified_recipe(RECIPE)
    sys.path.insert(0,str(UPSTREAM))
    from probe import upstream_helpers
    from prepare import extract
    decode=upstream_helpers()
    rows=evaluation_rows(report=REPORT);prepared={r['item']:r for r in json.loads(PREPARED.read_text())['records']}
    base=json.loads((UPSTREAM.parent/'asl_temporal_boundary_v17_20260922/baseline_oracle.json').read_text())['results']['reel']
    args=arguments(rows[0]['video_path'],REPORT/'evaluation_sessions');args.no_motion_trim=True
    reel=build_components(args)['classifier'];reel.args.no_motion_trim=reel.full.args.no_motion_trim=True
    detector=AppleVisionDetector(args.minimum_point_confidence)
    model=PretrainedBoundary().to('mps').eval();times=torch.tensor((np.arange(FRAMES)-TARGET)[None]/FPS,dtype=torch.float32,device='mps')
    cache={};cache_dir=REPORT/'evaluation_poses';cache_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for row in rows:
            item=row['source_item_id'];record=prepared.get(item)
            if digest(ROOT/row['video_path'])!=row['video_sha256']:raise ValueError('evaluation video changed')
            if record is not None and record['video_sha256']!=row['video_sha256']:raise ValueError('cached pose/video identity mismatch')
            if record is None:
                # Existing approved development video, not a new training admission.
                record=extract(dict(item=item,role='validation',video_path=row['video_path'],video_sha256=row['video_sha256'],
                                    intervals=[[e['start'],e['end']] for e in row['events']]),cache_dir)
            if digest(ROOT/record['pose_path'])!=record['pose_sha256']:raise ValueError('evaluation pose changed')
            pose=load_pose(record);projected=[];tick=time.perf_counter()
            for start in range(0,len(pose.body.data),16):
                x=np.stack([window_features(pose,j)[0] for j in range(start,min(start+16,len(pose.body.data)))])
                projected.append(model.project(torch.from_numpy(x).to('mps')).cpu())
            cache[item]=dict(projected=torch.cat(projected),observations=observations(row,args,detector),
                             projection_seconds=time.perf_counter()-tick,pose_sha256=record['pose_sha256'])
    training=json.loads((REPORT/'training_results.json').read_text())['runs']
    arms=[dict(seed=None,checkpoint=None),*training];output=[]
    for arm in arms:
        if arm['checkpoint']:
            if digest(ROOT/arm['checkpoint'])!=arm['checkpoint_sha256']:raise ValueError('checkpoint changed')
            state=torch.load(ROOT/arm['checkpoint'],map_location='cpu',weights_only=False)
            if state['recipe_sha256']!=digest(RECIPE):raise ValueError('wrong checkpoint recipe')
            model.load_state_dict(state['model_state_dict'],strict=True)
        model.eval();records=[];matched=0;candidates=0
        for row in rows:
            cached=cache[row['source_item_id']];z=cached['projected'];outputs=[];tick=time.perf_counter()
            with torch.inference_mode():
                for start in range(0,len(z),128):
                    h=model.encode_projected(z[start:start+128].to('mps'),times)
                    value=model.edge(h).sigmoid() if arm['checkpoint'] else model.backbone.sign_bio_head(h).log_softmax(-1)
                    outputs.append(value.cpu())
            values=torch.cat(outputs);head_seconds=time.perf_counter()-tick
            if arm['checkpoint']:
                decoder=BoundaryDecoder();events=[]
                for i,p in enumerate(values.numpy()):events.extend(decoder.update(i/FPS,p))
            else:
                events=[dict(start_seconds=s['start']/FPS,end_seconds=s['end']/FPS) for s in decode['filter_segments'](decode['likeliest_probs_to_segments'](values))]
            predictions=[classify_interval(reel,cached['observations'],event,args,.1) for event in events]
            for p in predictions:p['uses_eof_partial_context']=p['end_seconds']+.5>(len(z)-1)/FPS
            hyp=[p['committed_gloss'] for p in predictions if p['committed_gloss']]
            unmatched=set(range(len(row['events'])))
            for event in events:
                choices=[i for i in unmatched if abs(event['start_seconds']-row['events'][i]['start'])<=.2 and abs(event['end_seconds']-row['events'][i]['end'])<=.2]
                if choices:unmatched.remove(min(choices));matched+=1
            candidates+=len(events)
            records.append(dict(id=row['source_item_id'],reference=row['target_sequence'],hypothesis=hyp,
                                metrics=edit_counts(row['target_sequence'],hyp),predictions=predictions,
                                transitions=transition_commits(row,predictions),head_seconds=head_seconds,
                                projection_seconds=cached['projection_seconds'],pose_sha256=cached['pose_sha256']))
        summary=summarize(records,base)
        summary.update(boundary_candidates=candidates,boundary_matches_200ms=matched,boundary_recall_200ms=matched/summary['references'],
                       wholly_gap_contained_commits=sum(r['transitions']['wholly_gap_contained_commits'] for r in records),
                       committed_with_eof_partial_context=sum(bool(p['committed_gloss']) and p['uses_eof_partial_context'] for r in records for p in r['predictions']))
        output.append(dict(arm='frozen_pretrained_bio' if not arm['checkpoint'] else 'adapted_start_end',training=arm,summary=summary,records=records))
        atomic(REPORT/'evaluation.json',dict(runs=output,reel=reel.provenance(),promoted=False,
            limitation='Reused12development videos; cached pose conditional offline composition,not live timing. FrozenBIO groups adjacent B/I and closes EOF; adapted decoder requires predicted END. EOF context is explicitly padded/recorded.'))
        print(json.dumps(dict(seed=arm['seed'],**summary)),flush=True)
    if any(digest(ROOT/r['video_path'])!=r['video_sha256'] for r in rows):raise ValueError('video changed during evaluation')
    lines=['# Pretrained ASL boundary adaptation','',
           'No automatic promotion. Same12reused development videos/24known signs; frozen Reel.',
           'Cached-pose bounded-context evaluation is not a live latency or phone result.','',
           '| Arm | Epochs / selected | Correct | WER | Retained baseline |',
           '| --- | --- | --- | --- | --- |']
    for r in output:
        a,s=r['training'],r['summary'];lines.append(f"| {r['arm']} {a['seed']} | {a.get('epochs','—')} / {a.get('selected_epoch','—')} | {s['correct']}/24 | {s['wer']:.2%} | {s['retained_baseline_correct_events']}/4 |")
    lines+=['','All approved supervised20Hz windows used.5head warm-up epochs then all4attentionblocks adapted; CNN cached/frozen.',
            f"Maximum{recipe['maximum_epochs']}epochs,minimum{recipe['minimum_epochs']},patience{recipe['patience']}. Checkpoints selected only on train-parent calibration.",
            f"Training augmentation: {recipe.get('augmentation', 'none')}. Held-out windows remain unaugmented.",
            'Unknown gaps masked; explicit EOF partial-context predictions tracked. Original frozen BIO and adapted START/END decoders differ.',
            'See training_results.json,history files,evaluation.json and completion.json. Do not infer unseen-signer or real low-motion/repeat accuracy from this small replay.']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':evaluate()
