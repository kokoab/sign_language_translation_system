"""Read-only model comparison on the existing expanded membership; no fitting."""
import json,sys,time
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import torch
from active.v17.approved_phrase_data_v17 import digest
from active.v17.extract_v17 import AppleVisionDetector
from scripts.evaluate_boundary_expanded_v17 import local_rows,SPLIT,COMBINED,BASELINE,ARMS
from scripts.evaluate_temporal_boundary_v17 import evaluation_rows,arguments,observations,edit_counts,summarize,transition_commits
from scripts.live_boundary_v17 import BoundaryRecognizer,classify_interval
from scripts.live_reel_stage1_v17 import build_components
from scripts.train_temporal_boundary_v17 import atomic
OUT=Path(__file__).parent;EXPANDED=OUT.parent

def main():
 torch.set_num_threads(2)
 rows=evaluation_rows(report=OUT)
 for r in rows:r['subset']='asllrp12'
 local=local_rows()
 for r in local:r['subset']='local60'
 rows+=local
 supplied=json.loads((EXPANDED/'evaluation.json').read_text())
 temporal=json.loads((ROOT/'active/v17/temporal_boundary_manifest_20260922.json').read_text())
 prepared=json.loads((ROOT/'artifacts/reports/pose_boundary_transfer_v17_20260922/prepared_manifest.json').read_text())
 fit={r['video_sha256'] for r in temporal['records'] if r['split'] in ('train','calibration')}
 posefit={r['video_sha256'] for r in prepared['records'] if r['split'] in ('train','calibration')}
 current={r['source_item_id']:r for r in rows};assert len(current)==72
 audit=dict(videos=len(rows),references=sum(len(r['target_sequence']) for r in rows),local_signers=dict(Counter(str(r.get('signer_id',r.get('signer'))) for r in local)),
    boundary_train_or_calibration_overlap=[r['source_item_id'] for r in rows if r['video_sha256'] in fit|posefit],
    supplied_membership_videos=len(json.loads((EXPANDED/'evaluation_membership.json').read_text())['videos']),
    source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (SPLIT,COMBINED,BASELINE,EXPANDED/'evaluation.json',ROOT/'active/v17/temporal_boundary_manifest_20260922.json')},
    metric_discrepancies=[],verified_summaries={})
 assert not audit['boundary_train_or_calibration_overlap']
 for arm in supplied['runs']:
  assert {r['id'] for r in arm['records']}==set(current)
  assert len(arm['records'])==72
  for r in arm['records']:
   assert r['reference']==current[r['id']]['target_sequence']
   assert r['hypothesis']==[p['committed_gloss'] for p in r['predictions'] if p['committed_gloss']]
   assert json.loads(json.dumps(edit_counts(r['reference'],r['hypothesis'])))==r['metrics']
  if arm['checkpoint']:assert digest(Path(arm['checkpoint']))==arm['checkpoint_sha256']
  summaries={}
  for subset in ('asllrp12','local60','combined'):
   records=[r for r in arm['records'] if subset=='combined' or r['subset']==subset]
   recomputed=summarize(records)
   for k in ('substitutions','deletions','insertions','references','correct','wer','exact_videos','videos'):
    assert recomputed[k]==arm['summaries'][subset][k],(arm['arm'],subset,k)
   summaries[subset]=recomputed
  actual=sum(len(r['predictions']) for r in arm['records'] if r['subset']=='asllrp12')
  reported=arm['summaries']['asllrp12']['boundary_candidates']
  if actual!=reported:audit['metric_discrepancies'].append(dict(arm=arm['arm'],field='asllrp12.boundary_candidates',reported=reported,recomputed=actual))
  audit['verified_summaries'][arm['arm']]=summaries
 atomic(OUT/'audit.json',audit)
 membership=dict(videos=[{k:r[k] for k in ('source_item_id','video_path','video_sha256','target_sequence','subset')} for r in rows],
  inputs=audit['source_sha256'],scope='same72 reused development clips; no training; local intervals unknown')
 atomic(OUT/'evaluation_membership.json',membership)
 training=json.loads((ROOT/'artifacts/reports/asl_temporal_boundary_v17_20260922/training_results.json').read_text())['runs']
 args=arguments(rows[0]['video_path'],OUT/'sessions');args.no_motion_trim=True
 reel=build_components(args)['classifier'];reel.args.no_motion_trim=reel.full.args.no_motion_trim=True
 if reel.provenance()!=supplied['reel']:raise ValueError('Reel provenance differs from expanded evaluation')
 models=[]
 for trained in training:
  assert digest(ROOT/trained['checkpoint'])==trained['checkpoint_sha256']
  args.boundary_checkpoint=ROOT/trained['checkpoint']
  models.append((trained,BoundaryRecognizer(args,reel=reel),[]))
 detector=AppleVisionDetector(args.minimum_point_confidence)
 base=json.loads(BASELINE.read_text())['results']['reel'];started=time.perf_counter()
 for index,row in enumerate(rows):
  obs=observations(row,args,detector)
  for trained,model,records in models:
   model.reset();events=[]
   for observation in obs:
    tick=model.observe(observation)
    if tick:events.extend(tick['events'])
   predictions=[classify_interval(reel,obs,event,args,.1) for event in events]
   for p in predictions:p['uses_eof_partial_context']=False
   hyp=[p['committed_gloss'] for p in predictions if p['committed_gloss']]
   records.append(dict(id=row['source_item_id'],subset=row['subset'],reference=row['target_sequence'],hypothesis=hyp,
       metrics=edit_counts(row['target_sequence'],hyp),predictions=predictions,transitions=transition_commits(row,predictions)))
  print(json.dumps(dict(completed=index+1,total=len(rows),item=row['source_item_id'])),flush=True)
  output=[]
  for trained,model,records in models:
   summaries={name:summarize([r for r in records if name=='combined' or r['subset']==name],base if name=='asllrp12' else None) for name in ('asllrp12','local60','combined') if any(name=='combined' or r['subset']==name for r in records)}
   output.append(dict(training=trained,summaries=summaries,records=records))
  atomic(OUT/'evaluation.json',dict(status='running',completed=index+1,runs=output,reel=reel.provenance(),promoted=False))
 for r in rows:assert digest(ROOT/r['video_path'])==r['video_sha256']
 for p,sha in audit['source_sha256'].items():assert digest(ROOT/p)==sha
 # Retention relative to the strongest expanded reference, without using it for fitting.
 frozen={r['id']:r for r in supplied['runs'][0]['records']}
 for result in output:
  for subset in ('asllrp12','local60','combined'):
   records=[r for r in result['records'] if subset=='combined' or r['subset']==subset]
   retained=total=0
   for r in records:
    old={i for i,j in frozen[r['id']]['metrics']['matched_pairs']};new={i for i,j in r['metrics']['matched_pairs']}
    total+=len(old);retained+=len(old&new)
   result['summaries'][subset].update(frozen_pretrained_correct=total,retained_frozen_pretrained_correct=retained,
     wholly_gap_contained_commits=sum(r['transitions']['wholly_gap_contained_commits'] for r in records) if subset=='asllrp12' else None,
     interval_metrics='ASLLRP annotation only; unavailable for local60')
 atomic(OUT/'evaluation.json',dict(status='complete',completed=len(rows),runs=output,reel=reel.provenance(),elapsed_seconds=time.perf_counter()-started,promoted=False,
    limitation='Conditional full-video comparison uses native BoundaryRecognizer.observe plus same frozen Reel classify_interval (.1s context), not asynchronous live scheduler. TCN uses .2s future and drops unscored tail; pretrained uses .5s with partial EOF padding. No local interval/gap annotation.'))
 print('COMPLETE',flush=True)

if __name__=='__main__':main()
