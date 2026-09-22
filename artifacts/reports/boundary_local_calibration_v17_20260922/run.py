"""Bounded fixed-weight BIO calibration and separately held confirmation; no training."""
import json,sys,hashlib,time
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts import evaluate_boundary_expanded_v17 as e
from scripts.train_temporal_boundary_v17 import atomic
OUT=Path(__file__).parent

def scores(records,baseline):
 s=e.summarize(records,baseline)
 s['retention']=s['retained_baseline_correct_events']/s['baseline_correct_events'] if s['baseline_correct_events'] else 1.
 return s

def eligible(s,b):
 return s['wer']<b['wer'] and s['correct']>=b['correct'] and s['insertions']<=b['insertions'] and s['retention']>=.98

def shorter_filtered(records,minimum):
 result=[]
 for r in records:
  predictions=[p for p in r['predictions'] if (p['end_seconds']-p['start_seconds'])*20+1>=minimum-1e-7]
  hyp=[p['committed_gloss'] for p in predictions if p['committed_gloss']]
  result.append(dict(r,predictions=predictions,hypothesis=hyp,metrics=e.edit_counts(r['reference'],hyp)))
 return result

def main():
 torch.set_num_threads(2);started=time.perf_counter()
 if (OUT/'manifest.json').exists():raise FileExistsError('preserve prior split/results')
 split=json.loads(e.SPLIT.read_text());combined=json.loads(e.COMBINED.read_text())
 assert e.digest(ROOT/split['source_manifest'])==split['source_manifest_sha256']
 allowed={r['video_sha256'] for r in split['records'] if r.get('source')=='local_phrases' and r['experiment_role']=='train'}
 excluded={r['video_sha256'] for r in json.loads((e.REPORT/'tcn_comparison/evaluation_membership.json').read_text())['videos']}
 fit={r['video_sha256'] for r in json.loads(e.PREPARED.read_text())['records'] if r['split'] in ('train','calibration')}
 groups=defaultdict(list)
 for r in combined['records']:
  if r.get('source')=='local_phrases' and r['video_sha256'] in allowed:
   assert r['video_sha256'] not in excluded|fit
   groups[(r['signer_id'],tuple(r['target_sequence']))].append(r)
 phases={'calibration':[],'confirmation':[]};unused=[]
 for key,group in sorted(groups.items()):
  group=sorted(group,key=lambda r:hashlib.sha256(('bio-calibration-20260922:'+r['video_sha256']).encode()).hexdigest())
  assert len({r['video_sha256'] for r in group})==len(group)
  n=min(4,len(group)-2)
  phases['calibration']+=group[:n];phases['confirmation']+=group[n:n+2];unused+=group[n+2:]
 rows=sum(phases.values(),[])
 assert len({r['video_sha256'] for r in rows})==len(rows)
 for role,items in phases.items():
  for r in items:
   assert e.digest(ROOT/r['video_path'])==r['video_sha256']
   assert e.digest(ROOT/r['feature_path'])==r['feature_sha256']
   r.update(subset=role,all_events=[],events=[])
 paths=[e.SPLIT,e.COMBINED,e.PREPARED,e.REPORT/'tcn_comparison/evaluation_membership.json',Path(__file__),
        ROOT/'scripts/evaluate_boundary_expanded_v17.py',ROOT/'active/v17/pretrained_boundary_v17.py',ROOT/'scripts/live_boundary_v17.py',
        ROOT/'scripts/evaluate_temporal_boundary_v17.py']
 provenance={str(p.relative_to(ROOT)):e.digest(p) for p in paths}
 manifest=dict(status='frozen_before_inference',seed='bio-calibration-20260922',phases=phases,unused_clips=len(unused),
  candidates=['frozen_min3','frozen_min5','adapted_min3','adapted_min5'],
  policy='Up to4 calibration +2 confirmation per signer/phrase stratum; deterministic video hash ranking; confirmation never tunes selection.',
  selection='Lower calibrationWER, >=baseline correct, <=baseline insertions, >=98% retained baseline positions; otherwise frozen_min3. Confirmation must independently satisfy same rule.',
  limitations='Familiar-signer repeated-phrase development. No session IDs/near-duplicate certification; exact hashes unique. Transcript provenance validated, not new human relabeling. 6templates/15glosses. No interval/gap labels.',
  hashes=provenance,test_accessed=False,training=False)
 atomic(OUT/'manifest.json',manifest)
 print('split', {k:(len(v),sum(len(r['target_sequence']) for r in v)) for k,v in phases.items()},flush=True)
 sys.path.insert(0,str(e.UPSTREAM));from probe import upstream_helpers
 decode=upstream_helpers();args=e.arguments(rows[0]['video_path'],OUT/'sessions');args.no_motion_trim=True
 reel=e.build_components(args)['classifier'];reel.args.no_motion_trim=reel.full.args.no_motion_trim=True
 assert reel.provenance()==json.loads((e.REPORT/'evaluation.json').read_text())['reel']
 detector=e.AppleVisionDetector(args.minimum_point_confidence)
 times=torch.tensor((np.arange(e.FRAMES)-e.TARGET)[None]/e.FPS,dtype=torch.float32,device='mps')
 selected='frozen_min3';all_results={}
 for phase,items in phases.items():
  pose_dir=OUT/(phase+'_poses');pose_dir.mkdir()
  model=e.PretrainedBoundary().to('mps').eval()
  cache=e.cached_inputs(items,model,args,detector,pose_dir)
  candidates={};summaries={}
  for name,spec in [('frozen',e.ARMS[0]),('adapted',e.ARMS[1])]:
   if phase=='confirmation' and name=='adapted' and not selected.startswith('adapted'):continue
   _,records,_,_=e.run_arm(spec,items,cache,model,args,reel,times,decode,readout='bio')
   for minimum in (3,5):
    key=f'{name}_min{minimum}'
    if phase=='confirmation' and key not in ('frozen_min3',selected):continue
    candidates[key]=shorter_filtered(records,minimum)
    summaries[key]=scores(candidates[key],candidates['frozen_min3'])
  if phase=='calibration':
   baseline=summaries['frozen_min3']
   viable=[k for k,s in summaries.items() if eligible(s,baseline)]
   selected=min(viable,key=lambda k:(summaries[k]['wer'],summaries[k]['insertions'],-summaries[k]['correct'],k)) if viable else 'frozen_min3'
   atomic(OUT/'selection.json',dict(selected=selected,rule=manifest['selection'],summaries=summaries,locked_before_confirmation=True))
  all_results[phase]=dict(selected=selected,summaries=summaries,records=candidates)
  atomic(OUT/(phase+'.json'),all_results[phase]);print(phase,selected,summaries,flush=True)
  del cache,model;torch.mps.empty_cache()
 for r in rows:assert e.digest(ROOT/r['video_path'])==r['video_sha256']
 for p,sha in provenance.items():assert e.digest(ROOT/p)==sha
 confirmation=all_results['confirmation']['summaries'];passed=selected!='frozen_min3' and eligible(confirmation[selected],confirmation['frozen_min3'])
 atomic(OUT/'completion.json',dict(status='complete',selected=selected,confirmation_passed=passed,elapsed_seconds=time.perf_counter()-started,reel=reel.provenance(),promoted=False))
 lines=['# Local BIO calibration and confirmation','','No training or deployment. Selection frozen before confirmation.','',
  '| Phase | Candidate | WER | Correct | Insertions | Retained baseline |','|---|---|---:|---:|---:|---:|']
 for phase,r in all_results.items():
  for key,s in r['summaries'].items():lines.append(f"| {phase} | {key} | {s['wer']:.2%} | {s['correct']}/{s['references']} | {s['insertions']} | {s['retained_baseline_correct_events']}/{s['baseline_correct_events']} |")
 lines+=['',f'Selected: **{selected}**. Confirmation improvement gate passed: **{passed}**.',
  f"Elapsed: {(time.perf_counter()-started)/60:.1f} minutes. Calibration{len(phases['calibration'])} clips; confirmation{len(phases['confirmation'])}; unused{len(unused)}.",
  '',manifest['limitations'],'','Both backbones use original BIO, same500mslookahead/EOF handling, frozen Reel and100ms classifier context. Only minimum segment length3 versus5frames changes. Filtering reuses identical per-interval predictions; no repeated extraction or classifier work for the duration variants.',
  'Original371/60roles unchanged. New experiment only reserves the selected local clips from future boundary fitting; does not authorize training unused clips.72video expanded set untouched. No interval-level transition claim or unseen-signer/100sign accuracy claim.',
  'Hashes and transcript membership verified against approved manifests; natural-recording near duplicates/session linkage and prior Reel-training exposure remain limitations. Evidence is relative fixed-Reel calibration, not a fresh whole-pipeline test.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
