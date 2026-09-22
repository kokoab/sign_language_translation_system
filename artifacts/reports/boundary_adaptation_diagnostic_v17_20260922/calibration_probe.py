"""Fixed-candidate comparison on unchanged training-held calibration only."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts import evaluate_boundary_expanded_v17 as e
from scripts.train_temporal_boundary_v17 import CURATED
OUT=Path(__file__).parent

def main():
 torch.set_num_threads(2)
 if (OUT/'calibration.json').exists():raise FileExistsError('preserve results')
 approved=json.loads((OUT/'contract_audit.json').read_text())['complete_locked_vocabulary_calibration']
 ids={r['item'] for r in approved}
 prepared={r['item']:r for r in json.loads(e.PREPARED.read_text())['records']}
 rows=[r for r in json.loads(e.COMBINED.read_text())['records'] if r.get('source_item_id') in ids]
 events=json.loads(CURATED.read_text())['events']
 for r in rows:
  p=prepared[r['source_item_id']];assert p['split']=='calibration' and p['video_sha256']==r['video_sha256']
  assert e.digest(ROOT/r['video_path'])==r['video_sha256']
  r['subset']='calibration';r['all_events']=sorted([s for s in events if s['item']==r['source_item_id']],key=lambda s:s['start'])
  r['events']=[s for s in r['all_events'] if s['kind']=='known']
  assert [s['label'] for s in r['events']]==r['target_sequence']
 assert len(rows)==2 and sum(len(r['target_sequence']) for r in rows)==4
 expanded=json.loads((e.REPORT/'tcn_comparison/evaluation_membership.json').read_text())
 assert not {r['video_sha256'] for r in rows}&{r['video_sha256'] for r in expanded['videos']}
 sys.path.insert(0,str(e.UPSTREAM));from probe import upstream_helpers
 decode=upstream_helpers();args=e.arguments(rows[0]['video_path'],OUT/'calibration_sessions');args.no_motion_trim=True
 reel=e.build_components(args)['classifier'];reel.args.no_motion_trim=reel.full.args.no_motion_trim=True
 supplied=json.loads((e.REPORT/'evaluation.json').read_text());assert reel.provenance()==supplied['reel']
 detector=e.AppleVisionDetector(args.minimum_point_confidence)
 times=torch.tensor((np.arange(e.FRAMES)-e.TARGET)[None]/e.FPS,dtype=torch.float32,device='mps')
 model=e.PretrainedBoundary().to('mps').eval();started=time.perf_counter()
 cache=e.cached_inputs(rows,model,args,detector,OUT/'unused_pose_output')
 runs=[];base=None
 for spec in e.ARMS:
  for readout in (('bio',) if not spec['checkpoint'] else ('bio','edges')):
   arm,records,matched,candidates=e.run_arm(spec,rows,cache,model,args,reel,times,decode,readout=readout)
   if base is None:base=records
   summary=e.summarize(records,base)
   runs.append(dict(arm=spec['name'],readout=readout,summary=summary,records=records))
 baseline=runs[0]['summary'];eligible=[r for r in runs[1:] if r['summary']['wer']<baseline['wer'] and r['summary']['retained_baseline_correct_events']>=baseline['correct'] and r['summary']['insertions']<=baseline['insertions']]
 selected=min(eligible,key=lambda r:(r['summary']['wer'],-r['summary']['correct'],r['summary']['insertions'])) if eligible else runs[0]
 e.atomic(OUT/'calibration.json',dict(status='complete',runs=runs,selected=dict(arm=selected['arm'],readout=selected['readout']),
  elapsed_seconds=time.perf_counter()-started,reel=reel.provenance(),promoted=False,
  limitation='Two fixed training-held calibration videos/four signs only. No threshold/epoch search, no gradient steps. Too small for a robust deployment decision. Expanded videos excluded.'))
 print(json.dumps([dict(arm=r['arm'],readout=r['readout'],**r['summary']) for r in runs]))

if __name__=='__main__':main()
