"""Matched Reel core/context/gap diagnostic on approved development clips; no training."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
from scripts.live_reel_stage1_v17 import parser,ReelCascadeClassifier,VerifiedCommitLock
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from active.v17.extract_v17 import AppleVisionDetector
from scripts.diagnose_combined_transitions_v17 import selected_rows,known_gap
OUT=Path(__file__).parent

def small(r):
 return {k:r.get(k) for k in ['candidate_gloss','gloss','model_score','accepted','rejection_reasons','top3','frames','motion_trimmed_frames']}

def self_check():
 deadline=0.;kept=[]
 for index in range(30):
  t=index/30
  if t+1e-6<deadline:continue
  kept.append(t);deadline=max(deadline+1/20,t)
 assert len(kept)==20, len(kept)
 assert known_gap(.3,.5,[{'start':.1,'end':.2}])
 assert not known_gap(.3,.5,[{'start':.4,'end':.6}])

def main():
 self_check()
 torch.set_num_threads(2)
 a=parser().parse_args(['--no-display','--no-speech','--naturalizer','literal'])
 model=ReelCascadeClassifier(a);detector=AppleVisionDetector(a.minimum_point_confidence)
 m=json.loads((ROOT/'data/local/combined_dataset_v17_20260922/manifest.json').read_text());cur=json.loads((ROOT/'artifacts/reports/clean_boundary_subset_20260920/curated_manifest.json').read_text())
 rows,events,cores=selected_rows(m,cur);results=[];skipped=[]
 for row in [r for r in rows if r['role']=='validation']:
  path=ROOT/row['video_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['video_sha256']
  cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);obs=[];wrists={'left':None,'right':None};deadline=0.;index=0
  if not cap.isOpened() or fps<=0:raise ValueError(str(path))
  while True:
   ok,f=cap.read()
   if not ok:break
   t=index/fps;index+=1
   if t+1e-6<deadline:continue
   o=observe_stage2_frame(f,t,len(obs),detector,wrists,a);obs.append(o);deadline=max(deadline+1/a.processing_fps,t)
  cap.release();all_events=events[row['source_item_id']]
  windows=[]
  for c in cores[row['source_item_id']]:
   windows.extend([(c['identity'],'core',c['label'],c['start'],c['end']),
                   (c['identity'],'context_100ms',c['label'],max(0,c['start']-.1),c['end']+.1),
                   (c['identity'],'context_250ms',c['label'],max(0,c['start']-.25),c['end']+.25)])
  ordered=sorted(all_events,key=lambda e:e['start'])
  for left,right in zip(ordered,ordered[1:]):
   start,end=left['end']+.02,right['start']-.02
   if left['kind']=='known' and right['kind']=='known' and end>start and known_gap(start,end,all_events):
    windows.append((left['identity']+':gap','gap',None,start,end))
  for ident,kind,target,start,end in windows:
   chosen=[o for o in obs if start<=o.seconds<=end]
   if len(chosen)<4:
    skipped.append(dict(id=ident,kind=kind,frames=len(chosen)));continue
   p=model.classify(chosen);v=model.verify(chosen)
   agreement=p['model_score'] if p.get('candidate_gloss')==v.get('candidate_gloss') else 0
   lock=VerifiedCommitLock(a.commit_hits,a.instant_commit_score)
   commit=bool(p.get('accepted') and v.get('accepted')) and lock.update(str(v.get('candidate_gloss')),max(v['model_score'],agreement),proposal=str(p.get('candidate_gloss')),proposal_score=p['model_score'],minimum_score=a.commit_score)
   motion=[o.hand_quality>0 and o.motion>=a.start_motion for o in chosen];run=best=0
   for move in motion:run=run+1 if move else 0;best=max(best,run)
   results.append(dict(id=ident,video=row['video_path'],kind=kind,target=target,start=start,end=end,proposal=small(p),verifier=small(v),conditional_commit=commit,motion_start_possible_within_window=best>=a.start_frames))
  print(row['source_item_id'],len(results),flush=True)
 summary={}
 for kind in ['core','context_100ms','context_250ms','gap']:
  rs=[r for r in results if r['kind']==kind];summary[kind]=dict(n=len(rs),proposal_correct=sum(r['proposal']['candidate_gloss']==r['target'] for r in rs) if kind!='gap' else None,verifier_correct=sum(r['verifier']['candidate_gloss']==r['target'] for r in rs) if kind!='gap' else None,conditional_commits=sum(r['conditional_commit'] for r in rs),correct_conditional_commits=sum(r['conditional_commit'] and r['verifier']['candidate_gloss']==r['target'] for r in rs) if kind!='gap' else None,motion_start_possible=sum(r['motion_start_possible_within_window'] for r in rs))
 payload=dict(summary=summary,results=results,skipped=skipped,models=model.provenance(),settings={k:getattr(a,k) for k in ['processing_fps','start_motion','start_frames','minimum_score','minimum_margin','commit_score','commit_hits','instant_commit_score']},limitations=['annotated existing development, not latest user recording','context may legitimately contain neighboring sign; differences not automatically errors','conditional commit bypasses candidate scheduler and proposal stability; not live emissions','pure gaps only evaluated if at least four observed frames; excludes uncertain annotation overlap'],test_accessed=False)
 (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
