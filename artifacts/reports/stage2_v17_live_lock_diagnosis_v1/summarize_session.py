"""Auditable session counts and fixed-proposal gate counterfactuals (no model calls)."""
import hashlib,json
from collections import Counter
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
P=ROOT/'artifacts/reports/live_reel_continuous_v17/20260910_071844_645159/history.json'

def main():
 x=json.loads(P.read_text());p=x['predictions'];events=x['events']
 proposals=[r for r in p if r.get('lock_proposal')]
 finishes=[e for e in events if e['type']=='finished_sequence_selected']
 windows={}
 for r in p:windows.setdefault(str(r['start_seconds']),[]).append(r)
 trajectory=[];current=[]
 for e in events:
  if e['type']=='stage2_sequence_update':current.append(e)
  elif e['type']=='finished_sequence_selected':trajectory.append(dict(finish=e,updates=current));current=[]
 stats=dict(history=str(P.relative_to(ROOT)),history_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),source=x['source'],config=x['config'],capture=x.get('capture_stats'),
  probes=len(p),accepted=sum(bool(r['accepted']) for r in p),rejection_reasons=dict(Counter(v for r in p for v in r.get('rejection_reasons',[]))),
  stable_proposals=len(proposals),commits=sum(r.get('committed_gloss') is not None for r in p),events=dict(Counter(e['type'] for e in events)),
  hand_frame_fraction=dict(mean=statistics.mean(r['diagnostics']['hand_frame_fraction'] for r in p),minimum=min(r['diagnostics']['hand_frame_fraction'] for r in p),rejected_mean=statistics.mean(r['diagnostics']['hand_frame_fraction'] for r in p if not r['accepted'])),
  windows=[dict(start=float(s),end=max(r['end_seconds'] for r in rs),probes=len(rs),labels=[r['gloss'] for r in rs],commits=[r['committed_gloss'] for r in rs if r.get('committed_gloss')]) for s,rs in windows.items()],
  finish_trajectories=trajectory,proposal_counterfactuals=[],limitations=['User intended transcript not recorded: no user-session WER assigned.', 'Fixed recorded proposals only: counterfactuals do not simulate changed future boundaries, inference timing, or newly requested verifier outputs.','Hand detection coverage does not establish correct coordinates or visibility of all distinguishing features.'])
 for r in proposals:
  v=r['full_verifier'];same=r['lock_proposal']==v['candidate_gloss']
  stats['proposal_counterfactuals'].append(dict(start=r['start_seconds'],end=r['end_seconds'],proposal=r['lock_proposal'],proposal_score=r['model_score'],verifier=v['candidate_gloss'],verifier_score=v['model_score'],agreement=same,actual_commit=r['committed_gloss'],
   verifier_accepted=v['accepted'],verifier_reasons=v.get('rejection_reasons',[]),commit_score=v['commit_score'],
   lowering_commit_only_to_025_passes=bool(v['accepted'] and v['commit_score']>=.25),
   bypass_verifier_keep_045_passes=bool(r['accepted'] and r['model_score']>=.45)))
 assert stats['probes']==66 and stats['accepted']==40 and stats['stable_proposals']==14 and stats['commits']==10
 assert len(finishes)==4 and all(e['selected']==e['stage1'] for e in finishes)
 (OUT/'session_analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
 print({k:stats[k] for k in ['probes','accepted','stable_proposals','commits','hand_frame_fraction']})
 for r in stats['proposal_counterfactuals']:
  if not r['agreement']:print(r)

if __name__=='__main__':main()
