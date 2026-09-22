"""Fixed-policy replay of saved interval predictions. No inference/training/deployment."""
import sys,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.evaluate_temporal_boundary_v17 import arguments,edit_counts,summarize
from scripts.live_reel_stage1_v17 import VerifiedCommitLock
from active.v17.approved_phrase_data_v17 import digest
from scripts.train_temporal_boundary_v17 import atomic
OUT=Path(__file__).parent
SOURCE=ROOT/'artifacts/reports/boundary_local_calibration_v17_20260922'
SOFT={'low_score','low_margin'}

def current(p,a):
 q,v=p.get('proposal'),p.get('verifier')
 if not q or not v:return None
 agreement=q['model_score'] if q.get('candidate_gloss')==v.get('candidate_gloss') else 0.
 passed=bool(q.get('accepted') and v.get('accepted')) and VerifiedCommitLock(a.commit_hits,a.instant_commit_score).update(str(v.get('candidate_gloss')),max(v['model_score'],agreement),proposal=str(q.get('candidate_gloss')),proposal_score=q['model_score'],minimum_score=a.commit_score)
 return v.get('candidate_gloss') if passed else None

def rescue(p,a):
 old=current(p,a)
 if old:return old
 q,v=p.get('proposal'),p.get('verifier')
 if not q or not v or p.get('reason'):return None
 if any(set(x.get('rejection_reasons',[]))-SOFT for x in (q,v)):return None
 top=v.get('top3',[])
 if len(top)<2 or top[0]['gloss']!=v.get('candidate_gloss'):return None
 if v['model_score']<max(a.commit_score,a.instant_commit_score):return None
 if top[0]['model_score']-top[1]['model_score']<a.minimum_margin:return None
 return v['candidate_gloss']

def replay(records,a):
 result=[];blocks=Counter();rescues=[]
 for r in records:
  old=[current(p,a) for p in r['predictions']]
  assert old==[p.get('committed_gloss') for p in r['predictions']],r['id']
  assert [g for g in old if g]==r['hypothesis']
  assert json.loads(json.dumps(edit_counts(r['reference'],r['hypothesis'])))==r['metrics']
  new=[rescue(p,a) for p in r['predictions']]
  for index,(p,b,n) in enumerate(zip(r['predictions'],old,new)):
   if b:assert n==b
   else:
    q,v=p.get('proposal',{}),p.get('verifier',{})
    if not q or not v:blocks['missing_classification']+=1
    elif not q.get('accepted'):blocks['proposal_rejected_first']+=1
    elif not v.get('accepted'):blocks['verifier_rejected_after_proposal']+=1
    else:blocks['commit_score_after_both_accepted']+=1
    if n:rescues.append(dict(id=r['id'],interval=index,label=n,verifier_score=v['model_score'],proposal_reasons=q.get('rejection_reasons'),verifier_reasons=v.get('rejection_reasons')))
  hyp=[g for g in new if g]
  result.append(dict(id=r['id'],reference=r['reference'],hypothesis=hyp,metrics=edit_counts(r['reference'],hyp)))
 return dict(baseline=summarize(records,records),candidate=summarize(result,records),blocking_stages=dict(blocks),rescued_intervals=rescues,records=result)

def passes(d):
 b,s=d['baseline'],d['candidate']
 return s['wer']<b['wer'] and s['correct']>=b['correct'] and s['insertions']<=b['insertions'] and s['retained_baseline_correct_events']>=b['correct']

def main():
 a=arguments('unused',OUT/'unused');assert a.commit_hits==1
 # Runnable checks preserve existing emissions and all non-confidence rejection reasons.
 p=dict(proposal=dict(accepted=False,candidate_gloss='B',model_score=.1,rejection_reasons=['low_score']),verifier=dict(accepted=False,candidate_gloss='A',model_score=.9,rejection_reasons=['low_score'],top3=[dict(gloss='A',model_score=.9),dict(gloss='B',model_score=.05)]))
 assert rescue(p,a)=='A'
 p['verifier']['rejection_reasons']=['clip_too_long'];assert rescue(p,a) is None
 p['verifier']['rejection_reasons']=[];p['verifier']['model_score']=.7;assert rescue(p,a) is None
 files=[SOURCE/'calibration.json',SOURCE/'confirmation.json',SOURCE/'manifest.json',Path(__file__),ROOT/'scripts/live_boundary_v17.py',ROOT/'scripts/live_reel_stage1_v17.py']
 hashes={str(f.relative_to(ROOT)):digest(f) for f in files}
 if (OUT/'policy.json').exists():raise FileExistsError('preserve prior comparison')
 atomic(OUT/'policy.json',dict(hashes=hashes,policy='Keep every original commit. Rescue rejected intervals only when final verifier score>=existing instant threshold(.8), final top-two margin>=existing minimum(.08), and both models have no non-confidence rejection. No new sweep.',thresholds={k:getattr(a,k) for k in ('commit_hits','commit_score','instant_commit_score','minimum_score','minimum_margin')},gate='strictly lowerWER, no fewercorrects/no extra insertions, all baselinecorrect referencepositions retained',confirmation='Only evaluate fixed candidate if calibration passes',scope='cached interval decisions, not live stateful scheduling'))
 records=json.loads((SOURCE/'calibration.json').read_text())['records']['frozen_min3']
 result=replay(records,a);result['passed']=passes(result);atomic(OUT/'calibration.json',result)
 # Diagnostic only: retain verifier guesses before acceptance, not an admissible policy.
 unfiltered=[]
 for r in records:
  hyp=[p['verifier']['candidate_gloss'] for p in r['predictions'] if p.get('verifier',{}).get('candidate_gloss') not in (None,'UNKNOWN')]
  unfiltered.append(dict(id=r['id'],reference=r['reference'],hypothesis=hyp,metrics=edit_counts(r['reference'],hyp)))
 diagnostic=summarize(unfiltered,records);atomic(OUT/'unfiltered_verifier_diagnostic.json',dict(summary=diagnostic,scope='Not a candidate or an oracle. All verifier guesses, including wrong/unsupported intervals.'))
 confirmation=None
 if result['passed']:
  records=json.loads((SOURCE/'confirmation.json').read_text())['records']['frozen_min3']
  confirmation=replay(records,a);confirmation['passed']=passes(confirmation);atomic(OUT/'confirmation.json',confirmation)
 for path,sha in hashes.items():assert digest(ROOT/path)==sha
 rows=[('Current policy',result['baseline']),('Verifier-aware rescue',result['candidate']),('Unfiltered verifier (diagnostic only)',diagnostic)]
 lines=['# Reel rejection replay','','No model inference, retraining, threshold sweep or live changes. Existing59clip/150sign calibration; one policy declared before scoring.','',
 '| Policy | WER | Correct /150 | Substitutions | Deletions | Insertions | Retained /82 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,s in rows:lines.append(f"| {name} | {s['wer']:.2%} | {s['correct']} | {s['substitutions']} | {s['deletions']} | {s['insertions']} | {s['retained_baseline_correct_events']} |")
 lines+=['',f"Rescued intervals: {len(result['rescued_intervals'])}. Calibration gate passed: {result['passed']}.",f"First blocking stage counts: {result['blocking_stages']}.",
  'Rescue uses final verifier score>=0.8 and final margin>=0.08, bypassing only low_score/low_margin rejection. Original commits and physical/duration/learned-no-emit rejection safeguards remain. A high score is not a calibrated probability of correctness.',
  'Original decisions and all baseline edit counts reproduce exactly; functional policy guards and source/code hashes pass.',
  'Local phrase intervals lack per-sign truth. Recovered correct signs and errors are transcript-aligned; individual rescued intervals cannot all be assigned ground-truth labels. Blocking counts are not counts of recoverable correct signs.',
  'Unfiltered verifier is not an upper bound or deployable policy. It exposes the error trade-off of removing acceptance wholesale.',
  'One fresh commit lock per interval with commit_hits=1; instant threshold does not affect original decisions after score acceptance. This replay does not establish asynchronous live stability/repeat behavior.',
  'Familiar-signers/sixphrase reused development; no unseen-signer or continuous transition-negative guarantee.']
 if confirmation:
  lines+=['',f"Confirmation candidate WER {confirmation['candidate']['wer']:.2%}, baseline {confirmation['baseline']['wer']:.2%}; pass={confirmation['passed']}. No deployment."]
 else:lines+=['','Candidate failed calibration; confirmation predictions were not evaluated. No live change recommended from this replay.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(calibration={k:v for k,v in result.items() if k not in ('records','rescued_intervals')},rescues=len(result['rescued_intervals']),unfiltered=diagnostic,confirmation=confirmation and {k:v for k,v in confirmation.items() if k not in ('records','rescued_intervals')})))

if __name__=='__main__':main()
