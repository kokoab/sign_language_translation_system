from pathlib import Path
import json,statistics,hashlib
D=Path(__file__).resolve().parent
reference='MY FRIEND MY FRIEND MY NAME MY NAME MY FRIEND MY EAT MY FRIEND MY NAME'.split()
def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a):
  nxt=[i+1]
  for j,y in enumerate(b):nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(x!=y)))
  row=nxt
 return row[-1]
def stats(values):
 x=sorted(values)
 return dict(n=len(x),mean=statistics.mean(x),median=statistics.median(x),p95=x[int(.95*(len(x)-1))],max=max(x))
passes=[]
for index,arm in enumerate(['baseline','candidate','candidate','baseline']):
 p=D/f'my_transition_{arm}_{index}.json';r=json.loads(p.read_text());assert r['arm']==arm and r['pass']==index
 frames=r['processed_seconds'];compute=r['compute_seconds'];assert len(frames)==len(compute)>100 and all(b>a for a,b in zip(frames,frames[1:]))
 passes.append(dict(arm=arm,pass_index=index,hypothesis=r['hypothesis'],full_sequence_exact=r['hypothesis']==reference,token_errors=distance(r['hypothesis'],reference),MY_count=r['hypothesis'].count('MY'),frames=len(frames),last_processed_seconds=frames[-1],compute_ms=stats([x*1000 for x in compute]),over50ms=sum(x>.05 for x in compute),ready_mib=r['ready_mb'],peak_sampled_mib=r['peak_sampled_mb'],thermal_before=r['thermal_before'],thermal_after=r['thermal_after'],wall_seconds=r['wall_seconds'],stage3_ready=r['stage3_ready'],result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
summary={}
for arm in ['baseline','candidate']:
 rr=[r for r in passes if r['arm']==arm];raw=[json.loads((D/f'my_transition_{arm}_{r["pass_index"]}.json').read_text()) for r in rr]
 summary[arm]=dict(passes=len(rr),full_sequences_exact=sum(r['full_sequence_exact'] for r in rr),MY_count=sum(r['MY_count'] for r in rr),token_errors=sum(r['token_errors'] for r in rr),reference_tokens=16*len(rr),pooled_compute_ms=stats([v*1000 for r in raw for v in r['compute_seconds']]),ready_mib_range=[min(r['ready_mib'] for r in rr),max(r['ready_mib'] for r in rr)],peak_sampled_mib=max(r['peak_sampled_mib'] for r in rr),all_thermal_nominal=all(r['thermal_before']==r['thermal_after']==0 for r in rr))
out=dict(note='Physical iPhone13 ABBA camera-paced replay through normal app engine with resident English and overlay Vision. Known development video, not fresh natural signing. Frame schedules can differ because latest-frame load handling is part of the measured pipeline. Compute times exclude model loading and are not end-to-end publication latency. Four short passes do not establish long-duration thermals.',reference=reference,passes=passes,summary=summary,xctest_returncode=json.loads((D/'phone_test_status.json').read_text())['returncode'])
(D/'phone_comparison_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
