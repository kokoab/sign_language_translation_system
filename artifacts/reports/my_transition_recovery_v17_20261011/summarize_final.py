"""Aggregate completed experiments without treating repeated schedules as independent data."""
from pathlib import Path
import json,collections,hashlib,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D));from metrics import pairs
sys.path.insert(0,str(D.parent/'temporal_repair_training_v17_20261011'));from analyze_evaluation import phases,controls
G=D.parent/'ordered_my_like_v17_20261011/gap175'
ref='MY FRIEND MY FRIEND MY NAME MY NAME MY FRIEND MY EAT MY FRIEND MY NAME'.split()
def read(p):return json.loads(p.read_text())['records']
def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a):
  nxt=[i+1]
  for j,y in enumerate(b):nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(x!=y)))
  row=nxt
 return row[-1]
def compare(rows,base):
 base={r['id']:r for r in base};out={}
 for dataset in sorted({r.get('dataset','all') for r in rows}):
  rr=[r for r in rows if r.get('dataset','all')==dataset]
  out[dataset]=dict(n=len(rr),baseline_exact=sum(base[r['id']]['hypothesis']==r['reference'] for r in rr),candidate_exact=sum(r['hypothesis']==r['reference'] for r in rr),changed=sum(base[r['id']]['hypothesis']!=r['hypothesis'] for r in rr),formerly_correct_lost=sum(base[r['id']]['hypothesis']==r['reference'] and r['hypothesis']!=r['reference'] for r in rr),edit_errors_before=sum(distance(base[r['id']]['hypothesis'],r['reference']) for r in rr),edit_errors_after=sum(distance(r['hypothesis'],r['reference']) for r in rr))
 out['changes']=[dict(id=r['id'],reference=r['reference'],before=base[r['id']]['hypothesis'],after=r['hypothesis']) for r in rows if base[r['id']]['hypothesis']!=r['hypothesis']]
 return out
def schedule(rows):return dict(n=len(rows),strict_core_pairs=pairs(rows),full_sequences_exact=sum(r['hypothesis']==ref for r in rows),reference=ref,edit_errors=sum(distance(r['hypothesis'],ref) for r in rows),reference_tokens=len(ref)*len(rows),failed_sequences=[dict(id=r['id'],hypothesis=r['hypothesis']) for r in rows if r['hypothesis']!=ref])
if __name__=='__main__':
 prefix=sys.argv[1] if len(sys.argv)>1 else 'context_final'
 base=read(D.parent/'sept30_fix_port_v17_20261011/current_new_video.json')+read(D/'current_offsets.json')
 result=dict(note='Development only: 24 correlated schedules of one known video. Three physical pairs train; five other physical pairs develop. Protected Citizen test untouched. Cached and synthetic controls are regression checks, not independent generalization.',baseline_schedule=schedule(base),candidate_schedule=schedule(read(D/(prefix+'_all.json'))))
 for kind,bp,n in [('validation',G/'validation_on.json',570),('genuine',G/'genuine_on.json',59),('splice',D/'splice_current.json',66),('phase',G/'phase_on.json',72)]:
  p=D/(prefix+'_'+kind+'.json')
  if p.exists():
   rr=read(p);assert len(rr)==n,(kind,len(rr),n)
   result[kind]=compare(rr,read(bp))
   if kind=='phase':result['historical_reviewed']=dict(baseline=phases(bp),candidate=phases(p))
   if kind=='genuine':result['genuine_reviewed']=dict(baseline=controls(bp),candidate=controls(p))
 training={}
 for name in ['actual','balanced','restored','paired','jitter','confidence']:
  x=json.loads((D/(name+'_training_results.json')).read_text());arms=sorted({r['arm'] for r in x['history'] if r['arm']!='baseline'});training[name]=dict(arms=len(arms),selected=x['selected'],encoders_unchanged=x['encoders_unchanged'])
 result['training']=training;result['total_training_arms']=sum(x['arms'] for x in training.values())
 result['original_output_parity']=json.loads((D/'confidence_fp16_parity.json').read_text())
 (D/(prefix+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k in ['total_training_arms','original_output_parity']}))
 print('schedule',result['baseline_schedule']['strict_core_pairs']['pair'],result['candidate_schedule']['strict_core_pairs']['pair'],'full',result['candidate_schedule']['full_sequences_exact'])
 for k in ['validation','genuine','splice']:print(k,{a:b for a,b in result.get(k,{}).items() if a!='changes'})
