from pathlib import Path
import json,sys,collections
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]))
from scripts.evaluate_temporal_boundary_v17 import edit_counts
result={}
for path in P.glob('native_validation_*.json'):
 if '.current.' in path.name:continue
 d=json.loads(path.read_text());rows=d['records'];sets={}
 for dataset in sorted({r['dataset'] for r in rows}):
  rr=[r for r in rows if r['dataset']==dataset]
  if dataset=='tune':
   cc=[edit_counts(r['reference'],r['hypothesis']) for r in rr]
   v={k:sum(x[k] for x in cc) for k in ['correct','substitutions','deletions','insertions','references']};v['wer']=(v['substitutions']+v['deletions']+v['insertions'])/max(1,v['references']);v['n']=len(rr)
  else:
   v=dict(n=len(rr),exact=sum(r['reference']==r['hypothesis'] for r in rr),correct=sum(r['reference'][0] in r['hypothesis'] for r in rr),false_have=sum('HAVE' in r['hypothesis'] and r['reference']!=['HAVE'] for r in rr),false_my_on_like=sum('MY' in r['hypothesis'] and r['reference']==['LIKE'] for r in rr))
  sets[dataset]=v
 result[path.stem]={'complete':len(rows)==570,'sets':sets}
 if len(rows)==570:print(path.stem,sets)
for variant in ['hybrid','hybrid_v2']:
 f=P/f'{variant}_my_like_val.json'
 if not f.exists():continue
 a=json.loads((P/'ordered_ends_my_like_val.json').read_text())['records'];b=json.loads(f.read_text())['records']
 result[variant+'_59videos']=dict(n=len(b),changes=[dict(id=x['id'],reference=x['reference'],before=x['hypothesis'],after=y['hypothesis']) for x,y in zip(a,b) if x['hypothesis']!=y['hypothesis']])
 print(variant+'_59videos',result[variant+'_59videos'])
(P/'validation_summary.json').write_text(json.dumps(result,indent=2))
