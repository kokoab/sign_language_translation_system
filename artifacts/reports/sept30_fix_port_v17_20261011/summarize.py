"""Development comparisons only; does not consume protected evaluation."""
from pathlib import Path
import json,importlib.util,collections,csv
N=Path(__file__).resolve().parent;P=N.parent/'temporal_repair_training_v17_20261011';G=N.parent/'ordered_my_like_v17_20261011/gap175'
s=importlib.util.spec_from_file_location('analysis',P/'analyze_evaluation.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def rows(p):return json.loads(p.read_text())['records']
def delta(p,b):
 old={r['id']:r for r in rows(b)};rr=rows(p)
 return {'n':len(rr),'changed':sum(r['hypothesis']!=old[r['id']]['hypothesis'] for r in rr),'correct_to_wrong':[r['id'] for r in rr if old[r['id']]['hypothesis']==r['reference'] and r['hypothesis']!=r['reference']],'wrong_to_correct':[r['id'] for r in rr if old[r['id']]['hypothesis']!=r['reference'] and r['hypothesis']==r['reference']]}
summary={'note':'Correlated development replays, not new independent accuracy. Old LocalA tune overlaps its historical training lineage; tune scores are diagnostic only. Legacy settings use current engine with old word settings, not an exact September30 app restoration. No model was trained or installed.'}
for arm in ['localA','legacy_base','legacy_guards','rank']:
 result={}
 for kind,fun in [('validation',a.validation),('phase',a.phases),('genuine',a.controls)]:
  p=N/(arm+'_'+kind+'.json')
  if p.exists() and p.stat().st_mtime >= p.with_name(p.stem+'_command.json').stat().st_mtime:
   n=len(rows(p));expected={'validation':570,'phase':72,'genuine':59}[kind]
   if n==expected:
    b=G/({'phase':'phase_on.json','validation':'validation_on.json','genuine':'genuine_on.json'}[kind]);result[kind]={'metrics':fun(p),'vs_deployed':delta(p,b)}
 if result:summary[arm]=result
summary['new_video']={}
cores=[('MY_FRIEND_1',1.02,1.13,'FRIEND'),('MY_FRIEND_2',4.76,4.85,'FRIEND'),('MY_NAME_1',7.32,7.45,'NAME'),('MY_NAME_2',9.61,9.72,'NAME'),('MY_FRIEND_3',14.92,15.02,'FRIEND'),('MY_EAT',16.73,16.83,'EAT'),('MY_FRIEND_4',19.28,19.36,'FRIEND'),('MY_NAME_3',21.20,21.30,'NAME')]
for arm in ['current','localA','legacy_base','legacy_guards','rank']:
 p=N/(arm+'_new_video.json')
 if not p.exists():continue
 result=[]
 for name,lo,hi,following in cores:
  outcomes=collections.Counter();pairs=0
  for r in rows(p):
   ix=[i for i,w in enumerate(r['words']) if min(hi,w['end_seconds'])>max(lo,w['start_seconds'])]
   labels=[r['words'][i]['gloss'] for i in ix];outcomes[','.join(labels) or 'silent']+=1
   pairs+=labels==['MY'] and ix[0]+1<len(r['words']) and r['words'][ix[0]+1]['gloss']==following
  result.append({'event':name,'core':[lo,hi],'outcomes':dict(outcomes),'MY_then_following':pairs,'n':len(rows(p))})
 summary['new_video'][arm]=result
b=N/'legacy_base_validation.json';p=N/'legacy_guards_validation.json'
if b.exists() and p.exists() and len(rows(p))==570:
 summary['guard_port_vs_legacy_base']={'validation':delta(p,b)}
for kind in ['genuine','new_video','have']:
 b=N/('legacy_base_'+kind+'.json');p=N/('legacy_guards_'+kind+'.json')
 if b.exists() and p.exists():summary.setdefault('guard_port_vs_legacy_base',{})[kind]=delta(p,b)
for arm in ['legacy_base','legacy_guards','rank']:
 p=N/(arm+'_have.json')
 if p.exists():summary.setdefault(arm,{})['have']=a.phases(p)['have_take11']
for kind in ['validation','genuine','new_video','have']:
 b=N/('legacy_base_'+kind+'.json');p=N/('rank_'+kind+'.json')
 if b.exists() and p.exists() and len(rows(p))==len(rows(b)):summary.setdefault('rank_vs_legacy_base',{})[kind]=delta(p,b)
(N/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ['new_video','guard_port_vs_legacy_base']},indent=2))
