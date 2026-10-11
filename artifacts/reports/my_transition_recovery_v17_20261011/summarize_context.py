from pathlib import Path
import json,collections
D=Path(__file__).resolve().parent
cores=[('MY_FRIEND_1',1.02,1.13,'FRIEND'),('MY_FRIEND_2',4.76,4.85,'FRIEND'),('MY_NAME_1',7.32,7.45,'NAME'),('MY_NAME_2',9.61,9.72,'NAME'),('MY_FRIEND_3',14.92,15.02,'FRIEND'),('MY_EAT',16.73,16.83,'EAT'),('MY_FRIEND_4',19.28,19.36,'FRIEND'),('MY_NAME_3',21.20,21.30,'NAME')]
result={}
for name in ['current','localA']:
 p=D/(name+'_context.json')
 if not p.exists():continue
 rows=json.loads(p.read_text())['records'];result[name]={}
 for arm in sorted({r['id'].split('__')[0] for r in rows}):
  rr=[r for r in rows if r['id'].startswith(arm+'__')];events=[]
  for tag,lo,hi,following in cores:
   c=collections.Counter();pair=0
   for r in rr:
    ix=[i for i,w in enumerate(r['words']) if min(hi,w['end_seconds'])>max(lo,w['start_seconds'])]
    labels=[r['words'][i]['gloss'] for i in ix];c[','.join(labels) or 'silent']+=1
    pair+=labels==['MY'] and ix[0]+1<len(r['words']) and r['words'][ix[0]+1]['gloss']==following
   events.append(dict(event=tag,outcomes=dict(c),pair=pair,n=len(rr)))
  result[name][arm]={'pair':sum(e['pair'] for e in events),'n':8*len(rr),'events':events}
(D/'context_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{a:{x:y for x,y in v.items() if x!='events'} for a,v in r.items()} for k,r in result.items()},indent=2))
