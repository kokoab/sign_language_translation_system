from pathlib import Path
import json,collections,sys
D=Path(__file__).resolve().parent
cores=[(1.02,1.13,'FRIEND'),(4.76,4.85,'FRIEND'),(7.32,7.45,'NAME'),(9.61,9.72,'NAME'),(14.92,15.02,'FRIEND'),(16.73,16.83,'EAT'),(19.28,19.36,'FRIEND'),(21.2,21.3,'NAME')]
def pairs(rr):
 events=[]
 for lo,hi,fol in cores:
  good=0;outputs=collections.Counter()
  for x in rr:
   ix=[i for i,w in enumerate(x['words']) if min(hi,w['end_seconds'])>max(lo,w['start_seconds'])]
   labels=[x['words'][i]['gloss'] for i in ix];outputs[','.join(labels) or 'silent']+=1
   good+=labels==['MY'] and ix[0]+1<len(x['words']) and x['words'][ix[0]+1]['gloss']==fol
  events.append(dict(core=[lo,hi],pair=good,outcomes=dict(outputs),n=len(rr)))
 return dict(pair=sum(e['pair'] for e in events),n=8*len(rr),events=events)
def summarize(p):
 rows=json.loads(p.read_text())['records'];return {arm:pairs([x for x in rows if x['id'].split('__')[0]==arm]) for arm in sorted({x['id'].split('__')[0] for x in rows})}
if __name__=='__main__':
 for name in sys.argv[1:]:
  p=D/(name+'.json');r=summarize(p);p.with_name(p.stem+'_metrics.json').write_text(json.dumps(r,indent=2));print(name,{a:{k:v for k,v in r.items() if k!='events'} for a,r in r.items()})
