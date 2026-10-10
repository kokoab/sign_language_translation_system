from pathlib import Path
import json,collections
O=Path(__file__).resolve().parent
def edit(ref,hyp):
 d=list(range(len(hyp)+1))
 for i,a in enumerate(ref,1):
  e=[i]
  for j,b in enumerate(hyp,1):e.append(min(e[-1]+1,d[j]+1,d[j-1]+(a!=b)))
  d=e
 return d[-1]
def validation(path):
 rows=json.load(open(path))['records'];out={}
 for name in sorted({r['dataset'] for r in rows}):
  rr=[r for r in rows if r['dataset']==name]
  out[name]={'clips':len(rr),'exact':sum(r['reference']==r['hypothesis'] for r in rr),
   'errors':sum(edit(r['reference'],r['hypothesis']) for r in rr),'refs':sum(len(r['reference']) for r in rr),
   'silent':sum(not r['hypothesis'] for r in rr)}
  out[name]['wer']=out[name]['errors']/out[name]['refs']
 return out
def replays(path):
 rows=json.load(open(path))['records'];r={}
 sept=[x for x in rows if x['id'].startswith('sept30')]
 r['sept30_exact']=sum(x['hypothesis']==['HELLO','MY','FRIEND','I','FEEL','EXCITED'] for x in sept)
 r['sept30_n']=len(sept)
 for name,lo,hi in [('first_my',2.7,3.15),('second_contact',5.4,7.15),('third_contact',8.9,9.55)]:
  counts=collections.Counter()
  for x in rows:
   if not x['id'].startswith('oct11'):continue
   words=[w['gloss'] for w in x['words'] if lo<=w['start_seconds']<hi and w['gloss'] not in ['NAME']]
   counts[' '.join(words) or 'silent']+=1
  r[name]=dict(counts)
 for x in rows:
  if x['id']=='have_take11':r['have_take11']=dict(collections.Counter(x['hypothesis']));r['have_sequence']=x['hypothesis']
 return r
if __name__=='__main__':
 out={}
 baseline=O.parent/'my_have_phone_v17_20261011/guarded_rollback_validation.json'
 out['validation_installed']=validation(baseline)
 for name in ['control','continuous','continuous_contact','consensus','adapted']:
  p=O/(name+'_replay.json')
  if p.exists():out[name]=replays(p)
 for name in ['validation_control','validation_contact','validation_adapted']:
  p=O/(name+'.json')
  if p.exists():
   rows=json.load(open(p))['records']
   if len(rows)==570:
    out[name]=validation(p)
    b={(r['dataset'],r['id']):r for r in json.load(open(baseline))['records']}
    perclass=[]
    for g in sorted({r['reference'][0] for r in rows if r['dataset']=='citizen_val'}):
     rr=[r for r in rows if r['dataset']=='citizen_val' and r['reference']==[g]]
     before=sum(b[r['dataset'],r['id']]['hypothesis']==[g] for r in rr);after=sum(r['hypothesis']==[g] for r in rr)
     perclass.append({'gloss':g,'n':len(rr),'before':before,'after':after})
    out[name]['class_changes']={'better':sum(r['after']>r['before'] for r in perclass),'worse':sum(r['after']<r['before'] for r in perclass),'equal':sum(r['after']==r['before'] for r in perclass)}
    (O/(name+'_perclass.json')).write_text(json.dumps(perclass,indent=2)+'\n')
 (O/'measured_results.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
