from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent
results={}
for variant in ['baseline','candidate']:
 path=p/(variant+'_device.json')
 if not path.exists(): continue
 d=json.loads(path.read_text());out={'metadata':d['metadata'],'records':[]}
 for row in d['records']:
  out['records'].append({k:v for k,v in row.items() if k not in ['words','publications','timing']})
  out['records'][-1]['timing']={k:v for k,v in row['timing'].items() if k!='samples_ms'}
  if 'spell_events' in row:out['records'][-1]['spell_events']=row['spell_events']
 out['aggregate']={}
 for kind in ['trace','video']:
  rows=[r for r in d['records'] if r['kind']==kind];xs=sorted(x for r in rows for x in r['timing']['samples_ms'])
  if xs:out['aggregate'][kind]={'frames':len(xs),'median_ms':statistics.median(xs),'p90_ms':xs[int(.9*len(xs))],'p99_ms':xs[int(.99*len(xs))],'max_ms':max(xs),'mean_ms':statistics.mean(xs),'over50ms':sum(x>50 for x in xs),'peak_sampled_mb':max(r['peak_sampled_mb'] for r in rows)}
 results[variant]=out
if len(results)==2:
 a={(r['kind'],r['id'],r['pass']):r for r in results['baseline']['records']};b={(r['kind'],r['id'],r['pass']):r for r in results['candidate']['records']}
 results['hypothesis_differences']=[{'case':list(k),'baseline':a[k]['hypothesis'],'candidate':b[k]['hypothesis']} for k in a.keys() & b.keys() if a[k]['hypothesis']!=b[k]['hypothesis']]
(p/'summary.json').write_text(json.dumps(results,indent=2))
for name,data in results.items():
 if isinstance(data,dict):print(name,json.dumps({'metadata':data['metadata'],'aggregate':data['aggregate']}))
 else: print(name,json.dumps(data))
