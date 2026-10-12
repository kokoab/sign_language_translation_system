import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2]
p=O/'results.json';r=json.loads(p.read_text());assert r['state']=='complete'
role=R/'artifacts/reports/temporal_repair_training_v17_20261011/isolated_role_check.json'
roles=json.loads(role.read_text());r['exclusion_manifest_sha256']=hashlib.sha256(role.read_bytes()).hexdigest()
archive=O/'all_cached_predictions.json'
assert not archive.exists();archive.write_text(p.read_text())
for name,d in r['domains'].items():
 excluded=set(roles[name].get('excluded_validation_ids',[]));keep=[i for i,id in enumerate(d['ids']) if id not in excluded]
 d['excluded_count']=d['n']-len(keep);d['n']=len(keep)
 d['ids']=[d['ids'][i] for i in keep];d['targets']=[d['targets'][i] for i in keep]
 d['predictions']={k:[v[i] for i in keep] for k,v in d['predictions'].items()}
 d['correct']={k:sum(a==b for a,b in zip(v,d['targets'])) for k,v in d['predictions'].items()}
for k,v in r['summary'].items():
 v['n']=sum(d['n'] for d in r['domains'].values());v['correct']=sum(d['correct'][k] for d in r['domains'].values());v['accuracy_percent']=100*v['correct']/v['n']
r['scope']+=' Existing isolated_role_check exclusions applied equally after inference; unfiltered cache predictions preserved separately.'
p.write_text(json.dumps(r)+'\n')
