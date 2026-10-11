"""Hard-cut MY + successor + WRITE preservation fixtures. Development only; no training."""
from pathlib import Path
import json,copy,hashlib,itertools
D=Path(__file__).resolve().parent;J=D.parent/'my_have_joint_experiment_v17_20261011';G=D.parent/'ordered_my_like_v17_20261011/gap175'
items=json.loads((J/'validation_traces.json').read_text());baseline={r['id']:r for r in json.loads((G/'validation_on.json').read_text())['records']}
def choose(label,n):return [r for r in items if r['reference']==[label] and baseline[r['id']]['hypothesis']==[label]][:n]
def clip(item):
 frames=json.loads(Path(item['path']).read_text())['frames'];frames=[r for r in frames if 's' in r and ('e' in r or 'ef32' in r) and not r.get('held')]
 words=baseline[item['id']]['words'];lo=words[0]['start_seconds']-.1;hi=words[-1]['end_seconds']+.1
 return [r for r in frames if lo<=r['s']<=hi]
root=D/'triple_traces';root.mkdir(exist_ok=True);cases=[];metadata=[]
params=json.loads((D/'joint_all_cases.json').read_text())[0];params={k:v for k,v in params.items() if k not in ['id','path','phase','capture_offset','reference']}
for a,b,c,gap in itertools.product(choose('MY',2),sum([choose(x,1) for x in ['FRIEND','NAME','EAT']],[]),choose('WRITE',2),[.05,.2]):
 frames=[]
 for item in [a,b,c]:
  tail=copy.deepcopy(clip(item));offset=frames[-1]['s']+gap-tail[0]['s'] if frames else -tail[0]['s']
  for r in tail:r['s']+=offset
  frames+=tail
 name=f'{a["id"]}__{b["id"]}__{c["id"]}__gap{gap}';p=root/(name+'.json');p.write_text(json.dumps({'frames':frames}))
 cases.append(dict(params,id=name,dataset='synthetic_my_successor_write',reference=['MY',b['reference'][0],'WRITE'],path=str(p)))
 metadata.append(dict(id=name,sources=[x['path'] for x in [a,b,c]],gap=gap,source_sha256=[hashlib.sha256(Path(x['path']).read_bytes()).hexdigest() for x in [a,b,c]]))
(D/'triple_cases.json').write_text(json.dumps(cases,indent=2));(D/'triple_contract.json').write_text(json.dumps(dict(note='Separate isolated clips/signers with hard cuts. Negative control for accidental deletion of a real WRITE; not natural coarticulation, new independent accuracy or training.',cases=metadata),indent=2));print(len(cases))
