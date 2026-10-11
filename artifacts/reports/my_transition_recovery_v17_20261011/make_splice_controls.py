"""Stress fixtures, explicitly NOT naturally coarticulated phrases or training data."""
from pathlib import Path
import json,hashlib,copy
D=Path(__file__).resolve().parent;J=D.parent/'my_have_joint_experiment_v17_20261011';G=D.parent/'ordered_my_like_v17_20261011/gap175'
items=json.loads((J/'validation_traces.json').read_text());baseline={r['id']:r for r in json.loads((G/'validation_on.json').read_text())['records']}
first=[r for r in items if r['reference']==['LIKE']];second=[r for r in items if r['reference'] in [['FRIEND'],['NAME'],['EAT']]]
root=D/'splice_traces';root.mkdir(exist_ok=True);cases=[];metadata=[]
def clip(item):
 frames=json.loads(Path(item['path']).read_text())['frames'];frames=[r for r in frames if 's' in r and ('e' in r or 'ef32' in r) and not r.get('held')]
 word=[w for w in baseline[item['id']]['words'] if w['gloss']==item['reference'][0]]
 if word:
  lo=word[0]['start_seconds']-.1;hi=word[-1]['end_seconds']+.1
  frames=[r for r in frames if lo<=r['s']<=hi]
 return frames
for a in first:
 for b in second:
  aa=clip(a);bb=clip(b)
  for gap in [.05,.2]:
   frames=copy.deepcopy(aa);offset=frames[-1]['s']+gap-bb[0]['s'];tail=copy.deepcopy(bb)
   for r in tail:r['s']+=offset
   frames+=tail;name=f'{a["id"]}__{b["id"]}__gap{gap}';p=root/(name+'.json');p.write_text(json.dumps({'frames':frames}))
   cases.append(dict(id=name,dataset='synthetic_like_successor',reference=['LIKE',b['reference'][0]],path=str(p)))
   metadata.append(dict(id=name,sources=[a['path'],b['path']],first_ends=aa[-1]['s'],second_starts=tail[0]['s'],gap=gap,source_sha256=[hashlib.sha256(Path(x['path']).read_bytes()).hexdigest() for x in [a,b]]))
(D/'splice_cases.json').write_text(json.dumps(cases,indent=2));(D/'splice_boost_cases.json').write_text(json.dumps([dict(r,policy='old',kind='pair',eligibility='top3',alt_pre=.2,floor=.9) for r in cases],indent=2));(D/'splice_contract.json').write_text(json.dumps({'status':'Development stress only; separate signers and hard cuts, not a naturally signed phrase. No training. Never claim these as new independent accuracy.','cases':metadata},indent=2));print(len(cases),'synthetic LIKE→next-sign stress fixtures')
