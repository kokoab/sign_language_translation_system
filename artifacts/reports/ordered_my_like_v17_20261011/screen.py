from pathlib import Path
import json,sys,collections
from probe_motion import measure
from ordered_rule import assess,DEFAULT
O=Path(__file__).resolve().parent;P=O.parent/'temporal_repair_training_v17_20261011'
cases={r['id']:r for r in json.loads((O.parent/'my_have_joint_experiment_v17_20261011/validation_traces.json').read_text())};baseline=json.loads((P/'incumbent_validation.json').read_text())['records'];out=[]
for record in baseline:
 if not any(w['gloss'] in ['MY','LIKE'] for w in record['words']):continue
 rows=json.loads(Path(cases[record['id']]['path']).read_text())['frames'];features=measure(rows)
 for w in record['words']:
  if w['gloss'] not in ['MY','LIKE']:continue
  nexts=[x['start_seconds'] for x in record['words'] if x['start_seconds']>w['end_seconds'] and x['commit_seconds']<=w['commit_seconds']]
  # This passive screen may only use subsequent boundaries known at publication.
  result=assess(features,w['start_seconds'],w['end_seconds'],w['commit_seconds'],next_start=min(nexts) if nexts else None)
  out.append(dict(id=record['id'],dataset=record['dataset'],reference=record['reference'],word=w,**result))
(O/'passive_screen.json').write_text(json.dumps({'policy':DEFAULT,'rows':out,'note':'Original committed spans and available timestamps; candidate with delayed LIKE must be replayed independently.'},indent=2)+'\n')
print('scored',len(out));print('LIKE decisions',dict(collections.Counter(r['decision'] for r in out if r['word']['gloss']=='LIKE')));print('rescues',[{k:r[k] for k in ['id','reference','decision']} for r in out if r['word']['gloss']=='LIKE' and r['decision']=='MY'])
