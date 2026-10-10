from pathlib import Path
import json,importlib.util,collections
O=Path(__file__).resolve().parent;P=O.parent/'temporal_repair_training_v17_20261011'
s=importlib.util.spec_from_file_location('previous',P/'analyze_evaluation.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def records(p):return json.loads(p.read_text())['records']
def changes(p,b):
 base={r['id']:r for r in records(b)}
 return [{'id':r['id'],'before':base[r['id']]['hypothesis'],'after':r['hypothesis']} for r in records(p) if r['hypothesis']!=base[r['id']]['hypothesis']]
out={'model':'SpanRecognizerV17FragmentRejectLettersB8FP16','training':False,'scope':'Development/replay retention; correlated schedules are not independent phone accuracy.'}
for version in ['v3','v4']:
 d={}
 for kind,fn,base in [('phase',a.phases,'incumbent_control_replay.json'),('validation',a.validation,'incumbent_validation.json'),('genuine',a.controls,'incumbent_genuine_controls.json')]:
  p=O/f'{version}_{kind}_on.json'
  if not p.exists():continue
  rows=records(p);d[kind]={'completed_cases':len(rows),'metrics':fn(p),'changes':changes(p,P/base)}
 if d:out[version]=d
(O/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for version,d in out.items():
 if version not in ['v3','v4']:continue
 print(version,{k:{'n':v['completed_cases'],'changed':len(v['changes'])} for k,v in d.items()})
 if 'phase' in d:print('MY',d['phase']['metrics']['oct11']['target_only'],'/24')
