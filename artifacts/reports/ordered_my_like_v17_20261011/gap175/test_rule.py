from pathlib import Path
import json,subprocess,sys
G=Path(__file__).resolve().parent;O=G.parent;sys.path.insert(0,str(G));from ordered_rule import assess,selftest
selftest();cases=json.loads((O/'rule_fixtures.json').read_text())
for tag in ['on_1','on_2']:
 d=json.loads((O/f'phone_{tag}.json').read_text());e=next(e for e in d['motion_events'] if e.get('start',100)<3.2 and e.get('end',0)>2.7 and 'rows' in e)
 cases.append(dict(id='real_phone_'+tag,rows=e['rows'],start=e['start'],end=e['end'],available=e['available'],next_start=e['next_start'],expected='MY'))
for c in cases:assert assess(c['rows'],c['start'],c['end'],c['available'],next_start=c.get('next_start'))['decision']==c['expected'],c['id']
(G/'rule_fixtures.json').write_text(json.dumps(cases))
files=[str(G/'native'/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift']]
with (G/'unit_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(G/'unit_probe')]+files+[str(O/'unit/main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([str(G/'unit_probe'),str(G/'rule_fixtures.json'),str(G/'rule_tests.json')],check=True)
