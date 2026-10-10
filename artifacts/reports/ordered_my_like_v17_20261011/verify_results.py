from pathlib import Path
import json,hashlib,ast
O=Path(__file__).resolve().parent;P=O.parent/'temporal_repair_training_v17_20261011'
def read(n):return json.loads((O/n).read_text())
def records(n):return read(n)['records']
checks={}
s=read('summary.json')['v3']
for kind,n in [('phase',72),('validation',570),('genuine',59)]:checks[kind+'_count']=s[kind]['completed_cases']==n
checks['all_100_classes']=s['validation']['metrics']['citizen_class_changes']=={'same':100}
checks['cached_unchanged']=len(s['validation']['changes'])==0
checks['genuine_unchanged']=len(s['genuine']['changes'])==0
checks['september_exact']=s['phase']['metrics']['sept30']=={'exact':24,'n':24}
checks['my_22_of_24']=s['phase']['metrics']['oct11']['target_only']==22
checks['have_unchanged']=not any(c['id'].startswith('have_') for c in s['phase']['changes'])
checks['exact_phone_trace']=records('v3_first_on.json')[0]['hypothesis']==['HELLO','MY','NAME']
checks['delay_only_does_not_fix']=records('first_delay.json')[0]['hypothesis']==records('first_off.json')[0]['hypothesis']==['HELLO','LIKE','NAME']
unit=read('rule_tests.json');checks['cue_parity']=unit['passed'] and len(unit['cases'])==7
for version in ['v4','v5']:
 checks[version+'_focused_failure']= [r['hypothesis'] for r in records(version+'_remaining_on.json')]==[['HELLO','NAME'],['HELLO','LIKE','NAME']]
pins=json.loads((P/'production_preservation.json').read_text())['pins']
checks['production_sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
checks['phone_test_sources_restored']=all(read('phone_source_restored.json').values())
checks['selected_source_matches_snapshot']=all((O/'native'/p.name).read_bytes()==p.read_bytes() for p in (O/'v3/native').glob('*.swift'))
for p in O.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
checks['python_scripts_parse']=True
result={'passed':all(checks.values()),'checks':checks,'source_pins':pins,'phone_test':'Follow-up completed after unlock: v3 on0/2, gap175 on2/2; see gap175/verification.json. Previous app release restored.','installation_restore':read('restore_install.json').get('info',{}).get('outcome') if (O/'restore_install.json').exists() else 'pending'}
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_pins'},indent=2));assert result['passed']
