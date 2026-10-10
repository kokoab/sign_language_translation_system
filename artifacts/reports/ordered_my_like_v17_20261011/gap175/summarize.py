from pathlib import Path
import json,importlib.util,hashlib,ast
G=Path(__file__).resolve().parent;O=G.parent;P=O.parent/'temporal_repair_training_v17_20261011'
s=importlib.util.spec_from_file_location('a',P/'analyze_evaluation.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def read(p):return json.loads(p.read_text())
def records(p):return read(p)['records']
result={'model':'SpanRecognizerV17FragmentRejectLettersB8FP16','only_policy_change_from_v3':'maximum reliable-landmark gap 150ms to 175ms','new_training':False,'phone':read(G/'phone_summary.json'),'original_phone_v3':read(O/'phone_summary.json')['first_my_correct_by_arm']}
for k,func,base in [('phase',a.phases,'incumbent_control_replay.json'),('validation',a.validation,'incumbent_validation.json'),('genuine',a.controls,'incumbent_genuine_controls.json')]:
 p=G/f'{k}_on.json';rows=records(p);b={r['id']:r for r in records(P/base)};result[k]={'n':len(rows),'metrics':func(p),'changed':[{'id':r['id'],'before':b[r['id']]['hypothesis'],'after':r['hypothesis']} for r in rows if r['hypothesis']!=b[r['id']]['hypothesis']]}
(G/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
checks={'counts':all(result[k]['n']==n for k,n in [('phase',72),('validation',570),('genuine',59)]),'all_100_classes':result['validation']['metrics']['citizen_class_changes']=={'same':100},'cached_unchanged':not result['validation']['changed'],'genuine_unchanged':not result['genuine']['changed'],'september_24':result['phase']['metrics']['sept30']=={'exact':24,'n':24},'have_outputs_unchanged':not any(r['id'].startswith('have_') for r in result['phase']['changed']),'phone_MY_both_on':result['phone']['first_my_correct_by_arm']=={'off':0,'on':2},'phone_following_NAME':all(r['first_my_then_name'] for r in result['phone']['records'] if r['arm']=='on'),'phone_normal_thermal':all(r['thermal_before']==r['thermal_after']==0 for r in result['phone']['records']),'cue_parity_9':read(G/'rule_tests.json')['passed'] and len(read(G/'rule_tests.json')['cases'])==9,'test_executed':"testOrderedMyLikeGap175SharedApp()' passed" in (G/'phone_test.log').read_text(),'restored_sources':all(read(G/'phone_source_restored.json').values()),'previous_release_reinstalled':read(G/'restore_install.json')['info']['outcome']=='success'}
pins=read(P/'production_preservation.json')['pins'];checks['production_source_pins']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
for p in G.glob('*.py'):ast.parse(p.read_text())
checks['scripts_parse']=True
(G/'verification.json').write_text(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2)+'\n')
print('verification',checks);print('MY phase',result['phase']['metrics']['oct11']['target_only'],'/24');assert all(checks.values())
