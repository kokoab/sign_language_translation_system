"""Verify saved measured claims; no model invocation or held-out visual data access."""
import collections,hashlib,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda name:json.loads((P/name).read_text())
identity=read('identity.json');summary=read('summary.json');allrows=[]
for f,h in identity['sources'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
for prefix in ['', 'extended_', 'phone_', 'locking_']:
 cases=read(prefix+'cases.json');results=[json.loads(l) for l in (P/(prefix+'results.jsonl')).open()]
 assert len(cases)==len(results)
 assert [r['id'] for r in cases]==[r['id'] for r in results]
 assert len({r['id'] for r in cases})==len(cases)
 allrows.extend({**c,**r} for c,r in zip(cases,results))
assert len(allrows)==1188
assert len(read('vocabulary_audit.json'))==100
assert summary['total_cases']==865 and summary['incremental_cases']==765
assert summary['incremental_differences']==[]
assert summary['singleton_focus_missing']==['I','TALK']
assert summary['role_realization']=={'roles':{'total':80,'passed':78},'give_roles':{'total':40,'passed':40}}
assert summary['possession_owner_retained']==36
assert len(summary['past_future_scope']['failed'])==8 and summary['past_future_scope']['total']==32
assert summary['global_training_word_rows']['food']==0 and summary['training_rows']==52961
assert sum(v['total'] for v in summary['counterfactual_gate'].values())==120
assert all(v['faithful_accepted']==v['corrupted_accepted']==v['total'] for v in summary['counterfactual_gate'].values())
locks=[r for r in allrows if r['family']=='locking_regression']
assert all(r['expected_full']==r['full'] and r['expected_incremental']==r['incremental'] and r['expected_locks']==r['locked'] for r in locks)
locking=dict(cases=len(locks),full_matches=sum(r['expected_full']==r['full'] for r in locks),incremental_matches=sum(r['expected_incremental']==r['incremental'] for r in locks),lock_count_matches=sum(r['expected_locks']==r['locked'] for r in locks),active_locks=sum(r['locked']>0 for r in locks),full_incremental_different=sum(r['full']!=r['incremental'] for r in locks))
assert locking['cases']==323 and locking['active_locks']==180 and locking['full_incremental_different']==57
(P/'locking_summary.json').write_text(json.dumps(locking,indent=2)+'\n')
forms={'i':'I','me':'I','you':'YOU','we':'WE','us':'WE','they':'THEY','them':'THEY','he':'HE','him':'HE','she':'HE','her':'HE'}
topic=[]
for r in allrows:
 if r['family']=='topic_order':
  actual={forms[t] for t in re.findall(r'[a-z]+',r['full'].lower()) if t in forms};expected={r['subject'],r['object']}
  topic.append(dict(id=r['id'],expected=sorted(expected),actual=sorted(actual),pass_participant_set=actual==expected))
assert len(topic)==40 and sum(r['pass_participant_set'] for r in topic)==31
(P/'topic_participant_checks.json').write_text(json.dumps(topic,indent=2)+'\n')
byid={r['id']:r for r in allrows}
for key,text in {'meaning_003':'Where is my eat?','meaning_008':'The drink is cold.','ambiguous_004':'I give it to him.','time_scope_017':'Yesterday I worked and I slept tomorrow.','isolated_045':'Ask.'}.items():assert byid[key]['full']==text,(key,byid[key]['full'])
assert 'see her tomorrow' in byid['locking_gen034']['incremental'] and 'see them tomorrow' in byid['locking_gen034']['full']
# Review artifacts contain expected fallback-free paths. They do not prove meaning correctness.
assert all(r['mode'].startswith('t5_efficient_tiny') for r in allrows)
inputs=['cases.json','extended_cases.json','phone_cases.json','locking_cases.json','main.swift','build_experiment.py','extend_experiment.py','analyze.py','check_contracts.py','reproduce.sh']
final=dict(date='2026-10-11',total_native_cases=1188,main_summary=summary,historical_lock_controls=locking,topic_order_participant_sets=dict(total=40,preserved=31),corrupted_semantic_controls_accepted=120,lexical_controls=59,production_changed=False,training_performed=False,installed=False,api_spend=0,input_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in inputs})
(P/'final_summary.json').write_text(json.dumps(final,indent=2)+'\n')
print('PASS: 1,188 native cases; all100classes; 52,961train rows; 120corruptions; 323historical parity cases including180locks; source/model hashes unchanged.')
