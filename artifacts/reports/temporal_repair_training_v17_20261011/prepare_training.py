"""Versioned recipe admission after complete native materialization; original gates unchanged."""
from pathlib import Path
import json,sys,copy
from features import O,R,BASE,sha
P=O.parent/'temporal_repair_contract_v17_20261011';sys.path.insert(0,str(P))
from verify_contract import verify
contract=json.loads((P/'contract.json').read_text());verify(contract)
declared=json.loads((O/'declared_recipe.json').read_text());assert sha(P/'contract.json')==declared['contract_sha256']
material=json.loads((O/'materialization.json').read_text());iso=json.loads((O/'isolated_feature_manifest.json').read_text())
assert material['summary']['events_attempted']==144
assert {e['id'] for e in material['events']}=={r['id'] for r in contract['records']}
assert sum(e['eligible']>0 and e['role']=='train' and e['label']=='MY' for e in material['events'])==28
assert all(any(e['role']==role and e['label']==label and e['eligible'] for e in material['events']) for role,label in [('train','HAVE'),('train','LIKE'),('validation','HAVE'),('validation','LIKE')])
pins={**material['pins'],**iso['pins'],**{str(R/p):h for p,h in contract['pins'].items()}}
for name in ['declared_recipe.json','materialization.json','isolated_feature_manifest.json','train.py','features.py','materialize.py','prepare_training.py']:
 p=O/name;pins[str(p)]=sha(p)
for folder in ['native','native_cached','native_cached_fast','native_oriented','native_context','native_prefill']:
 for p in (O/folder).glob('*.swift'):pins[str(p)]=sha(p)
for name in ['extraction_parity.json','detection_cache_parity.json','detection_cache_fast_parity.json','isolated_role_check.json','replay_source_hashes.json','audit_replay_roles.py','exact_core_contract.json','timestamp_correction.json','exact_cases.json','exact_bounds.py','rotation_verification.json','prepare_rotation.py','prepare_context_probe.py','prepare_prefill.py','prefill_parity_verification.json','prefill_replacement.json']:
 p=O/name;pins[str(p)]=sha(p)
for p,digest in pins.items():assert sha(p)==digest,p
recipe=copy.deepcopy(declared);recipe.update(status='ready_for_this_bounded_experiment',recipe_training_ready=True,pins=pins,input_summary=material['summary'],isolated_sets=iso['sets'],remaining_admission_checks=[],promotion_ready=False)
recipe['isolated_replay_role_audit']=json.loads((O/'isolated_role_check.json').read_text())
recipe['admission_note']='Full native cache audited; feature-quality exclusions are recorded before classifier training. Original draft contract remains immutable; v2 enables only this trainer and these pinned inputs. Independent-phone capture blocks promotion, not this experiment.'
(O/'training_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n');print('Scoped training recipe v2 verified; original canonical training flag unchanged.')
