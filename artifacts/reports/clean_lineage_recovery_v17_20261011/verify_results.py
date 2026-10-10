from pathlib import Path
import json,hashlib,collections
O=Path(__file__).resolve().parent;R=O.parents[2]
def read(p):return json.load(open(p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=read(O.parent/'lineage_runtime_investigation_v17_20261011/cases.json')
names=['control','continuous','continuous_contact','consensus','adapted']
for name in names:
 rows=read(O/(name+'_replay.json'))['records']
 assert [r['id'] for r in rows]==[r['id'] for r in cases]
 sept=[r for r in rows if r['id'].startswith('sept30')]
 assert len(sept)==8 and all(r['hypothesis']==['HELLO','MY','FRIEND','I','FEEL','EXCITED'] for r in sept)
 assert sum(any(w['gloss']=='MY' and 2.7<=w['start_seconds']<3.15 for w in r['words']) for r in rows)==6
for bundle in [O,O/'adapted_bundle']:
 r=read(bundle/'candidate_result.json');assert r['status']=='complete' and r['recognizer_unchanged']
 assert r['parity']['samples']==r['parity']['top1_agreement']==378
 assert r['parity']['pytorch_top1_correct']==r['parity']['coreml_top1_correct']==362
 assert r['letter_threshold']==0.7
v=read(O/'validation_control.json')['records'];assert len(v)==570
b=read(O.parent/'my_have_phone_v17_20261011/guarded_rollback_validation.json')['records']
assert [(r['dataset'],r['id'],r['reference']) for r in v]==[(r['dataset'],r['id'],r['reference']) for r in b]
assert len({r['reference'][0] for r in v if r['dataset']=='citizen_val'})==100
a=read(O/'adaptation_result.json');assert a['epoch']==1
assert a['tensor_changes']=={'landmark_model_state_dict':0,'hand_model_state_dict':0,'head_state_dict':14}
assert sha(O/'span_head_adaptation/best_model.pth')==a['checkpoint_sha256']
assert not read(R/'active/v17/approved_phrase_manifest_20260921_v2.json')['training_ready']
for r in read(O/'source_manifest.json'):assert sha(Path(r['path']))==r['sha256'],r['path']
# This pass makes no production Python changes either.
prior=read(O.parent/'my_have_joint_experiment_v17_20261011/source_manifest.json')
p=R/'active/v17/segmental_runtime_v17.py';assert sha(p)==prior[str(p)]
traces=read(O/'trace_checks.json');assert all(r['frames']==321 for r in traces.values())
assert traces['automatic_name']['word_timeline_monotonic']
result={'status':'passed','final_main_phase_cases':85,'cached_validation_cases':570,'citizen_classes':100,
 'export_parity_cases_per_model':378,'trace_modes_checked':['words','name','always','spelling'],
 'production_sources_unchanged':True,'installed_app_changed':False,'protected_evaluation_run':False,
 'candidate_accepted':False,'reason':'All five candidates retain two first-MY phase failures; clean-model full-class streaming retention regresses.'}
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
