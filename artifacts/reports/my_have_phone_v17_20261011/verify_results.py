from pathlib import Path
import json,hashlib,collections
p=Path(__file__).resolve().parent;q=p.parent/'my_have_joint_experiment_v17_20261011'
def read(n):return json.loads((p/n).read_text())
base=json.loads((q/'native_validation_installed.json').read_text())['records'];new=read('guarded_rollback_validation.json')['records']
assert len(base)==len(new)==570
lookup={(r['dataset'],r['id']):r for r in base}
changes=[r['id'] for r in new if r['hypothesis']!=lookup[(r['dataset'],r['id'])]['hypothesis']]
assert not changes,changes
assert len({r['reference'][0] for r in new if r['dataset']=='citizen_val'})==100
controlsA=read('controls_baseline_device.json')['records'];controlsB=read('controls_reset_candidate_device.json')['records']
assert len(controlsA)==len(controlsB)==59
assert [(r['id'],r['hypothesis']) for r in controlsA]==[(r['id'],r['hypothesis']) for r in controlsB]
phase=read('phase_reset_device.json')['records'];assert len(phase)==8
assert all([w['gloss'] for w in r['words'] if 8.8<w['start_seconds']<9.5]==['MY'] for r in phase)
assert len({tuple(r['hypothesis']) for r in phase})==1
phone=read('guarded_rollback_device.json')['records'];assert len(phone)==16
for r in phone:
 if r['kind']!='trace':continue
 if r['id']=='first_miss':assert r['hypothesis']==['HELLO','LIKE','NAME']
 if r['id']=='9dc94ebe93dd51139d48':assert r['hypothesis']==['PLEASE','HELP','I']
 if r['id'] in ['take11','take0955','live1006']:assert r['hypothesis'].count('HAVE')=={'take11':5,'take0955':8,'live1006':4}[r['id']]
# The normal shared-app candidate was rejected because the target remains LIKE.
for n in ['shared_candidate_paced.json','shared_guarded_rollback_paced.json']:
 assert [w['gloss'] for w in read(n)['words'] if 8.8<w['start_seconds']<9.5]==['LIKE']
root=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios');before=read('source_hashes_before.json')
current={r:hashlib.sha256((root/r).read_bytes()).hexdigest() for r in before}
assert [r for r in before if before[r]!=current[r]]==['Runner/LiveReel/LiveReelDecoder.swift']
assert (root/'Runner/LiveReel/LiveReelDecoder.swift').read_bytes()==(p/'guarded_rollback_LiveReelDecoder.swift').read_bytes()
assert 'myFallback' not in (root/'Runner/LiveReel/LiveReelDecoder.swift').read_text()
source_manifest=json.loads((q/'source_manifest.json').read_text())
for name in ['LiveReelEngine.swift','LiveReelCore.swift','LiveReelModels.swift','LiveReelStage3.swift']:
 path=root/'Runner/LiveReel'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==source_manifest[str(path)]
py=Path('/Volumes/secret/SLT/SLT/active/v17/segmental_runtime_v17.py');assert hashlib.sha256(py.read_bytes()).hexdigest()==source_manifest[str(py)]
summary={'validation_cases':570,'validation_counts':dict(collections.Counter(r['dataset'] for r in new)), 'validation_hypothesis_changes':0,'citizen_classes':100,'paired_phone_MY_LIKE_controls':59,'phone_control_changes':0,'rollback_have_counts':[5,8,4],'known_MY_unresolved':True,'source_changes':['Runner/LiveReel/LiveReelDecoder.swift'],'temporary_source_changes_removed':True}
(p/'verification_summary.json').write_text(json.dumps(summary,indent=2))
print('PASS:570pairedvalidationhypothesesunchanged acrossall100classes;59pairedphonecontrolsunchanged;8resetchecks;rollback5/8/4HAVE,orderedwords,oneHELP;knownMYfailureexplicit;onlyintendeddecoderchanged,diagnosticstartupandtesthooksremoved.')
