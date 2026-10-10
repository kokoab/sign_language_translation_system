from pathlib import Path
import os,json,subprocess,hashlib
import torch
O=Path(__file__).resolve().parent;R=O.parents[2];d=O/'adapted_bundle'
r=json.load(open(d/'candidate_result.json'));assert r['status']=='complete'
name='SpanRecognizerV17AdaptedRecoveryLettersB8FP16'
t=O/'resources'/(name+'.mlpackage')
if not t.exists():t.symlink_to(d/t.name,target_is_directory=True)
# Provenance-only additions: exported tensors are not changed.
for bundle in [O,d]:
 p=bundle/'letter_head/model.pth';c=torch.load(p,map_location='cpu',weights_only=False)
 c['canonical_phrase_manifest_sha256']=hashlib.sha256((R/'active/v17/approved_phrase_manifest_20260921_v2.json').read_bytes()).hexdigest()
 c['recipe_manifest_sha256']=hashlib.sha256((R/'active/v17/phrase_segment_recipe_manifest_20261007.json').read_bytes()).hexdigest()
 c['letter_recipe_sha256']=hashlib.sha256((bundle/'letter_recipe.json').read_bytes()).hexdigest()
 torch.save(c,p)
env=dict(os.environ,MODEL=name,LETTER_THRESHOLD=str(r['letter_threshold']),FINISH='0')
with (O/'adapted_replay.log').open('w') as f:
 subprocess.run([str(O/'continuous_contact_probe'),str(O/'resources'),str(O.parent/'lineage_runtime_investigation_v17_20261011/cases.json'),str(O/'adapted_replay.json')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('Adapted phase replay complete',flush=True)
for name in ['candidate_paced','shared_candidate_paced']:
 times=O.parent/'my_have_phone_v17_20261011'/f'{name}_times.json'
 with (O/(name+'_adapted.log')).open('w') as f:
  subprocess.run([str(O/'continuous_contact_probe'),str(O/'resources'),str(O/'schedule_cases.json'),str(O/(name+'_adapted.json'))],env=dict(env,LIVE_TIMES=str(times)),stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'adapted replay complete',flush=True)
