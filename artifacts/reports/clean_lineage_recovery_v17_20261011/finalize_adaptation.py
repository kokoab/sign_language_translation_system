from pathlib import Path
import json,hashlib,subprocess,sys
import torch
O=Path(__file__).resolve().parent;R=O.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.load(open(O/'adaptation_completion.json'))['returncode']==0
p=O/'span_head_adaptation/best_model.pth'
c=torch.load(p,map_location='cpu',weights_only=False)
base=R/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe/august/span_recognizer/best_model.pth'
b=torch.load(base,map_location='cpu',weights_only=False)
groups=['landmark_model_state_dict','hand_model_state_dict','head_state_dict']
changes={g:sum(not torch.equal(t,b[g][k]) for k,t in c[g].items()) for g in groups}
assert changes[groups[0]]==changes[groups[1]]==0,changes
c['canonical_phrase_manifest_sha256']=sha(R/'active/v17/approved_phrase_manifest_20260921_v2.json')
c['recipe_manifest_sha256']=sha(R/'active/v17/phrase_segment_recipe_manifest_20261007.json')
c['recovery_recipe_sha256']=sha(O/'adaptation_recipe.json')
before=sha(p);torch.save(c,p)
check=torch.load(p,map_location='cpu',weights_only=False)
assert all(torch.equal(t,check[g][k]) for g in groups for k,t in c[g].items())
(O/'adaptation_result.json').write_text(json.dumps({'epoch':c['epoch'],'selected':c['span_adaptation']['selected'],'tensor_changes':changes,'checkpoint_before_metadata_sha256':before,'checkpoint_sha256':sha(p),'canonical_phrase_manifest_sha256':c['canonical_phrase_manifest_sha256'],'recipe_manifest_sha256':c['recipe_manifest_sha256'],'recovery_recipe_sha256':c['recovery_recipe_sha256']},indent=2)+'\n')
print(json.dumps({'epoch':c['epoch'],'tensor_changes':changes}),flush=True)
d=O/'adapted_bundle';d.mkdir(exist_ok=True)
s=(O/'build_candidate.py').read_text().replace("REC=R/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe/august/span_recognizer/best_model.pth'",f'REC=Path({str(p)!r})')
s=s.replace('SpanRecognizerV17CleanRecoveryLettersB8FP16','SpanRecognizerV17AdaptedRecoveryLettersB8FP16')
(d/'build_candidate.py').write_text(s)
with (d/'worker.log').open('w') as f:
 worker=subprocess.Popen([str(R/'venv/bin/python'),str(d/'build_candidate.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('Detached matched adapted letter-head/export, completion notification enabled, pid',worker.pid)
