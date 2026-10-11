from pathlib import Path
import json,hashlib,subprocess,plistlib
D=Path(__file__).resolve().parent;O=D.parent;R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios');A=Path('/Users/frnzlo/slt_my_transition_release_build/Build/Products/Release-iphoneos/Runner.app');B=Path('/Users/frnzlo/slt_ordered_my_like_gap175_build/Build/Products/Release-iphoneos/Runner.app')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks={}
checks['release_build_passed']=json.loads((D/'release_build_status.json').read_text())['returncode']==0
checks['nine_cleanup_parity_cases_passed']=all(json.loads((D/f).read_text())['all_pass'] for f in ['parity.json','like_parity.json'])
checks['deployed_source_hashes_match']=all(sha(R/p)==v for p,v in json.loads((D/'sources_deployed.json').read_text()).items())
checks['RunnerTests_unchanged']=sha(R/'RunnerTests/RunnerTests.swift')==json.loads((D/'sources_before.json').read_text())['RunnerTests/RunnerTests.swift']
checks['no_test_bundle_shipped']=not list(A.rglob('*.xctest'))
name='SpanRecognizerV17ConfidenceDualB8FP16'
checks['candidate_compiled_model_bundled']=(A/(name+'.mlmodelc/model.mil')).exists()
checks['compiled_candidate_weight_matches_tested_source']=sha(A/(name+'.mlmodelc/weights/weight.bin'))==sha(O/(name+'.mlpackage/Data/com.apple.CoreML/weights/weight.bin'))
checks['all_existing_compiled_weights_unchanged']=all((A/p.relative_to(B)).exists() and sha(A/p.relative_to(B))==sha(p) for p in B.glob('*.mlmodelc/weights/weight.bin'))
checks['bundled_model_selected_by_default']='init(name: String = "'+name+'"' in (R/'Runner/LiveReel/LiveReelModels.swift').read_text()
checks['no_diagnostic_model_override']='my_transition_20261012/' not in (R/'Runner/LiveReel/LiveReelModels.swift').read_text()
checks['correct_bundle_id']=plistlib.loads((A/'Info.plist').read_bytes())['CFBundleIdentifier']=='com.kokoab.sltMobileApp'
with (D/'codesign_verify.log').open('w') as f:p=subprocess.run(['codesign','--verify','--deep','--strict','--verbose=2',str(A)],stdout=f,stderr=subprocess.STDOUT)
checks['signature_valid']=p.returncode==0
for name in ['LiveReelDecoder.swift','LiveReelModels.swift']:
 p=subprocess.run(['git','diff','--no-index','--check',str(D/'before/Runner/LiveReel'/name),str(R/'Runner/LiveReel'/name)],capture_output=True,text=True)
 checks[name+'_diff_check']=p.returncode in (0,1) and not p.stdout and not p.stderr
out=dict(all_pass=all(checks.values()),checks=checks,app=str(A),executable_sha256=sha(A/'Runner'),compiled_candidate_weight_sha256=sha(A/'SpanRecognizerV17ConfidenceDualB8FP16.mlmodelc/weights/weight.bin'),retained_existing_weight_count=len(list(B.glob('*.mlmodelc/weights/weight.bin'))))
(D/'release_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(sum(checks.values()),'/',len(checks),'release checks pass');print([k for k,v in checks.items() if not v]);assert out['all_pass']
