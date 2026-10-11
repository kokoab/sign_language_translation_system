from pathlib import Path
import json,hashlib,subprocess
D=Path(__file__).resolve().parent;R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios');checks={}
def read(n):return json.loads((D/n).read_text())
def rows(n):return read(n)['records']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
s=read('merge_summary.json');stress=read('stress_summary.json');m=read('candidate_source_manifest.json')
for kind,n in [('all',24),('validation',570),('genuine',59),('phase',72),('splice',66)]:checks['complete_'+kind]=len(rows('merge_'+kind+'.json'))==n
checks['all_nominal_sequences_exact']=s['candidate_schedule']['full_sequences_exact']==24
checks['all_stress_sequences_exact']=stress['merge']['full_sequences_exact']==12
checks['stress_same_processed_timestamps']=stress['identical_processed_timestamps']
for kind in ['validation','genuine','splice']:checks[kind+'_hypotheses_unchanged']=all(v['changed']==0 for k,v in s[kind].items() if k!='changes')
a=rows('triple_baseline.json');b=rows('triple_merge.json');checks['triple_24_exact_preserved']=len(a)==len(b)==24 and all(x['id']==y['id'] and x['hypothesis']==y['hypothesis']==y['reference'] for x,y in zip(a,b))
checks['all_100_citizen_classes']=len({r['reference'][0] for r in rows('merge_validation.json') if r['dataset']=='citizen_val' and len(r['reference'])==1})==100
h=s['historical_reviewed'];checks['sept30_24_retained']=h['candidate']['sept30']['exact']==h['baseline']['sept30']['exact']==24
checks['older_first_MY_24']=h['candidate']['oct11']['target_then_following_sign']==24
checks['HAVE_reviewed_unchanged']=h['candidate']['have_take11']==h['baseline']['have_take11']
checks['HAVE_all_hypotheses_unchanged']=not any(x['id'].startswith('have_take11') for x in s['phase']['changes'])
checks['62_training_arms']=s['total_training_arms']==62
x=read('confidence_training_results.json');checks['encoders_frozen']=x['encoders_unchanged'] is True
checks['isolated_formerly_correct_retained']=all(v['previously_correct_lost']==0 for v in x['selected']['retention'].values())
for k,v in read('confidence_fp16_parity.json').items():checks['original_'+k+'_bit_exact']=v['bit_exact'] and v['n']==v['top1_agreement']==378 and v['max_abs']==0
recipe=read('confidence_training_recipe.json');checks['all_57_recipe_pins_current']=len(recipe['pins'])==57 and all(sha(Path(p))==v for p,v in recipe['pins'].items())
checks['canonical_not_training_ready']=json.loads(Path('active/v17/approved_phrase_manifest_20260921_v2.json').read_text())['training_ready'] is False
checks['candidate_source_and_model_pins_current']=all(sha(D/p)==v for p,v in m['pins'].items())
checks['production_sources_match_authorized_release']=all(sha(R/p)==v for p,v in read('deployment/sources_deployed.json').items())
checks['normal_release_deployment_verified']=read('deployment/verification.json')['all_pass']
checks['phone_build_success']=read('phone_build_status.json')['returncode']==0
phone=read('phone_comparison_summary.json')
checks['phone_xctest_passed']=read('phone_test_status.json')['returncode']==phone['xctest_returncode']==0
checks['phone_ABBA_complete']=[(p['arm'],p['pass_index']) for p in phone['passes']]==[('baseline',0),('candidate',1),('candidate',2),('baseline',3)]
checks['phone_candidate_both_exact']=phone['summary']['candidate']['full_sequences_exact']==2 and phone['summary']['candidate']['token_errors']==0
checks['phone_baseline_measured_errors']=phone['summary']['baseline']['full_sequences_exact']==0 and phone['summary']['baseline']['token_errors']==12
checks['phone_English_ready_and_nominal']=all(p['stage3_ready'] and p['thermal_before']==p['thermal_after']==0 for p in phone['passes'])
checks['phone_raw_results_valid']=all(read(f"my_transition_{p['arm']}_{p['pass_index']}.json")['hypothesis']==p['hypothesis'] and sha(D/f"my_transition_{p['arm']}_{p['pass_index']}.json")==p['result_sha256'] for p in phone['passes'])
checks['normal_release_install_verified']='App installed:' in (D/'restore_normal_app.log').read_text() and read('restore_normal_app_command.json')[-1]=='/Users/frnzlo/slt_ordered_my_like_gap175_build/Build/Products/Release-iphoneos/Runner.app'
checks['normal_release_launch_verified']='Launched application with com.kokoab.sltMobileApp bundle identifier.' in (D/'restore_normal_launch.log').read_text()
result=subprocess.run(['git','diff','--check','--','PROJECT_GROUND_TRUTH.md','docs/ground_truth/live-streaming/log.md','docs/ground_truth/live-streaming/high.md'],capture_output=True,text=True)
checks['task_scoped_diff_check']=result.returncode==0
out=dict(all_pass=all(checks.values()),checks=checks,n=len(checks),limitations=['Physical phone replay passed; candidate installed as normal release on user request. Fresh signing and sustained thermals untested.','Known-video development replays, not independent generalization.','Existing HAVE omissions remain.','Full git diff --check has pre-existing scripts/app_shell_v17.py:632 trailing whitespace.'])
(D/'final_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(sum(checks.values()),'/',len(checks),'checks pass');print([k for k,v in checks.items() if not v]);assert out['all_pass']
