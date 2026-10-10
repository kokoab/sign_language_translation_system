from pathlib import Path
import json,hashlib,subprocess,sys,os
R=Path('/Volumes/secret/SLT/SLT');O=Path(__file__).resolve().parent
source=R/'scripts/train_span_recognizer_v17.py';s=source.read_text()
s=s.replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(R)!r})')
needle='best, started = None, time.perf_counter()';assert needle in s
s=s.replace(needle,'best = dict(history[0], state=copy.deepcopy(model.state_dict())) if history[0]["eligible"] else None\n    started = time.perf_counter()')
(O/'train_recovery.py').write_text(s)
base=R/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe/august/span_recognizer/best_model.pth'
recipe=R/'active/v17/phrase_segment_recipe_manifest_20261007.json';raw=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
cmd=[str(R/'venv/bin/python'),str(O/'train_recovery.py'),'--output-dir',str(O/'span_head_adaptation'),'--base',str(base),'--recipe-manifest',str(recipe),'--raw-cache',str(raw),'--sets','local_train','--exclude-prefix-spans','--freeze-encoders','--epochs','6','--samples','6000','--span-share','0.75','--head-lr','0.0001','--seed','27927','--floors',json.dumps({'citizen':94.77,'semlex':88.0,'local':95.5}),'--cfg','{"lookahead":8,"span_mode":"both","alpha":0.7,"w_r":2,"w_a":0.25,"w_b":0.5,"c":-1,"log_theta":-1.6094}']
pins=json.load(open(O/'letter_recipe.json'))
contract={'scope':'frozen-encoder fusion-head continuation on approved native phrase span inputs and isolated replay; not phone-video training','command':cmd,'epoch_zero_eligible_for_selection':True,'no_synthetic_fragment_or_reject_views':True,'manifest_sha256':hashlib.sha256(recipe.read_bytes()).hexdigest(),'trainer_sha256':hashlib.sha256((O/'train_recovery.py').read_bytes()).hexdigest(),'input_pins':pins['pins'],'selection':'tune WER subject to fixed retention floors, including epoch0; phone failures are post-training diagnostics only','test_accessed':False}
(O/'adaptation_recipe.json').write_text(json.dumps(contract,indent=2)+'\n')
if sys.argv[1:]==['worker']:
 try:
  with (O/'adaptation.log').open('w') as f:r=subprocess.run(cmd,cwd=R,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT)
  (O/'adaptation_completion.json').write_text(json.dumps({'returncode':r.returncode})+'\n')
 finally:subprocess.run(['osascript','-e','display notification "Bounded span recovery experiment finished." with title "SLT experiment"'],check=False)
else:
 with (O/'adaptation_worker.log').open('w') as f:p=subprocess.Popen([str(R/'venv/bin/python'),__file__,'worker'],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (O/'adaptation_launch.json').write_text(json.dumps({'pid':p.pid,'detached':True,'notification':'osascript on completion'})+'\n');print('Detached bounded recovery adaptation launched',p.pid)
