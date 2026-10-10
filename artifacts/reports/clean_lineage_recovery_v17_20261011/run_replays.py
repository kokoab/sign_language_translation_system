from pathlib import Path
import subprocess,json,os,time,sys
O=Path(__file__).resolve().parent;P=O.parent/'lineage_runtime_investigation_v17_20261011';resources=O/'resources';resources.mkdir(exist_ok=True)
for p in (O.parent/'my_like_regression_v17_20261011/resources').iterdir():
 t=resources/p.name
 if not t.exists():t.symlink_to(p.resolve(),target_is_directory=p.is_dir())
p=O/'SpanRecognizerV17CleanRecoveryLettersB8FP16.mlpackage';t=resources/p.name
if not t.exists():t.symlink_to(p.resolve(),target_is_directory=True)
assert json.load(open(O/'candidate_result.json'))['status']=='complete'
for variant in (sys.argv[1:] or ['control','continuous','continuous_contact']):
 t=time.time()
 with (O/(variant+'_replay.log')).open('w') as log:
  subprocess.run([str(O/(variant+'_probe')),str(resources),str(P/'cases.json'),str(O/(variant+'_replay.json'))],env=dict(os.environ,FINISH='0',LETTER_THRESHOLD='0.7'),stdout=log,stderr=subprocess.STDOUT,check=True)
 print(variant,'done',round(time.time()-t,1),flush=True)
 for r in json.load(open(O/(variant+'_replay.json')))['records']:print(r['id'],r['hypothesis'],flush=True)
