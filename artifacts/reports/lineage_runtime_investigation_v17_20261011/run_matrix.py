from pathlib import Path
import os,json,subprocess,time
O=Path(__file__).resolve().parent;R=O.parent/'my_like_regression_v17_20261011/resources'
models={'old':'SpanRecognizerV17LocalALettersB8FP16','new':'SpanRecognizerV17FragmentRejectLettersB8FP16'}
for runtime in ['historical','current']:
 for name,model in models.items():
  key=f'{runtime}_{name}';t=time.time()
  with (O/(key+'.log')).open('w') as log:
   p=subprocess.run([str(O/(runtime+'_probe')),str(R),str(O/'cases.json'),str(O/(key+'.json'))],env=dict(os.environ,MODEL=model,FINISH='0'),stdout=log,stderr=subprocess.STDOUT)
  print(key,'exit',p.returncode,'seconds',round(time.time()-t,1),flush=True)
  if p.returncode:raise SystemExit(p.returncode)
  for row in json.load(open(O/(key+'.json')))['records']:
   print(row['id'],row['hypothesis'],flush=True)
