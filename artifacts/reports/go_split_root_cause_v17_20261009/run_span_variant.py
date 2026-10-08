"""Span-recognizer variant = installed fragment recipe (run_v2 base/data/seed/epochs/selection + --fragment-prob 0.3)
with overrides: run_span_variant.py NAME [--raw-cache DIR] [extra trainer args...]. Output run_NAME/span_recognizer."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent
V2=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2'
name,extra=sys.argv[1],sys.argv[2:];OUT=HERE/f'run_{name}';OUT.mkdir(exist_ok=True)
sel=json.loads((V2/'phrase_adapt/result.json').read_text())['selected']['domains']
floors={d:(v-1.0 if v-1.0>=90.0 else v) for d,v in ((d,sel[d]['top1']) for d in ('citizen','semlex','local'))}
SPAN_CFG='{"lookahead":8,"span_mode":"both","alpha":0.7,"w_r":2,"w_a":0.25,"w_b":0.5,"c":-1,"log_theta":-1.6094}'
raw=str(V2/'raw_cache_citizen_variant')
if '--raw-cache' in extra:i=extra.index('--raw-cache');raw=extra[i+1];extra=extra[:i]+extra[i+2:]
cmd=[str(ROOT/'venv/bin/python'),'scripts/train_span_recognizer_v17.py','--output-dir',str(OUT/'span_recognizer'),'--base',str(V2/'phrase_adapt/best_model.pth'),
     '--epochs','8','--seed','27927','--sets','local_train','--exclude-prefix-spans','--recipe-manifest',str(ROOT/'active/v17/phrase_segment_recipe_manifest_20261007.json'),
     '--raw-cache',raw,'--floors',json.dumps(floors),'--cfg',SPAN_CFG,'--fragment-prob','0.3']+extra
(OUT/'command.json').write_text(json.dumps(cmd,indent=1))
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
with open(OUT/'span_recognizer.log','w') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(OUT/'status.json').write_text(json.dumps(dict(returncode=r.returncode))+'\n');print('exit',r.returncode)
