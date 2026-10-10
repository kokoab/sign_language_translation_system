"""Await extraction artifacts, verify them, then run the admitted detached experiment."""
from pathlib import Path
import json,time,subprocess,sys
O=Path(__file__).resolve().parent;R=O.parents[2];py=R/'venv/bin/python';cases=json.loads((O/'exact_cases.json').read_text());last=None;stage='waiting_for_extraction'
try:
 while True:
  missing=[]
  for c in cases:
   roots=[O/'exact_native_inputs']+([O/'native_inputs'] if c['source']=='o5s5' else [])
   ready=False
   for root in roots:
    p=root/(c['id']+'.json')
    if not p.exists():continue
    try:d=json.loads(p.read_text())
    except json.JSONDecodeError:continue
    if len(d['rows'])==72*len(c['events']):ready=True;break
   if not ready:missing.append(c['id'])
  if not missing:break
  if last!=len(missing):print('Waiting for remaining parent outputs:',len(missing),flush=True);last=len(missing)
  time.sleep(5)
 for stage in ['finalize_inputs','materialize','prepare_training','test_admission','launch_training']:
  print('Running',stage,flush=True)
  with (O/(stage+'_pipeline.log')).open('w') as f:subprocess.run([str(py),str(O/(stage+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 (O/'pipeline_completion.json').write_text(json.dumps({'returncode':0,'stage':'training_complete'})+'\n');print('Materialization, admission and training complete.',flush=True)
except Exception as e:
 (O/'pipeline_completion.json').write_text(json.dumps({'returncode':1,'stage':stage,'error':repr(e)})+'\n');raise
