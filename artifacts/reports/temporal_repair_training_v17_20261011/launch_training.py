from pathlib import Path
import json,subprocess,sys,os
O=Path(__file__).resolve().parent;R=O.parents[2];py=R/'venv/bin/python'
if sys.argv[1:]==['worker']:
 code=1
 try:
  with (O/'training.log').open('w') as f:code=subprocess.run([str(py),str(O/'train.py')],cwd=R,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT).returncode
 finally:
  (O/'training_completion.json').write_text(json.dumps({'returncode':code})+'\n');subprocess.run(['osascript','-e','display notification "Bounded temporal head experiment finished; inspect results." with title "SLT experiment"'],check=False)
 sys.exit(code)
else:
 with (O/'training_worker.log').open('w') as f:p=subprocess.Popen([str(py),__file__,'worker'],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (O/'training_launch.json').write_text(json.dumps({'pid':p.pid,'detached':True,'notification':'osascript on completion'})+'\n');print('Detached four-arm training launched',p.pid,flush=True)
 # Await completion output once; do not poll training metrics or checkpoint files.
 code=p.wait();print('Training worker completed',code,flush=True);sys.exit(code)
