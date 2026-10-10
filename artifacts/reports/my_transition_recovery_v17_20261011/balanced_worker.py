from pathlib import Path
import subprocess,sys,json,time,traceback
D=Path(__file__).resolve().parent;started=time.time()
try:
 with (D/'balanced_training.log').open('w') as f:r=subprocess.run([sys.executable,str(D/'train_actual_balanced.py')],stdout=f,stderr=subprocess.STDOUT)
 result={'exit_code':r.returncode,'started':started,'finished':time.time()}
except Exception as e:result={'exit_code':-1,'error':traceback.format_exc(),'finished':time.time()}
(D/'balanced_completion.json').write_text(json.dumps(result,indent=2))
subprocess.run(['osascript','-e','display notification "Balanced-head MY recovery experiment finished. Results saved in my_transition_recovery_v17_20261011." with title "SLT experiment"'],capture_output=True)
