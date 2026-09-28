import json, os, subprocess, time
from pathlib import Path
root=Path('/Volumes/secret/SLT/SLT')
report=root/'artifacts/reports/stage3_composition_v17_20260929'
start=time.time(); result={'started_unix':start,'pid':os.getpid()}
try:
 for action in ['train','probe']:
  subprocess.run([str(root/'venv/bin/python'),'-u',str(root/'scripts/repair_stage3_composition_v17.py'),action],cwd=root,check=True)
 result['status']='complete'
except BaseException as e:
 result.update(status='failed',error=repr(e))
finally:
 result['seconds']=time.time()-start
 (report/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
 try:
  subprocess.run(['/usr/bin/osascript','-e','display notification "Stage 3 model repair '+result['status']+'" with title "SLT"'],timeout=10,capture_output=True)
 except Exception: pass
 print(json.dumps(result),flush=True)
