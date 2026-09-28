import json,os,subprocess,time
from pathlib import Path
root=Path('/Volumes/secret/SLT/SLT');r=root/'artifacts/reports/stage3_composition_v17_20260929/request_continuation';start=time.time();result={'pid':os.getpid(),'started_unix':start}
try:
 for action in ['train','probe']:
  args=[str(root/'venv/bin/python'),'-u',str(root/'scripts/repair_stage3_composition_v17.py'),action,'--report',str(r),'--output',str(root/'artifacts/models/stage3_composition_v17_20260929_v2')]
  if action=='train':args += ['--init',str(root/'artifacts/models/stage3_composition_v17_20260929')]
  subprocess.run(args,cwd=root,check=True)
 result['status']='complete'
except BaseException as e:result.update(status='failed',error=repr(e))
finally:
 result['seconds']=time.time()-start;(r/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
 try:subprocess.run(['/usr/bin/osascript','-e','display notification "Stage 3 request continuation '+result['status']+'" with title "SLT"'],timeout=10,capture_output=True)
 except Exception:pass
 print(json.dumps(result),flush=True)
