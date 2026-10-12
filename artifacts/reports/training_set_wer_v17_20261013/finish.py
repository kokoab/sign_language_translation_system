import os,time,subprocess,json
from pathlib import Path
O=Path(__file__).resolve().parent
R=O.parents[2]
while True:
 try:os.kill(77766,0)
 except ProcessLookupError:break
 time.sleep(20)
p=subprocess.run([str(R/'venv/bin/python'),str(O/'verify_results.py')],cwd=R,capture_output=True,text=True)
(O/'verification.log').write_text(p.stdout+p.stderr)
if p.returncode==0:
 subprocess.run([str(R/'venv/bin/python'),str(R/'scripts/index_large_artifacts_v17.py')],cwd=R,stdout=subprocess.DEVNULL)
 message='Training-set WER comparison complete and independently verified.'
else:message='Training-set WER evaluation needs review; see verification.log.'
(O/'completion.json').write_text(json.dumps({'verification_exit_code':p.returncode,'message':message})+'\n')
subprocess.run(['osascript','-e','display notification '+json.dumps(message)+' with title "ATLAS evaluation"'],capture_output=True)
