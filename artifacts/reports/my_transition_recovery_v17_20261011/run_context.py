from pathlib import Path
import json, subprocess,os,sys,time
D=Path(__file__).resolve().parent;R=D.parent/'my_like_regression_v17_20261011/resources'
name=sys.argv[1];model={'current':'SpanRecognizerV17FragmentRejectLettersB8FP16','localA':'SpanRecognizerV17LocalALettersB8FP16'}[name]
env=dict(os.environ,MODEL=model,LETTER_THRESHOLD='0.5',FINISH='1',MOTION='on')
cmd=[str(D/'video_probe'),str(R),str(D/'context_cases.json'),str(D/(name+'_context.json'))]
(D/(name+'_command.json')).write_text(json.dumps({'command':cmd,'environment':{k:env[k] for k in ['MODEL','LETTER_THRESHOLD','FINISH','MOTION']},'started':time.time()},indent=2))
with (D/(name+'_context.log')).open('w') as f: subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('COMPLETE',name,flush=True)
