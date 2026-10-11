from pathlib import Path
import sys,os,json,time,subprocess
D=Path(__file__).resolve().parent;R=D/'resources' if (D/'resources').exists() else D.parent/'my_like_regression_v17_20261011/resources'
exe,cases,name=sys.argv[1:4];model=sys.argv[4] if len(sys.argv)>4 else 'current'
env=dict(os.environ,MODEL={'current':'SpanRecognizerV17FragmentRejectLettersB8FP16','localA':'SpanRecognizerV17LocalALettersB8FP16'}.get(model,model),LETTER_THRESHOLD='0.5',FINISH='1',MOTION='on')
cmd=[str(D/exe),str(R),str(D/cases),str(D/(name+'.json'))]
(D/(name+'_command.json')).write_text(json.dumps({'command':cmd,'environment':{k:env[k] for k in ['MODEL','LETTER_THRESHOLD','FINISH','MOTION']},'started':time.time()},indent=2))
with (D/(name+'.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('COMPLETE',name,flush=True)
