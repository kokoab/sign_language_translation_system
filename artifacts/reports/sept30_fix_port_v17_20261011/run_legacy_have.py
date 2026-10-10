from pathlib import Path
import subprocess,os,json,time
N=Path(__file__).resolve().parent;R=N.parent/'my_like_regression_v17_20261011/resources'
for arm,motion in [('base','off'),('guards','on')]:
 env=dict(os.environ,MODEL='SpanRecognizerV17LocalALettersB8FP16',LETTER_THRESHOLD='0.5',FINISH='1',MOTION=motion,PORT=arm)
 command=[str(N/'legacy_video_probe'),str(R),str(N/'have_cases.json'),str(N/('legacy_'+arm+'_have.json'))]
 (N/('legacy_'+arm+'_have_command.json')).write_text(json.dumps({'command':command,'environment':{k:env[k] for k in ['MODEL','LETTER_THRESHOLD','FINISH','MOTION','PORT']}}))
 with (N/('legacy_'+arm+'_have.log')).open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(arm,'HAVE done',time.ctime(),flush=True)
