from pathlib import Path
import subprocess,os,json,time
N=Path(__file__).resolve().parent;R=N.parent/'my_like_regression_v17_20261011/resources';P=N.parent/'temporal_repair_training_v17_20261011'
for kind,probe,cases in [('have','video',N/'have_cases.json'),('new_video','video',N/'new_video_cases.json'),('validation','validation',N.parent/'my_have_joint_experiment_v17_20261011/validation_traces.json'),('genuine','video',P/'genuine_control_cases.json')]:
 env=dict(os.environ,MODEL='SpanRecognizerV17LocalALettersB8FP16',LETTER_THRESHOLD='0.5',FINISH='1',MOTION='on',PORT='guards')
 command=[str(N/('rank_'+probe+'_probe')),str(R),str(cases),str(N/('rank_'+kind+'.json'))]
 (N/('rank_'+kind+'_command.json')).write_text(json.dumps({'command':command,'environment':{k:env[k] for k in ['MODEL','LETTER_THRESHOLD','FINISH','MOTION','PORT']}}))
 with (N/('rank_'+kind+'.log')).open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(kind,'DONE',time.ctime(),flush=True)
