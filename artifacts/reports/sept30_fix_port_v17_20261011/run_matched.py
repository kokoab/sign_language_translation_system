from pathlib import Path
import subprocess,os,sys,time,json
N=Path(__file__).resolve().parent;G=N.parent/'ordered_my_like_v17_20261011/gap175';R=N.parent/'my_like_regression_v17_20261011/resources';P=N.parent/'temporal_repair_training_v17_20261011'
worker=sys.argv[1];jobs=[]
if worker=='localA':
 for kind,exe,cases in [('new_video',G/'video_probe',N/'new_video_cases.json'),('new_trace',G/'validation_probe',N/'new_trace_cases.json'),('validation',G/'validation_probe',N.parent/'my_have_joint_experiment_v17_20261011/validation_traces.json'),('phase',G/'video_probe',P/'evaluation_cases.json'),('genuine',G/'video_probe',P/'genuine_control_cases.json')]:jobs.append(('localA',kind,exe,cases,'on','guards'))
else:
 jobs.append(('current','new_video',G/'video_probe',N/'new_video_cases.json','on','guards'))
 jobs.append(('current','new_trace',G/'validation_probe',N/'new_trace_cases.json','on','guards'))
 for arm,motion in [('base','off'),('guards','on')]:
  for kind,exe,cases in [('new_video',N/'legacy_video_probe',N/'new_video_cases.json'),('validation',N/'legacy_validation_probe',N.parent/'my_have_joint_experiment_v17_20261011/validation_traces.json'),('genuine',N/'legacy_video_probe',P/'genuine_control_cases.json')]:jobs.append(('legacy_'+arm,kind,exe,cases,motion,arm))
for arm,kind,exe,cases,motion,port in jobs:
 env=dict(os.environ,MODEL='SpanRecognizerV17FragmentRejectLettersB8FP16' if arm=='current' else 'SpanRecognizerV17LocalALettersB8FP16',LETTER_THRESHOLD='0.5',FINISH='1',MOTION=motion,PORT=port)
 print('START',arm,kind,time.ctime(),flush=True)
 command=[str(exe),str(R),str(cases),str(N/(arm+'_'+kind+'.json'))]
 (N/(arm+'_'+kind+'_command.json')).write_text(json.dumps({'command':command,'environment':{k:env[k] for k in ['MODEL','LETTER_THRESHOLD','FINISH','MOTION','PORT']}}))
 with (N/(arm+'_'+kind+'.log')).open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',arm,kind,time.ctime(),flush=True)
