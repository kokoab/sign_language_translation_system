from pathlib import Path
import sys,json,subprocess,os,time
O=Path(__file__).resolve().parent;name=sys.argv[1];model='SpanRecognizerV17FragmentRejectLettersB8FP16' if name=='incumbent' else f'SpanRecognizerV17Temporal{name.title()}B8FP16'
if name!='incumbent':
 p=O/(model+'.mlpackage');assert p.is_dir();q=O/'resources'/p.name
 if not q.exists():q.symlink_to(p.resolve(),target_is_directory=True)
for variant in ['control','continuous_contact']:
 output=O/f'{name}_{variant}_replay.json';start=time.time()
 with (O/f'{name}_{variant}_replay.log').open('w') as f:subprocess.run([str(O/(variant+'_probe')),str(O/'resources'),str(O/'evaluation_cases.json'),str(output)],env=dict(os.environ,MODEL=model,LETTER_THRESHOLD='0.5',FINISH='0'),stdout=f,stderr=subprocess.STDOUT,check=True)
 result=json.loads(output.read_text());assert len(result['records'])==72;print(name,variant,'complete',round(time.time()-start,1),flush=True)
