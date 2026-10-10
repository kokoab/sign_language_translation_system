from pathlib import Path
import os,subprocess,json,time
P=Path(__file__).resolve().parent; R=P.parents[2]
M=R/'artifacts/reports/my_like_regression_v17_20261011'; H=R/'artifacts/reports/have_root_fix_v17_20261010'
cases={n:M/(n+'_trace.json') for n in ['first_miss','long_word_run','repeat_opening']}
cases.update(take11=H/'take11_trace.json',take0955=H/'0955_trace.json',live1006=R/'data/local/phone_session_pull_v17_20261010b/live_reel_traces/20261010_100646.json')
arms=['current','rollback_both','ordered','rollback_none','rollback_geometry','rollback_roll']
for arm in arms:
 for case,trace in cases.items():
  out=P/f'{arm}_{case}.json'
  if out.exists():continue
  env=dict(os.environ,ARM=arm)
  with (P/f'{arm}_{case}.log').open('w') as f:
   subprocess.run([str(P/'trace_probe'),str(M/'resources'),str(trace),str(out)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads(out.read_text()); print(arm,case,','.join(d['hypothesis']),flush=True)
 out=P/f'{arm}_videos.json'
 if not out.exists():
  with (P/f'{arm}_videos.log').open('w') as f:
   subprocess.run([str(P/'video_probe'),str(M/'resources'),str(M/'rollback_review/videos.json'),str(out)],env=dict(os.environ,ARM=arm,FINISH='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads(out.read_text()); print(arm,'VIDEOS',[(r['id'],r['hypothesis'],r['spell_events'][:2]) for r in d['records']],flush=True)
print('COMPLETE',flush=True)
