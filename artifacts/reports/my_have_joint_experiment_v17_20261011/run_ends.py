from pathlib import Path
import json,subprocess,os
P=Path(__file__).resolve().parent;R=P.parents[2];M=R/'artifacts/reports/my_like_regression_v17_20261011';H=R/'artifacts/reports/have_root_fix_v17_20261010'
cases={n:M/(n+'_trace.json') for n in ['first_miss','long_word_run','repeat_opening']}
cases.update(take11=H/'take11_trace.json',take0955=H/'0955_trace.json',live1006=R/'data/local/phone_session_pull_v17_20261010b/live_reel_traces/20261010_100646.json')
for case,path in cases.items():
 out=P/f'ordered_ends_{case}.json'
 with out.with_suffix('.log').open('w') as f:subprocess.run([str(P/'ends_trace_probe'),str(M/'resources'),str(path),str(out)],env=dict(os.environ,ARM='ordered'),stdout=f,stderr=subprocess.STDOUT,check=True)
 print(case,json.loads(out.read_text())['hypothesis'],flush=True)
for short,model in [('local_a','SpanRecognizerV17LocalALettersB8FP16'),('fragment','SpanRecognizerV17FragmentLettersB8FP16')]:
 out=P/f'{short}_ordered_ends_videos.json'
 with out.with_suffix('.log').open('w') as f:subprocess.run([str(P/'ends_video_probe'),str(M/'resources'),str(M/'rollback_review/videos.json'),str(out)],env=dict(os.environ,ARM='ordered',MODEL=model,FINISH='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
 print(short,'VIDEOS DONE',flush=True)
