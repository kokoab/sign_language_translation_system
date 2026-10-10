from pathlib import Path
import os,subprocess,json
P=Path(__file__).resolve().parent;R=P.parents[2];M=R/'artifacts/reports/my_like_regression_v17_20261011';H=R/'artifacts/reports/have_root_fix_v17_20261010'
cases={n:M/(n+'_trace.json') for n in ['long_word_run','repeat_opening']};cases.update(take11=H/'take11_trace.json',take0955=H/'0955_trace.json',live1006=R/'data/local/phone_session_pull_v17_20261010b/live_reel_traces/20261010_100646.json')
for case,path in cases.items():
 out=P/f'hybrid_{case}.json'
 with out.with_suffix('.log').open('w') as f:subprocess.run([str(P/'hybrid_trace_probe'),str(M/'resources'),str(path),str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(case,json.loads(out.read_text())['hypothesis'],flush=True)
