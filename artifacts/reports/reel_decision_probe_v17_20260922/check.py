"""Run linear probe self-checks and verify completed comparison invariants."""
import json,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
m=runpy.run_path(str(HERE/'run.py'));m['self_check']()
def row(x,kind):
 return dict(temporal=[x]*4,kind=kind,proposal={'model_score':.5},verifier={'model_score':.5})
r=[row(-1,'gap'),row(-2,'gap'),row(1,'core'),row(2,'core')]
s=m['fit_scores'](r,r)
assert np.isfinite(s).all() and max(s[:2])<min(s[2:])
if (HERE/'results.json').exists():
 result=json.loads((HERE/'results.json').read_text())
 evidence=json.loads((HERE/'evidence.json').read_text())
 old=json.loads((HERE.parent/'reel_matched_windows_v17_20260922/results.json').read_text())
 current={(r['id'],r['kind']):r for r in evidence['results'] if r['role']=='validation'}
 assert len(current)==len(old['results'])
 for r in old['results']:
  v=current[r['id'],r['kind']]
  assert r['conditional_commit']==v['conditional_commit']
  assert r['verifier']['candidate_gloss']==v['verifier']['candidate_gloss']
  assert abs(r['verifier']['model_score']-v['verifier']['model_score'])<1e-5
 for name,arm in result['arms'].items():
  for kind,metrics in arm['validation'].items():
   baseline=result['arms']['unchanged']['validation'][kind]
   assert metrics['conditional_commits']<=baseline['conditional_commits']
 print('Completed evidence matches previous replay; veto counts consistent')
print('Sampling, overlap and linear-fit checks passed')
