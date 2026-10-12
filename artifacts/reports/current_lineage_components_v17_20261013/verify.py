import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
r=json.loads((O/'results.json').read_text());assert r['state']=='complete'
assert {k:v['n'] for k,v in r['domains'].items()}=={'citizen':378,'semlex':953,'local':2626}
for d in r['domains'].values():
 assert len(d['ids'])==len(set(d['ids']))==d['n']
 for k,p in d['predictions'].items():
  assert len(p)==d['n'] and sum(a==b for a,b in zip(p,d['targets']))==d['correct'][k]
for k,v in r['summary'].items():
 assert v['n']==3957 and v['correct']==sum(d['correct'][k] for d in r['domains'].values())
prior=json.loads((O.parents[2]/'docs/capstone_papers/review_assets/20261013_verification/verification.json').read_text())
assert r['summary']['current_primary']['correct']==prior['heads']['original']['correct']
assert r['summary']['transition_adapted']['correct']==prior['heads']['adapted']['correct']
lines=['# Current-lineage component validation — 2026-10-13','','Fresh PyTorch inference on identical admitted validation inputs for every component. No training, protected-test access or paper changes. Accuracy is total correct divided by total evaluated.','','| Component | Citizen | SemLex | Local | Overall |','|---|---:|---:|---:|---:|']
for k,v in r['summary'].items():
 vals=[f"{100*d['correct'][k]/d['n']:.2f}%" for d in r['domains'].values()]
 lines.append('| '+k+' | '+' | '.join(vals)+f" | {v['accuracy_percent']:.2f}% ({v['correct']}/{v['n']}) |")
lines+=['','Standalone landmark/hand states exactly match pinned fusion parents. Current primary and adapted totals independently reproduce the prior full validation verification. Results are component-level; the adapted output alone is not the conditional runtime accuracy. PyTorch only here, not new phone or FP16 evaluation.','', 'Current lineage is rooted in the 96.83% part-wise/global base, not the historical full-roll checkpoint. Historical 95.50/80.69/96.30 numbers were Citizen-only and cannot be compared directly with the overall column.','', 'This review uses 953 admitted SemLex examples rather than the older fusion record’s 978. All components were rescored on the same reduced membership.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
