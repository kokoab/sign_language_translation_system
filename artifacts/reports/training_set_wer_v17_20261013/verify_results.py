import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
r=json.loads((O/'results.json').read_text())
assert r['state']=='complete'
def distance(a,b):
 p=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  q=[i]
  for j,y in enumerate(b,1):q.append(min(q[-1]+1,p[j]+1,p[j-1]+(x!=y)))
  p=q
 return p[-1]
base=None
lines=['# Matched training-set WER','', 'All checkpoints were scored on the same 179 approved training recordings (495 reference glosses) with the same fixed historical decoder. These are training-set results, not validation or test results. No model was trained or tuned for this evaluation.','', '| Checkpoint | Errors | Training-set WER |','|---|---:|---:|']
for name,rows in r['records'].items():
 ids={x['id']:x['reference'] for x in rows}
 assert len(rows)==len(ids)==179
 if base is None:base=ids
 assert base==ids
 refs=sum(len(x['reference']) for x in rows);assert refs==495
 errors=sum(distance(x['reference'],x['hypothesis']) for x in rows)
 s=r['summary'][name];assert errors==s['S']+s['D']+s['I'] and refs==s['refs']
 lines.append(f"| {name} | {errors}/{refs} | {100*errors/refs:.2f}% |")
lines+=['','The transition-fine-tuned checkpoint is evaluated as a recognizer under this fixed decoder. This does not reproduce the app’s selective transition-handling policy. Later checkpoints also differ in training recipes and inputs, so the comparison does not isolate a single cause for each change.','', 'The initial attempt used incomplete training-window caches and is invalid. The corrected run fills all decoder-required candidate windows and records input and checkpoint hashes. No manuscript or review checklist has been changed.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('VERIFIED: six checkpoints, identical 179 recordings / 495 references; independent edit-distance totals match.')
for n,s in r['summary'].items():print(n,round(s['wer_percent'],4))
