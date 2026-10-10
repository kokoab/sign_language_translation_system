from pathlib import Path
import json,shutil
O=Path(__file__).resolve().parent
assert json.loads((O/'exact_non_o5_completion.json').read_text())['returncode']==0
cases=json.loads((O/'exact_cases.json').read_text());count=0
for c in cases:
 if c['source']=='o5s5':
  original=next(r for r in json.loads((O/'cases.json').read_text()) if r['id']==c['id']);assert original['events']==c['events']
  for suffix in ['.json']+[f'.{i}.bin' for i in range(4)]:
   p=O/'native_inputs'/(c['id']+suffix);q=O/'exact_native_inputs'/p.name
   if not q.exists():
    assert p.exists(),p
    q.symlink_to(p.resolve())
 p=O/'exact_native_inputs'/(c['id']+'.json');d=json.loads(p.read_text());assert len(d['rows'])==len(c['events'])*72;count+=len(c['events'])
assert count==144
(O/'exact_extraction_completion.json').write_text(json.dumps({'returncode':0,'parents':len(cases),'events':count,'requested_views':count*72,'boundaries':'exact rational source-frame PTS for132events; unchanged annotated times for12O5S5events'})+'\n');print('All exact native inputs available:',count)
