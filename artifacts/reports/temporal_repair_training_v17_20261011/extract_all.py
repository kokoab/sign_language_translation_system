from pathlib import Path
import json,subprocess,hashlib,time,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];P=O.parent/'temporal_repair_contract_v17_20261011'
# The only optimization skips stateless embeddings outside all contexts; confirm exact parity first.
a=json.loads((O/'parity.json').read_text());b=json.loads((P/'pilot_inputs_v2.json').read_text());rr=[r for r in b['rows'] if r['id']==a['rows'][0]['id']]
assert a['rows']==rr
for i,shape in enumerate(a['shapes']):
 x=np.fromfile(O/f'parity.{i}.bin',dtype='<f4');y=np.fromfile(P/f'pilot_inputs_v2.{i}.bin',dtype='<f4')[:len(x)];assert np.array_equal(x,y),(i,float(abs(x-y).max()))
(O/'extraction_parity.json').write_text(json.dumps({'rows':72,'all_four_arrays_bit_identical':True})+'\n')
cases=json.loads((O/'cases.json').read_text());out=O/'native_inputs';out.mkdir(exist_ok=True);start=time.time()
try:
 for index,case in enumerate(cases):
  prefix=out/case['id'];inp=out/(case['id']+'.case.json');inp.write_text(json.dumps([case])+'\n')
  if prefix.with_suffix('.json').exists():continue
  with (out/(case['id']+'.log')).open('w') as f:
   subprocess.run([str(O/'native_probe'),str(O.parent/'clean_lineage_recovery_v17_20261011/resources'),str(inp),str(prefix)],stdout=f,stderr=subprocess.STDOUT,check=True)
  payload=json.loads(prefix.with_suffix('.json').read_text());assert len(payload['rows'])==72*len(case['events'])
  print(json.dumps({'parents_complete':index+1,'parents_total':len(cases),'events':len(case['events']),'limit':case['limit'],'elapsed_seconds':time.time()-start}),flush=True)
 (O/'extraction_completion.json').write_text(json.dumps({'returncode':0,'parents':len(cases),'seconds':time.time()-start})+'\n')
except Exception as e:
 (O/'extraction_completion.json').write_text(json.dumps({'returncode':1,'error':repr(e)})+'\n');raise
finally:subprocess.run(['osascript','-e','display notification "Temporal input extraction finished; inspect completion status." with title "SLT experiment"'],check=False)
