from pathlib import Path
import json,subprocess,time
O=Path(__file__).resolve().parent
out=O/'exact_native_inputs';out.mkdir(exist_ok=True);cases=[c for c in json.loads((O/'exact_cases.json').read_text()) if c['source']!='o5s5'];start=time.time()
for i,c in enumerate(cases):
 prefix=out/c['id'];inp=out/(c['id']+'.case.json');inp.write_text(json.dumps([c])+'\n')
 if prefix.with_suffix('.json').exists():continue
 with (out/(c['id']+'.log')).open('w') as f:subprocess.run([str(O/'cached_fast_probe'),str(O.parent/'clean_lineage_recovery_v17_20261011/resources'),str(inp),str(prefix)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(json.dumps({'parents_complete':i+1,'total':len(cases),'seconds':time.time()-start}),flush=True)
(O/'exact_non_o5_completion.json').write_text(json.dumps({'returncode':0,'parents':len(cases),'seconds':time.time()-start})+'\n')
