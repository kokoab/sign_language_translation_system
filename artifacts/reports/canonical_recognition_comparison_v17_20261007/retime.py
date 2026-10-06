"""Repeat CPU timing without concurrent export/build work; preserve initial results."""
import json,sys,time,statistics
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.coreml_runtime_v17 import lightweight_imports
lightweight_imports()
import torch,numpy as np
from scripts.benchmark_stage1_families_v17 import build_model
from active.v17.model_v17 import SLTStage1V17,Stage1V17Config
from active.v17.export_unified_multimodal_coreml_v17 import load_model,load_pair,sha256_file
OUT=Path(__file__).resolve().parent
result=json.loads((OUT/'results.json').read_text())
torch.set_num_threads(1)
inputs=[]
for r in json.loads((OUT/'inputs.json').read_text()):
    arrays=load_pair(ROOT/r['landmarks'],ROOT/r['hand_features'])
    inputs.append(tuple(torch.from_numpy(v) if i!=2 else torch.from_numpy(v)>.5 for i,v in enumerate(arrays)))
models={}
for name,r in result['models'].items():
    path=ROOT/r['checkpoint'];assert sha256_file(path)==r['sha256']
    c=torch.load(path,map_location='cpu',weights_only=False)
    if r['kind']=='family':m=build_model('transformer',100);m.load_state_dict(c['state_dict'])
    elif r['kind']=='landmark':m=SLTStage1V17(Stage1V17Config(**c['model_config']));m.load_state_dict(c['model_state_dict'])
    else:m,_=load_model(path)
    models[name]=m.eval();r['timing_runs']=[]
names=list(models)
with torch.inference_mode():
    for order in [names,names[::-1],names[3:]+names[:3]]:
        for name in order:
            m=models[name];r=result['models'][name];multi=r['kind']=='multimodal'
            for v in inputs[:30]:m(*v) if multi else m(v[0])
            times=[];predictions=[]
            for v in inputs:
                t=time.perf_counter_ns();out=m(*v) if multi else m(v[0]);times.append((time.perf_counter_ns()-t)/1e6)
                predictions.append(int(out.argmax(1)))
            assert predictions==r['predictions']
            r['timing_runs'].append({'median_ms':statistics.median(times),'p90_ms':float(np.percentile(times,90)),'samples_ms':times})
            print(name,statistics.median(times),flush=True)
for r in result['models'].values():
    r['median_ms']=statistics.median(x['median_ms'] for x in r['timing_runs'])
    r['p90_ms']=statistics.median(x['p90_ms'] for x in r['timing_runs'])
result['protocol']['timing_run']='isolated rerun after all export/build/phone work; exploratory timing excluded'
(OUT/'verified_results.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPLETE',flush=True)
