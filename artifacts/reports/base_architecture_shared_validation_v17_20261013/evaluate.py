from pathlib import Path
import sys,json,hashlib,time
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from active.v17.coreml_runtime_v17 import lightweight_imports
lightweight_imports()
import numpy as np
import torch
from scripts.benchmark_stage1_families_v17 import build_model
from active.v17.model_stage1_family_v17 import build_stage1_landmark_model
O=Path(__file__).resolve().parent
paths={'Flat Transformer':'artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark/transformer/best_model.pth','Flat Squeezeformer':'artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth','Part-wise and global Squeezeformer':'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
roles=json.loads((R/'artifacts/reports/temporal_repair_training_v17_20261011/isolated_role_check.json').read_text())
reference=json.loads((R/'artifacts/reports/current_lineage_components_v17_20261013/results.json').read_text())
labels=json.loads((R/'active/v17/citizen100_manifest.json').read_text())
torch.set_num_threads(2);device=torch.device('mps');models={};results={'state':'running','models':{},'domains':{},'scope':'Pinned base checkpoints evaluated on same admitted validation membership as current-lineage component evaluation. No training/test/CPU speed measurement. Training histories are not matched.'}
for n,p in paths.items():
 c=torch.load(R/p,map_location='cpu',weights_only=False)
 if n=='Flat Transformer':m=build_model('transformer',100);m.load_state_dict(c['state_dict'],strict=True)
 else:m=build_stage1_landmark_model(c['model_config']);m.load_state_dict(c['model_state_dict'],strict=True)
 models[n]=m.eval().to(device);results['models'][n]={'path':p,'sha256':sha(R/p),'parameters':sum(p.numel() for p in m.parameters())}
start=time.monotonic()
for d in ['citizen','semlex','local']:
 p=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'/(d+'_val.npz')
 with np.load(p,allow_pickle=False) as z:
  ids=z['item_ids'].tolist();keep=np.array([i not in set(roles[d].get('excluded_validation_ids',[])) for i in ids]);x=z['landmarks'][keep];y=z['targets'][keep];ids=[id for id,k in zip(ids,keep) if k]
 assert ids==reference['domains'][d]['ids'] and y.tolist()==reference['domains'][d]['targets']
 rr={'n':len(y),'ids':ids,'targets':y.tolist(),'input_sha256':sha(p),'models':{}}
 for n,m in models.items():
  pp=[]
  with torch.inference_mode():
   for i in range(0,len(y),128):
    out=m(torch.from_numpy(x[i:i+128]).float().to(device));assert torch.isfinite(out).all();pp.extend(out.argmax(1).cpu().tolist())
  rr['models'][n]={'correct':int((np.array(pp)==y).sum()),'predictions':pp}
 results['domains'][d]=rr;(O/'results.json').write_text(json.dumps(results)+'\n');print(d,{k:v['correct'] for k,v in rr['models'].items()},flush=True)
for n,r in results['models'].items():
 r['correct']=sum(d['models'][n]['correct'] for d in results['domains'].values());r['n']=sum(d['n'] for d in results['domains'].values());r['accuracy']=100*r['correct']/r['n']
results['state']='complete';results['seconds']=time.monotonic()-start;(O/'results.json').write_text(json.dumps(results)+'\n');print(results['models'],flush=True)
