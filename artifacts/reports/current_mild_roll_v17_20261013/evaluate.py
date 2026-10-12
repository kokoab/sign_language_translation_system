import sys,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from active.v17.coreml_runtime_v17 import lightweight_imports
lightweight_imports()
from active.v17.model_stage1_family_v17 import build_stage1_landmark_model
from active.v17.train_stage_1_v17 import rotate_camera_roll_v17
O=Path(__file__).resolve().parent;torch.set_num_threads(2);device='mps'
paths={'base':'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth','adapted':'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/local_replay/best_promotion_gate_model.pth'}
models={};out={'scope':'Fresh software-rotated landmark validation, not rotated camera images or isolated causal augmentation ablation. Same current admitted membership; no training/test access.','angles':[-12,-6,0,6,12],'models':{},'domains':{},'state':'running'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for name,path in paths.items():
 c=torch.load(R/path,map_location='cpu',weights_only=False);m=build_stage1_landmark_model(c['model_config']);m.load_state_dict(c['model_state_dict']);models[name]=m.eval().to(device);out['models'][name]={'path':path,'sha256':sha(R/path)}
roles=json.loads((R/'artifacts/reports/temporal_repair_training_v17_20261011/isolated_role_check.json').read_text())
for domain in ['citizen','semlex','local']:
 p=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'/(domain+'_val.npz')
 with np.load(p,allow_pickle=False) as z:
  ids=z['item_ids'].tolist();excluded=set(roles[domain].get('excluded_validation_ids',[]));keep=np.array([id not in excluded for id in ids]);x=z['landmarks'][keep];y=z['targets'][keep]
 d={'n':len(y),'input_sha256':sha(p),'correct':{name:{} for name in models}}
 with torch.inference_mode():
  for name,m in models.items():
   for angle in out['angles']:
    correct=0
    for i in range(0,len(y),128):
     xx=torch.as_tensor(x[i:i+128],device=device).float();xx=rotate_camera_roll_v17(xx,math.radians(angle));logits=m(xx);assert torch.isfinite(logits).all();correct+=int((logits.argmax(1).cpu().numpy()==y[i:i+128]).sum())
    d['correct'][name][str(angle)]=correct
 out['domains'][domain]=d;(O/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(domain,d['correct'],flush=True)
out['summary']={name:{str(a):sum(d['correct'][name][str(a)] for d in out['domains'].values()) for a in out['angles']} for name in models};out['n']=sum(d['n'] for d in out['domains'].values());out['state']='complete';(O/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['summary'],flush=True)
