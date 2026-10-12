"""Read-only checkpoint swap diagnostic on existing isolated validation inputs."""
import copy,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.train_unfrozen_phrase_adapt_v17 import tensors
family=sys.argv[1];prefix='mp_' if family=='mediapipe' else '';suffix='_20261004' if family=='mediapipe' else ''
D=R/'artifacts/models'
O=Path(__file__).resolve().parent
torch.set_num_threads(2);device=torch.device('mps')
a,ac=load_model(D/(prefix+'stage1_v17_unified_phrase_activity_adapt_reel_v2'+suffix)/'best_model.pth');b,bc=load_model(D/(prefix+'span_recognizer_v17_local_a'+suffix)/'best_model.pth')
labels=sorted(ac['label_to_index'],key=ac['label_to_index'].get);assert ac['label_to_index']==bc['label_to_index']
models={'before':a,'after':b,'head_only':copy.deepcopy(a),'encoders_only':copy.deepcopy(b)}
models['head_only'].fusion_head.load_state_dict(b.fusion_head.state_dict());models['encoders_only'].fusion_head.load_state_dict(a.fusion_head.state_dict())
changes={}
for key in ['landmark_model_state_dict','hand_model_state_dict','head_state_dict']:
 changes[key]={'changed_tensors':sum(not torch.equal(v,bc[key][k]) for k,v in ac[key].items()),'total_tensors':len(ac[key])}
for m in models.values():m.eval().to(device)
result={'scope':'Checkpoint swaps diagnose parameter changes; not independently trained ablations or deployment candidates. Validation only.','weight_changes':changes,'domains':{}}
for domain,n in [('citizen',378)]:
 path=R/'artifacts/generated'/('mp_unfrozen_phrase_adapt_v17_20261004' if family=='mediapipe' else 'unfrozen_phrase_adapt_v17')/f'{domain}_val.npz'
 with np.load(path,allow_pickle=False) as z:
  vals=tensors({k:z[k] for k in z.files});ids=z['item_ids'].tolist() if 'item_ids' in z else list(range(n))
 target=vals[4].numpy();assert len(target)==n;pred={}
 with torch.inference_mode():
  for name,m in models.items():
   pp=[]
   for i in range(0,n,32):
    x=[v[i:i+32].to(device) for v in vals[:4]];x[1]=x[1].float();out=m(*x);assert torch.isfinite(out).all();pp.extend(out.argmax(1).cpu().tolist())
   pred[name]=np.array(pp)
 lost=np.flatnonzero((pred['before']==target)&(pred['after']!=target));gained=np.flatnonzero((pred['before']!=target)&(pred['after']==target))
 result['domains'][domain]={'n':n,'correct':{k:int((v==target).sum()) for k,v in pred.items()},'lost':len(lost),'gained':len(gained),'changed':int((pred['before']!=pred['after']).sum()),'gained_examples':[{'id':ids[i],'target':labels[target[i]],'before':labels[pred['before'][i]]} for i in gained],'lost_examples':[{'id':ids[i],'target':labels[target[i]],'after':labels[pred['after'][i]],'head_only':labels[pred['head_only'][i]],'encoders_only':labels[pred['encoders_only'][i]]} for i in lost]}
 print(domain,{k:v for k,v in result['domains'][domain].items() if k!='lost_examples'},flush=True)
result['aggregate']={k:sum(d['correct'][k] for d in result['domains'].values()) for k in models}
(O/(family+'_results.json')).write_text(json.dumps(result,indent=2)+'\n')
