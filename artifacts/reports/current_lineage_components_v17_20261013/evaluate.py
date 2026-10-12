import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
O=Path(__file__).resolve().parent
V=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2'
paths={'fusion':V/'fusion/best_model.pth','phrase':V/'phrase_adapt/best_model.pth','interval':V/'span_recognizer/best_model.pth','current_primary':R/'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/best_model.pth','transition_adapted':R/'artifacts/reports/my_transition_recovery_v17_20261011/confidence_selected.pth'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
torch.set_num_threads(2);device=torch.device('mps');models={};meta={};labels=None
for n,p in paths.items():
 m,c=load_model(p)
 if labels is None:labels=c['label_to_index']
 assert c['label_to_index']==labels
 models[n]=m.eval().to(device);meta[n]=c
for part,key in [('landmark','landmark_model_state_dict'),('hand','hand_model_state_dict')]:
 src=meta['fusion']['source_checkpoints'][part];p=Path(src['path']);p=p if p.is_absolute() else R/p
 assert sha(p)==src['sha256']
 c=torch.load(p,map_location='cpu',weights_only=False)
 assert all(torch.equal(v,c['model_state_dict'][k]) for k,v in meta['fusion'][key].items()),part
for key in ['landmark_model_state_dict','hand_model_state_dict']:
 assert all(torch.equal(v,meta['transition_adapted'][key][k]) for k,v in meta['current_primary'][key].items())
result={'scope':'Fresh cached isolated validation; identical current admitted membership for every stage. Standalone source branches verified tensor-identical to frozen fusion encoders. Transition output alone is not conditional runtime accuracy. No training or test access.','models':{n:{'path':str(p),'sha256':sha(p)} for n,p in paths.items()},'standalone_sources':meta['fusion']['source_checkpoints'],'domains':{},'state':'running'}
start=time.monotonic()
for domain in ['citizen','semlex','local']:
 p=V/'raw_cache_citizen_variant'/(domain+'_val.npz')
 with np.load(p,allow_pickle=False) as z:
  a=[z[k] for k in ['landmarks','hand_embeddings','hand_valid','hand_boxes']];targets=z['targets'];ids=z['item_ids'].tolist()
 assert len(set(ids))==len(ids);n=len(targets)
 pred={k:[] for k in ['landmark_only','hand_only','fusion','phrase','interval','current_primary','transition_adapted']}
 with torch.inference_mode():
  for i in range(0,n,128):
   xx=[torch.as_tensor(x[i:i+128],device=device).float() for x in a];xx[2]=xx[2]>.5
   for name in ['fusion','phrase','interval','current_primary']:
    m=models[name];ll,lf=m.landmark_model(xx[0],return_embeddings=True);hf=m.hand_model.forward_features(*xx[1:]);hl=m.hand_model.classifier(hf);out=m.fusion_head(lf,hf,ll,hl)
    assert torch.isfinite(out).all();pred[name].extend(out.argmax(1).cpu().tolist())
    if name=='fusion':
     pred['landmark_only'].extend(ll.argmax(1).cpu().tolist());pred['hand_only'].extend(hl.argmax(1).cpu().tolist())
    if name=='current_primary':pred['transition_adapted'].extend(models['transition_adapted'].fusion_head(lf,hf,ll,hl).argmax(1).cpu().tolist())
 result['domains'][domain]={'n':n,'input_sha256':sha(p),'ids':ids,'targets':targets.tolist(),'predictions':pred,'correct':{k:int((np.array(v)==targets).sum()) for k,v in pred.items()}}
 (O/'results.json').write_text(json.dumps(result)+'\n');print(domain,result['domains'][domain]['correct'],n,'seconds',round(time.monotonic()-start,1),flush=True)
result['summary']={k:{'correct':sum(d['correct'][k] for d in result['domains'].values()),'n':sum(d['n'] for d in result['domains'].values())} for k in pred}
for v in result['summary'].values():v['accuracy_percent']=100*v['correct']/v['n']
result['state']='complete';result['inference_seconds']=time.monotonic()-start
(O/'results.json').write_text(json.dumps(result)+'\n');print(result['summary'],flush=True)
