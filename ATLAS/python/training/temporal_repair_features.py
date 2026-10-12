"""Cache complete frozen encoder outputs, never per-frame approximations."""
from pathlib import Path
import os,sys,json,hashlib,time
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.8')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.4')
import numpy as np,torch
O=(Path(__file__).resolve().parents[3] / 'artifacts/reports/temporal_repair_training_v17_20261011');R=O.parents[2];sys.path.insert(0,str(R))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
BASE=R/'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/best_model.pth'
RAW=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
KEYS=['landmarks','hand_embeddings','hand_valid','hand_boxes']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def encode(model,arrays,device):
 result=[[],[],[],[]];parity=0.
 with torch.inference_mode():
  for start in range(0,len(arrays[0]),64):
   x=[torch.as_tensor(np.asarray(a[start:start+64]),device=device).float() for a in arrays];x[2]=x[2]>.5
   ll,lf=model.landmark_model(x[0],return_embeddings=True);hf=model.hand_model.forward_features(*x[1:]);hl=model.hand_model.classifier(hf)
   z=(lf,hf,ll,hl)
   if start==0:
    delta=(model(*x)-model.fusion_head(*z)).abs().max().item();assert delta<1e-5;parity=max(parity,delta)
   for dest,value in zip(result,z):dest.append(value.cpu().numpy())
 return [np.concatenate(a) for a in result],parity
def main():
 torch.set_num_threads(2);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model,checkpoint=load_model(BASE);model.to(device).eval()
 out=O/'features';out.mkdir(exist_ok=True);pins={str(BASE):sha(BASE)};rows=[]
 previous=json.loads((O.parent/'clean_lineage_recovery_v17_20261011/letter_recipe.json').read_text());approved={r['path']:r['sha256'] for r in previous['pins']}
 for name in ['citizen_train','citizen_val','semlex_train','semlex_val','local_train','local_val']:
  path=RAW/(name+'.npz');digest=sha(path);assert approved[str(path)]==digest,'replay cache changed';pins[str(path)]=digest
  with np.load(path,allow_pickle=False) as data:
   arrays=[data[k] for k in KEYS];targets=data['targets'];ids=data['item_ids'];assert len(np.unique(ids))==len(ids)
   assert all(np.isfinite(a).all() for a in arrays)
   features,parity=encode(model,arrays,device)
  dest=out/(name+'.npz');np.savez(dest,**{f'f{i}':a for i,a in enumerate(features)},targets=targets,item_ids=ids)
  pins[str(dest)]=sha(dest);rows.append(dict(name=name,n=len(targets),classes=len(set(targets.tolist())),encoder_cache_logit_max_abs=parity));print(rows[-1],flush=True)
 (O/'isolated_feature_manifest.json').write_text(json.dumps(dict(base=str(BASE),label_to_index=checkpoint['label_to_index'],pins=pins,sets=rows,source='Exact existing admitted raw replay caches, hash-matched to prior recipe',device=str(device)),indent=2)+'\n')
if __name__=='__main__':main()
