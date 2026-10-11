from pathlib import Path
import sys,json,importlib.util
import numpy as np,torch
D=Path(__file__).resolve().parent;P=D.parent/'temporal_repair_training_v17_20261011';sys.path.insert(0,str(D.parents[2]));sys.path.insert(0,str(P))
from features import encode,sha,BASE
from active.v17.export_unified_multimodal_coreml_v17 import load_model
records=json.loads((D/'successor_dump.json').read_text())['records'];metadata=[];arrays=[]
for r in records:
 data=np.fromfile(D/'successor_windows'/(r['id']+'.bin'),dtype=np.float32).reshape(-1,34576)
 assert len(data)==len(r['dumped']) and np.isfinite(data).all()
 metadata.extend([dict(x,phase=r['id']) for x in r['dumped']]);arrays.append(data)
a=np.concatenate(arrays);shapes=[(32,61,5),(16,3,512),(16,3),(16,3,4)];offset=0;xx=[]
for shape in shapes:
 n=int(np.prod(shape));xx.append(a[:,offset:offset+n].reshape((-1,)+shape));offset+=n
assert offset==a.shape[1]
torch.set_num_threads(2);device=torch.device('mps');model,base=load_model(BASE);model.to(device).eval();f,delta=encode(model,xx,device)
with torch.inference_mode():pred=model.fusion_head(*(torch.tensor(x,device=device) for x in f)).cpu().numpy()
labels=sorted(base['label_to_index'],key=base['label_to_index'].get);targets=np.array([labels.index(x['label']) for x in metadata])
np.savez(D/'successor_features.npz',**{f'f{i}':x for i,x in enumerate(f)},targets=targets,item_ids=np.array([x['event'] for x in metadata]),baseline=pred)
(D/'successor_metadata.json').write_text(json.dumps(metadata,indent=2))
result={'n':len(metadata),'parity_max_abs':delta,'events':{event:{'n':sum(x['event']==event for x in metadata),'correct':int(sum(x['event']==event and pred[i].argmax()==targets[i] for i,x in enumerate(metadata)))} for event in sorted({x['event'] for x in metadata})},'feature_sha256':sha(D/'successor_features.npz')}
(D/'successor_feature_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
