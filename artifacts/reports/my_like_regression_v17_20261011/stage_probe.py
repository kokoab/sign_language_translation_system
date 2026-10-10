"""Read-only checkpoint/stage comparison on native-prepared phone span inputs."""
import json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
HERE=Path(__file__).resolve().parent
torch.set_num_threads(2)
samples=json.loads((HERE/'dump.json.inputs.json').read_text())
x=[torch.tensor(np.stack([np.array(r[k],np.float32).reshape(shape) for r in samples])) for k,shape in [('landmarks',(32,61,5)),('embeddings',(16,3,512)),('valid',(16,3)),('boxes',(16,3,4))]];x[2]=x[2]>.5
C=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run'
B=ROOT/'artifacts/reports/canonical_recognition_comparison_v17_20261007'
paths={'old_phone':ROOT/'artifacts/models/span_recognizer_v17_local_a/best_model.pth',
'new_fusion':C/'fusion/best_model.pth','new_phrase':C/'phrase_adapt/best_model.pth','new_recognizer':C/'span_recognizer/best_model.pth',
'pre_filter_fusion':B/'chain_9683_floor361_mildroll/fusion/best_model.pth','pre_filter_phrase':B/'downstream_recipe/chain_9683/phrase_adapt/best_model.pth','pre_filter_recognizer':B/'downstream_recipe/chain_9683/span_recognizer/best_model.pth'}
results={}
with torch.inference_mode():
 for name,path in paths.items():
  m,c=load_model(path);labels=sorted(c['label_to_index'],key=c['label_to_index'].get)
  ll,lf=m.landmark_model(x[0],return_embeddings=True)
  hf=m.hand_model.forward_features(*x[1:]);hl=m.hand_model.classifier(hf)
  fused=m.fusion_head(lf,hf,ll,hl)
  rows=[]
  for i,s in enumerate(samples):
   r={k:s[k] for k in ['start','end','pad']}
   for branch,logits in [('landmark',ll),('hand',hl),('fused',fused)]:
    pp=logits[i].softmax(-1).numpy();r[branch]={'top':labels[int(pp.argmax())],'my':float(pp[labels.index('MY')]),'like':float(pp[labels.index('LIKE')])}
   rows.append(r)
  results[name]=rows;print(name,rows,flush=True)
  (HERE/'stage_scores.json').write_text(json.dumps(results,indent=2))
