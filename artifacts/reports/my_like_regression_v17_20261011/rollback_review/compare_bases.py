"""Read-only matched checkpoints on phone contacts and isolated validation; no training/test."""
from pathlib import Path
import sys,json,gc
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
HERE=Path(__file__).resolve().parent;torch.set_num_threads(2);device=torch.device('mps')
D=ROOT/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe'
C=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008'
G=ROOT/'artifacts/reports/go_split_root_cause_v17_20261009'
paths={'old_phone':ROOT/'artifacts/models/span_recognizer_v17_local_a/best_model.pth',
'old_base_fusion':ROOT/'artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth',
'old_base_clean_phrase':D/'august/phrase_adapt/best_model.pth','old_base_clean_recognizer':D/'august/span_recognizer/best_model.pth',
'v2':C/'run_v2/span_recognizer/best_model.pth','fragment':G/'run_frag_0.3/span_recognizer/best_model.pth',
'current':G/'run_frag25_reject/span_recognizer/best_model.pth'}
samples=json.loads((HERE/'contacts.inputs.json').read_text())
phone=[np.stack([np.array(r[k],np.float32).reshape(shape) for r in samples]) for k,shape in [('landmarks',(32,61,5)),('embeddings',(16,3,512)),('valid',(16,3)),('boxes',(16,3,4))]]
keys=['landmarks','hand_embeddings','hand_valid','hand_boxes'];cache=C/'run/raw_cache_citizen_variant'
results={}
def score(m,xx):
 output=[]
 for start in range(0,len(xx[0]),32):
  batch=[torch.from_numpy(a[start:start+32].astype(np.float32)).to(device) for a in xx];batch[2]=batch[2]>.5
  with torch.inference_mode():out=m(*batch)
  output.append(out.cpu().numpy())
 return np.concatenate(output)
for name,path in paths.items():
 m,meta=load_model(path);m.to(device).eval();labels=sorted(meta['label_to_index'],key=meta['label_to_index'].get)
 def rows(logits):
  pp=torch.from_numpy(logits).softmax(-1).numpy()
  return [{'top':labels[int(r.argmax())],'my':float(r[labels.index('MY')]),'like':float(r[labels.index('LIKE')])} for r in pp]
 result={'checkpoint':str(path.relative_to(ROOT)),'phone':[dict(id=s['id'],**r) for s,r in zip(samples,rows(score(m,phone)))],'validation':{}}
 print(name,'phone',[r['top'] for r in result['phone']],flush=True)
 for split in ['citizen_val','semlex_val','local_val']:
  with np.load(cache/(split+'.npz'),allow_pickle=False) as data:
   yy=data['targets'];xx=[data[k] for k in keys];ids=data['item_ids']
  logits=score(m,xx);pred=logits.argmax(-1);cm=np.zeros((100,100),int);np.add.at(cm,(yy,pred),1)
  result['validation'][split]={'n':len(yy),'correct':int((pred==yy).sum()),'confusion':cm.tolist(),'labels':labels}
  print(name,split,result['validation'][split]['correct'],len(yy),flush=True)
  np.savez_compressed(HERE/(name+'_'+split+'_scores.npz'),logits=logits,targets=yy,item_ids=ids)
  del xx,logits
 results[name]=result;(HERE/'base_comparison.json').write_text(json.dumps(results))
 m.cpu();del m;gc.collect();torch.mps.empty_cache()
