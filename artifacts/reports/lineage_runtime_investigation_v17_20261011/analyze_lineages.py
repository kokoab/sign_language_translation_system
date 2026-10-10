from pathlib import Path
import json,sys,gc
import numpy as np,torch
R=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(R));O=Path(__file__).resolve().parent
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.model_v17 import SLTStage1V17,Stage1V17Config
D=R/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe'
C=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008'
G=R/'artifacts/reports/go_split_root_cause_v17_20261009'
B=R/'artifacts/reports/my_like_regression_v17_20261011'
samples=json.load(open(B/'dump.json.inputs.json'))
x=[torch.tensor(np.stack([np.array(r[k],np.float32).reshape(shape) for r in samples])) for k,shape in [('landmarks',(32,61,5)),('embeddings',(16,3,512)),('valid',(16,3)),('boxes',(16,3,4))]];x[2]=x[2]>.5
torch.set_num_threads(2)
paths={'old_phone':R/'artifacts/models/span_recognizer_v17_local_a/best_model.pth','old_clean':D/'august/span_recognizer/best_model.pth','old_clean_phrase':D/'august/phrase_adapt/best_model.pth','old_fusion':R/'artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth','new_before_filter':D/'chain_9683/span_recognizer/best_model.pth','v1':C/'run/span_recognizer/best_model.pth','v2':C/'run_v2/span_recognizer/best_model.pth','fragment':G/'run_frag_0.3/span_recognizer/best_model.pth','current':G/'run_frag25_reject/span_recognizer/best_model.pth'}
result={}
with torch.inference_mode():
 for name,path in paths.items():
  m,c=load_model(path);labels=sorted(c['label_to_index'],key=c['label_to_index'].get)
  ll,lf=m.landmark_model(x[0],return_embeddings=True);hf=m.hand_model.forward_features(*x[1:]);hl=m.hand_model.classifier(hf);fused=m.fusion_head(lf,hf,ll,hl)
  rows=[]
  for i,s in enumerate(samples):
   row={k:s[k] for k in ['start','end','pad']}
   for bn,l in [('landmark',ll),('hand',hl),('fused',fused)]:
    pp=l[i].softmax(-1).numpy();row[bn]={'top':labels[int(pp.argmax())],'my':float(pp[labels.index('MY')]),'like':float(pp[labels.index('LIKE')])}
   rows.append(row)
  result[name]={'path':str(path),'rows':rows};print(name,rows[1]['fused'],flush=True)
  del m,c;gc.collect()
(O/'lineage_spans.json').write_text(json.dumps(result,indent=2)+'\n')
# Count tensor equality in the two old-lineage recognizers, without mixing model parts.
a=torch.load(paths['old_phone'],map_location='cpu',weights_only=False);b=torch.load(paths['old_clean'],map_location='cpu',weights_only=False)
comparison={}
for group in ['landmark_model_state_dict','hand_model_state_dict','head_state_dict']:
 common=set(a[group])&set(b[group]);changed=[k for k in sorted(common) if not torch.equal(a[group][k],b[group][k])]
 comparison[group]={'same_keys':set(a[group])==set(b[group]),'tensors':len(common),'changed':changed}
(O/'old_clean_tensor_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
base=json.load(open(B/'rollback_review/base_comparison.json')); labels=base['old_phone']['validation']['citizen_val']['labels']
counts={name:sum(np.array(v['confusion']) for v in base[name]['validation'].values()) for name in ['old_phone','old_base_clean_recognizer','current']}
rows=[]
for i,label in enumerate(labels):
 row={'class':label,'n':int(counts['current'][i].sum())}
 for name,c in counts.items():row[name]=int(c[i,i])
 row['current_minus_old']=row['current']-row['old_phone'];rows.append(row)
(O/'class_retention.json').write_text(json.dumps(sorted(rows,key=lambda r:r['current_minus_old']),indent=2)+'\n')
print('class regressions',sorted(rows,key=lambda r:r['current_minus_old'])[:10],flush=True)
print('tensor differences',{k:(v['tensors'],len(v['changed'])) for k,v in comparison.items()},flush=True)
