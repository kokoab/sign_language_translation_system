from pathlib import Path
import sys,json
import numpy as np,torch
O=Path(__file__).resolve().parent;R=O.parents[2];P=O.parent/'lineage_runtime_investigation_v17_20261011'
sys.path.insert(0,str(R))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
torch.set_num_threads(2)
keys=json.load(open(P/'windows.json'))
xx=[torch.tensor(np.fromfile(P/f'windows.{i}.bin',np.float32).reshape((-1,)+shape)) for i,shape in enumerate([(32,61,5),(16,3,512),(16,3),(16,3,4)])];xx[2]=xx[2]>.5
with torch.inference_mode():
 m,c=load_model(O/'span_head_adaptation/best_model.pth');labels=sorted(c['label_to_index'],key=c['label_to_index'].get)
 pp=torch.cat([m(*[a[i:i+8] for a in xx]).softmax(-1) for i in range(0,len(keys),8)]).numpy()
 rows=[dict(k,top=labels[int(p.argmax())],my=float(p[labels.index('MY')]),like=float(p[labels.index('LIKE')])) for k,p in zip(keys,pp)]
 result={'rows':rows,'my_top':sum(r['top']=='MY' for r in rows),'n':len(rows),'interpretation':'Overlapping diagnostic windows, not independent accuracy or selection data.'}
 (O/'adapted_window_results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(result['my_top'],result['n'])
