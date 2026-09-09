import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.model_stage2_v17 import load_stage2_model_v17
from active.v17.train_stage_2_other_ctc_v17 import CombinedDataset, _metric_summary, collapse_ctc
from active.v17.train_stage_2_v17 import RealPhraseDataset, collate
from active.v17.train_stage_2_accuracy_repair_v17 import IsolatedPoolDataset

torch.set_num_threads(2)
report = json.loads((ROOT/'artifacts/reports/stage2_v17_transition_adapt_v2/training_result.json').read_text())
out = ROOT/'artifacts/reports/stage2_v17_transition_diagnosis_v1'
from active.v17.train_stage_2_other_ctc_v17 import sha256
datasets = {
 'train': [RealPhraseDataset(ROOT/'data/local/stage2_v17_frozen_features','train'), RealPhraseDataset(ROOT/'data/local/stage2_v17_asllrp_other_frozen_features','train'), IsolatedPoolDataset(ROOT/'data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz','isolated_citizen_train',augment_boundaries=False),RealPhraseDataset(ROOT/'data/local/stage2_v17_transition_adapt_v2/frozen_features','train'),RealPhraseDataset(ROOT/'data/local/stage2_v17_asllrp_segmented_train_frozen_features','train')],
 'validation': [RealPhraseDataset(ROOT/'data/local/stage2_v17_frozen_features','validation'), RealPhraseDataset(ROOT/'data/local/stage2_v17_asllrp_other_frozen_features','validation'), IsolatedPoolDataset(ROOT/'data/local/stage2_v17_isolated_replay/citizen_validation.npz','isolated_citizen_validation',augment_boundaries=False),RealPhraseDataset(ROOT/'data/local/stage2_v17_transition_adapt_v2/frozen_features','validation'),RealPhraseDataset(ROOT/'data/local/stage2_v17_asllrp_segmented_validation_frozen_features','validation')]
}
target=ROOT/'artifacts/models/stage2_v17_other_preservation_v1';target.mkdir(exist_ok=True)
for role,children in datasets.items():
 dataset=CombinedDataset(children);loader=DataLoader(dataset,batch_size=16,shuffle=False,collate_fn=collate)
 rows=[dict(item_id=dataset[i].item_id,source=dataset[i].source,reference=dataset[i].targets.tolist(),windows=len(dataset[i].features)) for i in range(len(dataset))]
 paths={'accepted':report['teacher']};paths.update({str(r['seed']):r['checkpoint'] for r in report['results'] if r['arm']=='with_stem'})
 result={'rows':rows,'role':role,'models':{}}
 for name,path in paths.items():
  model,_=load_stage2_model_v17(ROOT/path);model.eval();values=[]
  with torch.inference_mode():
   for b in loader:
    if name=='accepted':features,lengths=model(b['features'],b['window_mask'])
    else:features,lengths=model.encode(b['features'],b['window_mask'])
    for x,n in zip(features,lengths.tolist()):values.append(x[:n].clone())
  result['models'][name]={'values':values,'path':path,'sha256':sha256(ROOT/path)}
  print(role,name,len(values),flush=True)
 torch.save(result,target/f'{role}_cache.pt')
