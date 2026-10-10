"""Post-selection recognition of the two reviewed phone cores; no training or selection."""
import collections, json
import numpy as np
import torch
from features import O, BASE, encode
from train import predict
from active.v17.export_unified_multimodal_coreml_v17 import load_model

torch.set_num_threads(2)
prefix = O.parent/'temporal_repair_contract_v17_20261011/pilot_inputs_v2'
metadata = json.loads(prefix.with_suffix('.json').read_text())
rows = [r for r in metadata['rows'] if r['id'].startswith('phone_development:')]
indexes = [r['array_index'] for r in rows]
arrays = [np.fromfile(str(prefix)+f'.{i}.bin', np.float32).reshape(-1,*shape)[indexes] for i,shape in enumerate(metadata['shapes'])]
base, checkpoint = load_model(BASE)
device = torch.device('mps');base.to(device).eval()
features, parity = encode(base, arrays, device)
features = tuple(torch.from_numpy(a) for a in features)
result = {'note':'Known phone development cores, conservative reviewed boundaries with contextual margins. Correlated views, post-selection only; never training or independent accuracy.', 'encoder_parity':parity}
labels = sorted(checkpoint['label_to_index'],key=checkpoint['label_to_index'].get)
for name,path in [('incumbent',BASE),('selected',O/'selected.pth'),('diagnostic',O/'diagnostic.pth')]:
    model,_ = load_model(path)
    logits = predict(model.fusion_head,features)
    output=[]
    for i,row in enumerate(rows):
        label=row['id'].split(':')[1];pred=labels[logits[i].argmax().item()]
        other=logits[i].clone();other[labels.index(label)]=-torch.inf
        output.append(dict(id=row['id'],phase=row['phase'],offset=row['capture_offset'],view=row['view'],prediction=pred,margin=(logits[i,labels.index(label)]-other.max()).item()))
    result[name]={label:{'n':len(rr),'predictions':dict(collections.Counter(r['prediction'] for r in rr))} for label in ['MY','HAVE'] for rr in [[r for r in output if r['id'].split(':')[1]==label]]}
    (O/f'{name}_development_core_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
(O/'development_core_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
