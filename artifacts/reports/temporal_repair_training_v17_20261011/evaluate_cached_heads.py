"""Post-selection train/development diagnostics; never used for checkpoint selection."""
import json
import numpy as np
import torch
from features import O, BASE
from train import read, predict
from active.v17.export_unified_multimodal_coreml_v17 import load_model

torch.set_num_threads(2)
result = {}
for name, path in [('incumbent', BASE), ('selected', O/'selected.pth'), ('diagnostic', O/'diagnostic.pth')]:
    model, checkpoint = load_model(path)
    labels = sorted(checkpoint['label_to_index'], key=checkpoint['label_to_index'].get)
    arm = {}
    for dataset in ['temporal_train', 'temporal_validation']:
        x, target, ids = read(dataset)
        logits = predict(model.fusion_head, x)
        prediction = logits.argmax(1)
        events = []
        for ident in dict.fromkeys(ids.tolist()):
            ix = np.flatnonzero(ids == ident)
            correct = int((prediction[ix] == target[ix]).sum())
            other = logits[ix].clone(); other[torch.arange(len(ix)), target[ix]] = -torch.inf
            margin = logits[ix, target[ix]] - other.max(1).values
            events.append({'id': ident, 'label': labels[target[ix[0]].item()], 'views': len(ix), 'correct': correct, 'minimum_margin': margin.min().item(), 'median_margin': margin.median().item()})
        arm[dataset] = {label: {'events': len(rr), 'all_views_correct_events': sum(r['correct'] == r['views'] for r in rr), 'mean_event_view_accuracy': sum(r['correct']/r['views'] for r in rr)/len(rr)} for label in sorted({r['label'] for r in events}) for rr in [[r for r in events if r['label'] == label]]}
        (O/f'{name}_{dataset}_head_diagnostics.json').write_text(json.dumps(events, indent=2)+'\n')
    source = O.parent/'lineage_runtime_investigation_v17_20261011'
    shapes = [(32,61,5), (16,3,512), (16,3), (16,3,4)]
    inputs = [torch.from_numpy(np.fromfile(source/f'windows.{i}.bin', np.float32).reshape((-1,)+shape)) for i, shape in enumerate(shapes)]
    inputs[2] = inputs[2] > .5
    with torch.inference_mode():
        logits = model(*inputs)
    words = [labels[i] for i in logits.argmax(1).tolist()]
    arm['same_phone_62_windows'] = {'n': len(words), 'top1_counts': {label: words.count(label) for label in sorted(set(words))}, 'note': 'Overlapping model-input windows from the same known phone failure; not 62 independent signs or an accuracy estimate.'}
    result[name] = arm
(O/'cached_head_summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
