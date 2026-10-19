"""Compare actual-video CoreML primary outputs with the pinned PyTorch primary."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model

OUT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/best_model.pth'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


torch.set_num_threads(2)
model, metadata = load_model(CHECKPOINT)
label_to_index = metadata['label_to_index']
labels = [None] * len(label_to_index)
for label, i in label_to_index.items():
    labels[i] = label
assert labels == json.loads((ROOT / 'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/Models/live_reel_labels.json').read_text())
prior = json.loads((ROOT / 'artifacts/reports/current_lineage_components_v17_20261013/results.json').read_text())['models']['current_primary']
assert sha(CHECKPOINT) == prior['sha256']
manifest = json.loads((ROOT / 'artifacts/reports/model_class_examples_20261019/manifest.json').read_text())
sources = {r['label']: r for r in manifest['examples']}
native = json.loads((OUT / 'primary_native.json').read_text())['records']
assert len(native) == 100
shapes = {'landmarks': (1, 32, 61, 5), 'hand_embeddings': (1, 16, 3, 512), 'hand_valid': (1, 16, 3), 'hand_boxes': (1, 16, 3, 4)}
rows = []
with torch.inference_mode():
    for r in native:
        label = r['id']
        s = sources[label]
        path = ROOT / 'data/local/model_class_examples' / s['filename']
        assert Path(r['path']) == path
        assert sha(path) == s['source_sha256'] == sha(ROOT / s['source_video'])
        row = dict(label=label, filename=s['filename'], source_sha256=s['source_sha256'], train=True)
        if 'error' in r:
            rows.append(dict(row, error=r['error']))
            continue
        xx = [torch.from_numpy(np.fromfile(OUT / 'inputs' / (label + '_' + key + '.f32'), dtype=np.float32).reshape(shape)) for key, shape in shapes.items()]
        logits = model(xx[0], xx[1], xx[2] > .5, xx[3])[0]
        assert torch.isfinite(logits).all()
        probs = logits.softmax(-1)
        top = probs.topk(5)
        row['pytorch_top5'] = [dict(label=labels[i], probability=float(p), logit=float(logits[i])) for p, i in zip(top.values.tolist(), top.indices.tolist())]
        row['coreml_primary_top5'] = r['primary_top5']
        row['coreml_primary'] = r['primary_top5'][0]['label']
        row['pytorch_primary'] = row['pytorch_top5'][0]['label']
        row['coreml_correct'] = row['coreml_primary'] == label
        row['pytorch_correct'] = row['pytorch_primary'] == label
        row['agreement'] = row['coreml_primary'] == row['pytorch_primary']
        row['max_coreml_top5_logit_delta'] = max(abs(float(logits[label_to_index[v['label']]]) - v['logit']) for v in r['primary_top5'])
        rows.append(row)
summary = dict(total=len(rows), coreml_correct=sum(r.get('coreml_correct', False) for r in rows), pytorch_correct=sum(r.get('pytorch_correct', False) for r in rows), top1_agreement=sum(r.get('agreement', False) for r in rows), failures=[r['label'] for r in rows if not r.get('coreml_correct') or not r.get('pytorch_correct')])
result = dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=sha(CHECKPOINT), scope='Full copied-video native pixels; primary output only, no boundaries, expected-label selection or MY repair; PyTorch uses identical fresh native tensors.', summary=summary, records=rows)
(OUT / 'primary_verified.json').write_text(json.dumps(result, indent=2) + '\n')
with (OUT / 'per_video.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['class', 'file', 'CoreML primary', 'CoreML score', 'PyTorch primary', 'PyTorch score', 'both correct', 'sha256'])
    for r in rows:
        writer.writerow([r['label'], r['filename'], r.get('coreml_primary'), r.get('coreml_primary_top5', [{}])[0].get('probability'), r.get('pytorch_primary'), r.get('pytorch_top5', [{}])[0].get('probability'), r.get('coreml_correct', False) and r.get('pytorch_correct', False), r['source_sha256']])
print(json.dumps(summary), flush=True)
