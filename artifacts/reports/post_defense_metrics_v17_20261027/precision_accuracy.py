"""FP32 vs FP16 Core ML recognizer on the paper's 3,957 validation clips (same inputs/membership as calibration.py).

Both packages come from the same checkpoint and export script; only compute precision differs. Outputs:
original_word_logits = primary head (paper's 95.50%), word_logits = transition-adapted head. Mac Core ML, CPU_ONLY,
so the comparison isolates numerical precision from processor scheduling. Validation only.
"""
import sys, json
from pathlib import Path
import numpy as np
import coremltools as ct

O = Path(__file__).resolve().parent
R = O.parents[2]
SRC = json.loads((R / 'artifacts/reports/current_lineage_components_v17_20261013/results.json').read_text())
CACHE = R / 'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
PACKAGES = {'fp16': R / 'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/Models/SpanRecognizerV17ConfidenceDualB8FP16.mlpackage',
            'fp32': O / 'fp32_models/SpanRecognizerV17ConfidenceDualB8FP32.mlpackage'}
KEYS = ['landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes']

inputs, targets = {k: [] for k in KEYS}, []
for domain in ['citizen', 'semlex', 'local']:
    with np.load(CACHE / (domain + '_val.npz'), allow_pickle=False) as z:
        pos = {k: i for i, k in enumerate(z['item_ids'].tolist())}
        keep = np.array([pos[k] for k in SRC['domains'][domain]['ids']])
        for k in KEYS:
            inputs[k].append(z[k][keep].astype(np.float32))
        targets.append(z['targets'][keep])
inputs = {k: np.concatenate(v) for k, v in inputs.items()}
targets = np.concatenate(targets)
n = len(targets)
calib = np.load(O / 'calibration_logits.npz')
assert (calib['targets'] == targets).all()

out = {}
for arm, pkg in PACKAGES.items():
    model = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    logits = {'original_word_logits': [], 'word_logits': []}
    for i in range(0, n, 8):
        feed = {}
        for k in KEYS:
            a = inputs[k][i:i + 8]
            if len(a) < 8:
                a = np.concatenate([a, np.repeat(a[-1:], 8 - len(a), axis=0)])
            feed[k] = a
        pred = model.predict(feed)
        for k in logits:
            logits[k].append(np.asarray(pred[k], dtype=np.float64)[:min(8, n - i)])
    out[arm] = {k: np.concatenate(v) for k, v in logits.items()}
    print(arm, {k: int((v.argmax(1) == targets).sum()) for k, v in out[arm].items()}, flush=True)

pytorch_primary = calib['logits'].astype(np.float64)
result = dict(scope='Mac Core ML CPU_ONLY, 3,957 admitted validation clips; same checkpoint, only compute precision differs.',
              packages={k: str(v.relative_to(R)) for k, v in PACKAGES.items()}, n=n, heads={})
for head in ['original_word_logits', 'word_logits']:
    a, b = out['fp32'][head], out['fp16'][head]
    result['heads'][head] = dict(
        fp32_correct=int((a.argmax(1) == targets).sum()), fp16_correct=int((b.argmax(1) == targets).sum()),
        fp32_accuracy=float((a.argmax(1) == targets).mean()), fp16_accuracy=float((b.argmax(1) == targets).mean()),
        top1_agreement=int((a.argmax(1) == b.argmax(1)).sum()), max_abs_logit_diff=float(np.abs(a - b).max()),
        mean_abs_logit_diff=float(np.abs(a - b).mean()))
result['heads']['original_word_logits']['fp32_vs_pytorch_top1_agreement'] = int((out['fp32']['original_word_logits'].argmax(1) == pytorch_primary.argmax(1)).sum())
result['heads']['original_word_logits']['fp16_vs_pytorch_top1_agreement'] = int((out['fp16']['original_word_logits'].argmax(1) == pytorch_primary.argmax(1)).sum())
(O / 'precision_accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['heads'], indent=1))
