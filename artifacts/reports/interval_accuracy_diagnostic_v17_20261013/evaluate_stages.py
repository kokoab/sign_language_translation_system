"""One evaluator for every stage checkpoint: 4,252 isolated validation clips plus tune-pool WER.

Isolated inputs: artifacts/generated/unfrozen_phrase_adapt_v17/{citizen,semlex,local}_val.npz (local
mouth-masked).  Landmark-only stages see only the landmark tensor.  Development WER: the interval
experiment's decoder and cfg on the 89-clip lab tune pool (scripts/train_span_recognizer_v17.tuning_wer);
landmark-only stages are decoded through LandmarkOnlySpan (hand inputs ignored).  The lab held-out test is
never loaded.  CPU, validation/development only.  usage: evaluate_stages.py SPEC.json OUT.json
SPEC: [{"chain":..., "stage":..., "path":..., "kind":"landmark"|"unified"}, ...]
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
R = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R))
from active.v17.export_unified_multimodal_coreml_v17 import load_model, sha256_file
from active.v17.model_stage1_family_v17 import build_stage1_landmark_model
from active.v17.train_unfrozen_phrase_adapt_v17 import tensors
import scripts.train_span_recognizer_v17 as tsr
from scripts import segmental_lab_v17 as lab

CFG = json.loads('{"lookahead":8,"span_mode":"both","alpha":0.7,"w_r":2,"w_a":0.25,"w_b":0.5,"c":-1,"log_theta":-1.6094}')
torch.set_num_threads(4)
CPU = torch.device('cpu')


def load(row):
    if row['kind'] == 'unified':
        return load_model(R / row['path'])
    ck = torch.load(R / row['path'], map_location='cpu', weights_only=False)
    net = build_stage1_landmark_model(ck['model_config']); net.load_state_dict(ck['model_state_dict'], strict=True)
    return tsr.LandmarkOnlySpan(net.eval()).eval(), ck


@torch.inference_mode()
def predict(model, values):
    out = []
    for i in range(0, len(values[4]), 128):
        x = [v[i:i + 128] for v in values[:4]]; x[1] = x[1].float(); out.append(model(*x))
    return torch.cat(out)


def main(spec_path, out_path):
    spec = json.loads(Path(spec_path).read_text())
    raw = {}
    for d in ('citizen', 'semlex', 'local'):
        with np.load(R / f'artifacts/generated/unfrozen_phrase_adapt_v17/{d}_val.npz', allow_pickle=False) as z:
            raw[d] = tensors({k: z[k] for k in z.files})
    tune = [(r, lab.load_dgs(r)[0], lab.load_memo(r)) for r in lab.rows_for('tune')]
    out = Path(out_path); results = json.loads(out.read_text()) if out.is_file() else {}
    for row in spec:
        key = f"{row['chain']}/{row['stage']}"
        digest = sha256_file(R / row['path'])
        if key in results and results[key]['sha256'] == digest:
            continue
        model, ck = load(row)
        labels = [l for l, _ in sorted(ck['label_to_index'].items(), key=lambda r: r[1])]
        domains, preds = {}, {}
        for d, v in raw.items():
            logits = predict(model, v); p = logits.argmax(1)
            domains[d] = {'n': len(p), 'correct': int((p == v[4]).sum()),
                          'top5': int((logits.topk(5, 1).indices == v[4][:, None]).any(1).sum())}
            preds[d] = p.tolist()
        wer = tsr.tuning_wer(model, tune, labels, CFG, CPU)
        results[key] = {**row, 'sha256': digest, 'domains': domains,
                        'pooled_correct': sum(x['correct'] for x in domains.values()), 'pooled_n': 4252,
                        'tune': {k: wer[k] for k in ('wer', 'correct', 'S', 'D', 'I', 'refs', 'hyps')},
                        'predictions': preds}
        out.write_text(json.dumps(results) + '\n')
        print(key, results[key]['domains'], results[key]['pooled_correct'], 'WER %.4f' % wer['wer'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
