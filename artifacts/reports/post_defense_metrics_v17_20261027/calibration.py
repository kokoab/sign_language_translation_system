"""Posterior calibration of the primary recognizer on the paper's 3,957 validation clips.

Recomputes logits with the same checkpoint, inputs and exclusions as
current_lineage_components_v17_20261013 (which saved only argmax), then reports
NLL, Brier, ECE/MCE (15 equal-width bins) and a reliability diagram, before and
after temperature scaling. Temperature is cross-fitted: fit on one half, scored on
the other, then swapped, so the "after" numbers never use their own fitting data.
Validation only; no training of the recognizer and no protected-test access.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch

O = Path(__file__).resolve().parent
R = O.parents[2]
sys.path.insert(0, str(R))
from active.v17.export_unified_multimodal_coreml_v17 import load_model

SOURCE = R / 'artifacts/reports/current_lineage_components_v17_20261013/results.json'
CACHE = R / 'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
BINS = 15


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def metrics(logits, targets, t=1.0):
    z = logits / t
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    hit = (pred == targets).astype(float)
    onehot = np.eye(p.shape[1])[targets]
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, BINS - 1)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(BINS):
        m = idx == b
        if not m.any():
            bins.append(dict(low=edges[b], high=edges[b + 1], n=0))
            continue
        gap = abs(hit[m].mean() - conf[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
        bins.append(dict(low=edges[b], high=edges[b + 1], n=int(m.sum()),
                         accuracy=float(hit[m].mean()), confidence=float(conf[m].mean())))
    return dict(accuracy=float(hit.mean()), mean_confidence=float(conf.mean()),
                nll=float(-np.log(np.clip(p[np.arange(len(targets)), targets], 1e-12, 1)).mean()),
                brier=float(((p - onehot) ** 2).sum(1).mean()), ece=float(ece), mce=float(mce), bins=bins)


def fit_temperature(logits, targets):
    grid = np.exp(np.linspace(np.log(0.05), np.log(10), 400))
    nll = [metrics(logits, targets, t)['nll'] for t in grid]
    best = grid[int(np.argmin(nll))]
    fine = np.linspace(best * 0.97, best * 1.03, 121)
    return float(fine[int(np.argmin([metrics(logits, targets, t)['nll'] for t in fine]))])


def main():
    src = json.loads(SOURCE.read_text())
    ckpt = Path(src['models']['current_primary']['path'])
    assert sha(ckpt) == src['models']['current_primary']['sha256']
    model, _ = load_model(ckpt)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.eval().to(device)
    all_logits, all_targets, all_domains, all_ids, pred_check = [], [], [], [], 0
    for domain in ['citizen', 'semlex', 'local']:
        p = CACHE / (domain + '_val.npz')
        assert sha(p) == src['domains'][domain]['input_sha256'], domain
        with np.load(p, allow_pickle=False) as z:
            arrays = [z[k] for k in ['landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes']]
            ids = z['item_ids'].tolist()
            targets = z['targets']
        position = {k: i for i, k in enumerate(ids)}
        keep = np.array([position[k] for k in src['domains'][domain]['ids']])
        out = []
        with torch.inference_mode():
            for i in range(0, len(keep), 128):
                rows = keep[i:i + 128]
                x = [torch.as_tensor(a[rows], device=device).float() for a in arrays]
                x[2] = x[2] > .5
                ll, lf = model.landmark_model(x[0], return_embeddings=True)
                hf = model.hand_model.forward_features(*x[1:])
                hl = model.hand_model.classifier(hf)
                out.append(model.fusion_head(lf, hf, ll, hl).float().cpu().numpy())
        logits = np.concatenate(out).astype(np.float64)
        t = targets[keep]
        assert t.tolist() == src['domains'][domain]['targets']
        pred_check += int((logits.argmax(1) == np.array(src['domains'][domain]['predictions']['current_primary'])).sum())
        all_logits.append(logits); all_targets.append(t)
        all_domains += [domain] * len(t); all_ids += src['domains'][domain]['ids']
    logits = np.concatenate(all_logits); targets = np.concatenate(all_targets)
    n = len(targets)
    assert n == 3957 and int((logits.argmax(1) == targets).sum()) == 3779, 'membership drift'

    # Two-fold cross-fit, stratified by domain through a stable hash of the clip id.
    fold = np.array([int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 for i in all_ids])
    scaled = np.empty_like(logits)
    temps = {}
    for f in (0, 1):
        t = fit_temperature(logits[fold != f], targets[fold != f])
        temps[f'fit_on_fold_{1 - f}_applied_to_fold_{f}'] = t
        scaled[fold == f] = logits[fold == f] / t
    full_t = fit_temperature(logits, targets)

    rng = np.random.default_rng(20261027)
    boot = {'before': [], 'after': []}
    for _ in range(1000):
        s = rng.integers(0, n, n)
        boot['before'].append(metrics(logits[s], targets[s])['ece'])
        boot['after'].append(metrics(scaled[s], targets[s])['ece'])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

    result = dict(
        scope='Primary recognition head, PyTorch FP32, 3,957 admitted isolated-sign validation clips '
              '(citizen/semlex/local). Validation only; Citizen test gate untouched.',
        checkpoint=str(ckpt), checkpoint_sha256=src['models']['current_primary']['sha256'],
        membership_source=str(SOURCE.relative_to(R)), argmax_agreement_with_saved=f'{pred_check}/{n}',
        bins=BINS, method='Temperature scaling (single scalar T dividing logits), NLL-minimised by grid search; '
                          'two-fold cross-fit by sha256(clip id) parity.',
        temperatures=temps, temperature_fit_on_all=full_t, n=n,
        before=metrics(logits, targets), after_cross_fit=dict(metrics(scaled, targets)),
        ece_bootstrap_95ci=ci,
        per_domain={d: dict(n=int(sum(x == d for x in all_domains)),
                            before=metrics(logits[np.array(all_domains) == d], targets[np.array(all_domains) == d]),
                            after=metrics(scaled[np.array(all_domains) == d], targets[np.array(all_domains) == d]))
                    for d in ['citizen', 'semlex', 'local']},
    )
    for d in result['per_domain'].values():
        d['before'].pop('bins'); d['after'].pop('bins')
    (O / 'calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(O / 'calibration_logits.npz', logits=logits.astype(np.float32), targets=targets,
                        ids=np.array(all_ids), fold=fold)
    b, a = result['before'], result['after_cross_fit']
    print(f"acc {b['accuracy']:.4f} | ECE {b['ece']*100:.2f}% -> {a['ece']*100:.2f}% | MCE {b['mce']*100:.2f}% -> {a['mce']*100:.2f}% "
          f"| NLL {b['nll']:.4f} -> {a['nll']:.4f} | Brier {b['brier']:.4f} -> {a['brier']:.4f} | mean conf {b['mean_confidence']:.4f} "
          f"| T {temps} all {full_t:.3f} | CI {ci}")


if __name__ == '__main__':
    main()
