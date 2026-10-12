"""Read-only: does interval adaptation move isolated predictions toward the classes its spans contain?

Interval training draws half of every batch from decoder-matched phrase spans.  Sampling is class-
balanced within each source, so each span class receives 0.5/K of draws (K span classes) versus
0.5/100 per class from isolated replay.  This script measures, on the 4,252 isolated validation clips,
whether predictions, probability mass and logits shift toward the span classes between a parent and an
interval checkpoint.  Validation only; CPU; no training.
usage: mechanism.py NAME PARENT.pth INTERVAL.pth [OUT.json]
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
R = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.train_unfrozen_phrase_adapt_v17 import tensors
import scripts.train_span_recognizer_v17 as tsr

O = Path(__file__).resolve().parent
torch.set_num_threads(3)


def span_counts(labels):
    recipe = json.loads((R / 'active/v17/phrase_segment_recipe_manifest_20261007.json').read_text())
    tsr.EXCLUDE_PREFIX = True
    tsr.ALLOWED_TRAIN_VIDEOS = {e['video_sha256'] for e in recipe['entries'] if e['split'] == 'train'}
    spans, _, videos = tsr.span_dataset(['local_train'], labels)
    assert len(videos) == 179
    return Counter(labels[t] for t in spans['targets'].tolist())


@torch.inference_mode()
def logits(model, values):
    out = []
    for i in range(0, len(values[4]), 128):
        x = [v[i:i + 128] for v in values[:4]]; x[1] = x[1].float(); out.append(model(*x))
    return torch.cat(out).numpy()


def main(name, parent, interval, out_path=None):
    a, ac = load_model(Path(parent)); b, bc = load_model(Path(interval))
    labels = sorted(ac['label_to_index'], key=ac['label_to_index'].get); assert ac['label_to_index'] == bc['label_to_index']
    counts = span_counts(labels); span = np.array([l in counts for l in labels])
    before, after, targets, domain = [], [], [], []
    for d in ('citizen', 'semlex', 'local'):
        with np.load(R / f'artifacts/generated/unfrozen_phrase_adapt_v17/{d}_val.npz', allow_pickle=False) as z:
            v = tensors({k: z[k] for k in z.files})
        before.append(logits(a.eval(), v)); after.append(logits(b.eval(), v)); targets.append(v[4].numpy()); domain += [d] * len(v[4])
    before, after, targets, domain = np.concatenate(before), np.concatenate(after), np.concatenate(targets), np.array(domain)
    pb, pa = before.argmax(1), after.argmax(1)
    sb, sa = (torch.from_numpy(x).softmax(1).numpy() for x in (before, after))
    lost = (pb == targets) & (pa != targets); gained = (pb != targets) & (pa == targets)
    changed = pb != pa
    in_span_t = span[targets]
    def acc(mask): return {'n': int(mask.sum()), 'before': int((pb == targets)[mask].sum()), 'after': int((pa == targets)[mask].sum())}
    # Logit shift per class, centred per clip so a uniform offset does not count as a shift.
    shift = (after - after.mean(1, keepdims=True)) - (before - before.mean(1, keepdims=True))
    class_shift = shift.mean(0)
    freq = np.array([counts.get(l, 0) for l in labels], float)
    rank = lambda x: np.argsort(np.argsort(x))
    margin_b = np.sort(sb, 1)[:, -1] - np.sort(sb, 1)[:, -2]
    result = {
        'chain': name, 'parent': str(parent), 'interval': str(interval), 'scope': 'validation only, read-only',
        'span_classes': dict(sorted(counts.items(), key=lambda r: -r[1])), 'span_examples': int(sum(counts.values())),
        'sampling_share_per_span_class': 0.5 / len(counts) + 0.5 / 100, 'sampling_share_per_other_class': 0.5 / 100,
        'correct': {'before': int((pb == targets).sum()), 'after': int((pa == targets).sum()), 'n': len(targets)},
        'lost': int(lost.sum()), 'gained': int(gained.sum()), 'changed': int(changed.sum()),
        'accuracy_by_target': {'span_classes': acc(in_span_t), 'other_classes': acc(~in_span_t)},
        'lost_target_in_span': int((lost & in_span_t).sum()), 'lost_prediction_to_span_class': int((lost & span[pa]).sum()),
        'gained_target_in_span': int((gained & in_span_t).sum()),
        'changed_prediction_to_span_class': int((changed & span[pa]).sum()),
        'changed_prediction_from_span_class': int((changed & span[pb]).sum()),
        'predictions_of_span_classes': {'before': int(span[pb].sum()), 'after': int(span[pa].sum()),
                                        'true_span_targets': int(in_span_t.sum())},
        'false_positive_span_predictions': {'before': int((span[pb] & (pb != targets)).sum()), 'after': int((span[pa] & (pa != targets)).sum())},
        'mean_probability_on_span_classes': {'before': float(sb[:, span].sum(1).mean()), 'after': float(sa[:, span].sum(1).mean()),
            'before_on_other_targets': float(sb[~in_span_t][:, span].sum(1).mean()), 'after_on_other_targets': float(sa[~in_span_t][:, span].sum(1).mean())},
        'centred_logit_shift': {'span_classes_mean': float(class_shift[span].mean()), 'other_classes_mean': float(class_shift[~span].mean()),
            'spearman_with_span_frequency': float(np.corrcoef(rank(class_shift), rank(freq))[0, 1]),
            'span_classes_with_positive_shift': int((class_shift[span] > 0).sum()), 'span_class_count': int(span.sum()),
            'other_classes_with_positive_shift': int((class_shift[~span] > 0).sum())},
        'parent_top2_margin': {'lost_median': float(np.median(margin_b[lost])) if lost.any() else None,
                               'all_correct_median': float(np.median(margin_b[pb == targets])),
                               'lost_in_lowest_decile_of_correct': int((margin_b[lost] <= np.quantile(margin_b[pb == targets], .1)).sum())},
        'by_domain': {d: {'before': int((pb == targets)[domain == d].sum()), 'after': int((pa == targets)[domain == d].sum()),
                          'lost': int((lost & (domain == d)).sum()), 'gained': int((gained & (domain == d)).sum())} for d in ('citizen', 'semlex', 'local')},
        'transitions': [{'domain': str(domain[i]), 'target': labels[targets[i]], 'before': labels[pb[i]], 'after': labels[pa[i]],
                         'kind': 'lost' if lost[i] else 'gained' if gained[i] else 'wrong_to_wrong',
                         'after_is_span_class': bool(span[pa[i]]), 'target_is_span_class': bool(span[targets[i]]),
                         'parent_margin': float(margin_b[i])} for i in np.flatnonzero(changed)],
    }
    out = Path(out_path) if out_path else O / f'mechanism_{name}.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'transitions'}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
