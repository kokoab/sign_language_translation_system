"""Run the existing frozen evaluator and report positive-only held-out LG accuracy."""
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from active.v17 import train_stage1_window_v17 as training
from scripts import evaluate_stage1_window_v17 as evaluation
from scripts.select_stage1_window_v17 import select

REPORT = Path('artifacts/reports/o5s5_augmented_v17_20260914')
MODELS = Path('artifacts/models/stage1_window_o5s5_v17_seed17111')
BASE = Path('artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth')
load = evaluation.load_stage1_window_checkpoint


class MPS(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.to('mps')

    def forward(self, features):
        return self.model(features.to('mps')).cpu()


def load_mps(path):
    model, labels, payload = load(path)
    return MPS(model).eval(), labels, payload


@torch.inference_mode()
def lg_accuracy(model, samples):
    predictions = []
    for start in range(0, len(samples), 64):
        features = np.stack([r.features for r in samples[start:start+64]])
        predictions.extend(model(torch.from_numpy(features)).argmax(-1).tolist())
    rows = [dict(identity=r.identity, target=r.target, prediction=p, schedule=r.schedule)
            for r, p in zip(samples, predictions)]
    assert len(rows) == len(samples) == 318
    totals = Counter(r['target'] for r in rows)
    correct = Counter(r['target'] for r in rows if r['target'] == r['prediction'])
    by_schedule = {}
    for schedule in sorted({r['schedule'] for r in rows}):
        subset = [r for r in rows if r['schedule'] == schedule]
        count = sum(r['target'] == r['prediction'] for r in subset)
        by_schedule[schedule] = dict(correct=count, total=len(subset), accuracy=count/len(subset))
    return dict(correct=sum(correct.values()), total=len(rows), accuracy=sum(correct.values())/len(rows),
                macro_accuracy=float(np.mean([correct[k]/v for k, v in totals.items()])),
                classes=len(totals), no_emit=sum(r['prediction'] == 100 for r in rows),
                by_schedule=by_schedule, rows=rows)


def main():
    torch.set_num_threads(2)
    training.verify_development_freeze(REPORT/'development_freeze.json', BASE, REPORT/'combined_supervision.json')
    checkpoint = torch.load(BASE, map_location='cpu', weights_only=False)
    labels = {str(k): int(v) for k, v in checkpoint['label_to_index'].items() if int(v) < 100}
    samples, audit = training.load_context_samples(REPORT/'combined_supervision.json', labels, 'validation')
    lg = [r for r in samples if r.source == 'o5s5']
    assert len(lg) == 318 and {r.signer for r in lg} == {'LG'}
    assert {r.category for r in lg} == {'context'} and len({r.target for r in lg}) == 27
    lg_report = dict(metric='positive-window classification; no full-narrative WER',
                     limitations='Incomplete annotation; overlapping windows are correlated, one held-out signer only.',
                     supervision_sha256=training._sha256(REPORT/'combined_supervision.json'), models={})
    # Original classifier is the own-start retention reference, without a random NO_EMIT head.
    base = MPS(training._load_start(checkpoint).base).eval()
    lg_report['models']['original_base'] = dict(checkpoint_sha256=training._sha256(BASE), **lg_accuracy(base, lg))
    del base
    for epoch in (1, 4):
        path = Path(f'artifacts/models/stage1_window_v17_seed17111/epoch_{epoch:02}.pth')
        model, _, _ = load_mps(path)
        lg_report['models'][f'prior_no_o5s5_epoch_{epoch:02}'] = dict(checkpoint_sha256=training._sha256(path), **lg_accuracy(model, lg))
        del model
    evaluation.load_stage1_window_checkpoint = load_mps
    for epoch in range(1, 13):
        path = MODELS/f'epoch_{epoch:02}.pth'
        output = REPORT/f'evaluations/epoch_{epoch:02}.json'
        if output.exists():
            raise FileExistsError(output)
        result = evaluation.evaluate(path, REPORT/'evaluation_manifest.json', output)
        assert len(result['rows']) == 334 and result['epoch'] == epoch
        assert result['metrics']['asllrp_other_ctc']['reference_tokens'] == 284
        result['evaluation_device'] = 'mps'
        output.write_text(json.dumps(result, indent=2)+'\n')
        model, _, payload = load_mps(path)
        assert payload['stage1_window']['development_freeze_sha256'] == training._sha256(REPORT/'development_freeze.json')
        lg_report['models'][f'o5s5_epoch_{epoch:02}'] = dict(checkpoint_sha256=result['checkpoint_sha256'], **lg_accuracy(model, lg))
        del model
        (REPORT/'lg_evaluation.json').write_text(json.dumps(lg_report, indent=2)+'\n')
        print(json.dumps(dict(completed_epoch=epoch, lg_accuracy=lg_report['models'][f'o5s5_epoch_{epoch:02}']['accuracy'])), flush=True)
    decision = select(REPORT, REPORT/'evaluations')
    (REPORT/'selection.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(dict(eligible=decision['eligible'], diagnostic_epoch=decision['diagnostic_checkpoint']['epoch'])), flush=True)


if __name__ == '__main__':
    main()
