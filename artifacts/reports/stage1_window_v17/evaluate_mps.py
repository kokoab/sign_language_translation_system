"""Accelerate the frozen evaluator; require full epoch-1 CPU/MPS label parity."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
from scripts import evaluate_stage1_window_v17 as evaluation

root = Path('artifacts/reports/stage1_window_v17')
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


evaluation.load_stage1_window_checkpoint = load_mps
parity = evaluation.evaluate(Path('artifacts/models/stage1_window_v17_seed17111/epoch_01.pth'),
                             root/'evaluation_manifest.json', root/'mps_parity_epoch01.json')
cpu = json.loads((root/'evaluations/epoch_01.json').read_text())
assert len(cpu['rows']) == len(parity['rows']) == 334
for before, after in zip(cpu['rows'], parity['rows']):
    assert before['item_id'] == after['item_id']
    assert before['final'] == after['final']
    assert [r['prediction'] for r in before['updates']] == [r['prediction'] for r in after['updates']]
assert cpu['transition_predictions'] == parity['transition_predictions']
print('CPU/MPS parity: 334 transcripts, 7212 window labels, 16 transition labels', flush=True)
for epoch in range(5, 13):
    path = root/f'evaluations/epoch_{epoch:02}.json'
    if path.exists():
        raise ValueError(f'refusing to replace {path}')
    report = evaluation.evaluate(Path(f'artifacts/models/stage1_window_v17_seed17111/epoch_{epoch:02}.pth'),
                                 root/'evaluation_manifest.json', path)
    report['evaluation_device'] = 'mps'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('Completed epoch', epoch, flush=True)
