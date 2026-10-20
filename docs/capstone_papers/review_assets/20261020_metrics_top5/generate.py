"""Separate Top-5 dashboard batch; originals are read-only inputs."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = OUT.parent / '20261020_primary_metrics'
SOURCE = ROOT / 'artifacts/reports/current_lineage_components_v17_20261013/results.json'
result = json.loads(SOURCE.read_text())
assert result['state'] == 'complete'
targets = np.array([y for d in result['domains'].values() for y in d['targets']])
predictions = np.array([p for d in result['domains'].values()
                        for p in d['predictions']['current_primary']])
assert targets.shape == predictions.shape == (3957,)
assert set(targets) == set(range(100))
assert set(predictions) <= set(range(100))
matrix = np.zeros((100, 100), dtype=int)
np.add.at(matrix, (targets, predictions), 1)
tp = matrix.diagonal()
precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(100), where=matrix.sum(axis=0) > 0)
recall = tp / matrix.sum(axis=1)
f1 = np.divide(2 * precision * recall, precision + recall,
               out=np.zeros(100), where=precision + recall > 0)
values = np.array([tp.sum() / matrix.sum(), precision.mean(), recall.mean(), f1.mean()])
reference = [accuracy_score(targets, predictions), *precision_recall_fscore_support(
    targets, predictions, labels=list(range(100)), average='macro', zero_division=0)[:3]]
np.testing.assert_allclose(values, reference, rtol=0, atol=1e-12)
assert int(tp.sum()) == result['summary']['current_primary']['correct'] == 3779
np.testing.assert_allclose(values[0] * 100, result['summary']['current_primary']['accuracy_percent'])
names = ['Accuracy', 'Macro precision', 'Macro recall', 'Macro F1']
receipt = {
    'scope': 'Primary recognition head; pooled isolated-sign validation, not live-system or phone accuracy.',
    'source': str(SOURCE.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'model': result['models']['current_primary'],
    'n': len(targets), 'correct': int(tp.sum()), 'classes': 100,
    'domains': {k: d['n'] for k, d in result['domains'].items()},
    'averaging': 'Precision, recall and F1 calculated per class, then averaged equally across all 100 signs. Zero denominators yield zero.',
    'metrics_percent': dict(zip(names, (values * 100).tolist())),
    'verification': 'Confusion-matrix calculations match sklearn to 1e-12; accuracy reproduces saved 3779/3957.'
}
HISTORY = ROOT / 'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/history.json'
COMMAND = ROOT / 'ATLAS/python/recipes/span_fragment_reject_command.json'
TRAINER = ROOT / 'ATLAS/python/training/train_span_recognizer_v17.py'
history = json.loads(HISTORY.read_text())['history']
command = json.loads(COMMAND.read_text())
trainer = TRAINER.read_text()
assert '--head-lr' not in command and '--encoder-lr' not in command
assert "ap.add_argument('--encoder-lr', type=float, default=1e-5)" in trainer
assert "ap.add_argument('--head-lr', type=float, default=2e-5)" in trainer
assert 'CosineAnnealingLR(optimizer, T_max=args.epochs)' in trainer
assert [h['epoch'] for h in history] == list(range(9))
receipt['dashboard'] = {
    'history': str(HISTORY.relative_to(ROOT)),
    'history_sha256': hashlib.sha256(HISTORY.read_bytes()).hexdigest(),
    'command': str(COMMAND.relative_to(ROOT)),
    'trainer_sha256': hashlib.sha256(TRAINER.read_bytes()).hexdigest(),
    'selected_epoch': 4,
    'accuracy_curve': 'Recorded per-domain validation accuracy; historical membership differs from pooled summary.',
    'macro_metrics': 'Selected-checkpoint snapshot only; per-epoch classification metrics were not logged.',
    'loss': 'Recorded composite training objective; validation loss was not logged.',
    'learning_rate': 'Reconstructed cosine schedule, eta_min=0, T_max=8; head=2e-5, encoders=1e-5; rate used within each epoch.'
}
(OUT / 'metrics.json').write_text(json.dumps(receipt, indent=2) + '\n')
# Pool correct counts, rather than averaging dataset percentages.
counts = {'citizen': 378, 'semlex': 978, 'local': 2626}
pooled = []
for h in history:
    correct = {d: round(h['domains'][d] * n / 100) for d, n in counts.items()}
    for d, n in counts.items():
        assert abs(correct[d] / n * 100 - h['domains'][d]) < 1e-4
    pooled.append(100 * sum(correct.values()) / sum(counts.values()))
receipt['dashboard']['accuracy_curve'] = 'Single pooled accuracy: summed correct counts / 3982 original validation clips.'
receipt['dashboard']['curve_correct_counts'] = [round(v / 100 * 3982) for v in pooled]
(OUT / 'metrics.json').write_text(json.dumps(receipt, indent=2) + '\n')
BASE_HISTORY = ROOT / 'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/history.json'
base_history = json.loads(BASE_HISTORY.read_text())
base_receipt = json.loads((OLD / 'base_landmark_metrics.json').read_text())
base_values = list(base_receipt['metrics_percent'].values())
assert base_history[99]['epoch'] == base_receipt['epoch'] == 100
np.testing.assert_allclose(base_history[99]['top1'], base_values[0])
np.testing.assert_allclose(base_history[99]['macro_f1'], base_values[3])
base_receipt['history'] = str(BASE_HISTORY.relative_to(ROOT))
base_receipt['history_sha256'] = hashlib.sha256(BASE_HISTORY.read_bytes()).hexdigest()
base_receipt['scope'] = 'Original Citizen validation, 378 clips; no protected test access. Selected checkpoint precision/recall freshly calculated from cached validation inputs; accuracy and macro F1 exactly reproduce recorded values.'
(OUT / 'base_landmark_metrics.json').write_text(json.dumps(base_receipt, indent=2) + '\n')
top5_receipt = json.loads((OUT / 'primary_top5.json').read_text())
assert top5_receipt['n'] == receipt['n']
assert top5_receipt['model']['sha256'] == receipt['model']['sha256']
assert top5_receipt['source_sha256'] == receipt['source_sha256']
base_top5 = base_history[99]['top5']
receipt['dashboard']['loss'] = 'Loss panel omitted at user request.'
receipt['metrics_percent']['Top-5 accuracy'] = top5_receipt['top5_percent']
receipt['dashboard']['top5'] = 'Selected-checkpoint measurement only; no per-epoch Top-5 was recorded.'
base_receipt['metrics_percent']['Top-5 accuracy'] = base_top5
base_receipt['top5_source'] = 'Recorded selected epoch 100 and full per-epoch base history.'
assert base_top5 >= base_values[0]
assert top5_receipt['top5_percent'] >= receipt['metrics_percent']['Accuracy']
(OUT / 'metrics.json').write_text(json.dumps(receipt, indent=2) + '\n')
(OUT / 'base_landmark_metrics.json').write_text(json.dumps(base_receipt, indent=2) + '\n')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
teal, purple, gold, red = '#00b394', '#7561d7', '#e8b348', '#df5545'
for base in ([False] if '--primary-only' in sys.argv else [False, True]):
    title = 'Base Landmark Model' if base else 'Primary Recognizer'
    name = 'base_landmark_metrics_top5' if base else 'primary_recognizer_metrics_top5'
    selected = 100 if base else 4
    scores = base_values if base else values * 100
    epochs = [h['epoch'] for h in base_history] if base else list(range(9))
    accuracy = [h['top1'] for h in base_history] if base else pooled
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(left=.08, right=.97, top=.88, bottom=.26, hspace=.42, wspace=.27)
    fig.suptitle(title, fontsize=25, weight='bold', y=.97)
    ax = axes[0, 0]
    ax.plot(epochs, accuracy, color=teal, lw=2.3, label='Top-1 accuracy')
    top5 = base_top5 if base else top5_receipt['top5_percent']
    if base:
        ax.plot(epochs, [h['top5'] for h in base_history], color=purple, lw=2, label='Top-5 accuracy')
    else:
        ax.scatter([selected], [top5], color=purple, marker='D', s=55, zorder=5,
                   label=f'Top-5 (selected): {top5:.2f}%')
    ax.axvline(selected, color=red, ls='--', alpha=.65, label=f'Selected epoch {selected}')
    ax.set(title='Accuracy (%)', xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 101))
    ax.legend(loc='lower right', fontsize=9)
    ax = axes[0, 1]
    if base:
        ax.plot(epochs, [h['macro_f1'] for h in base_history], color=red, lw=2, label='Macro F1')
        ax.set(title='Macro F1 (%)', xlabel='Epoch', ylabel='Score (%)', ylim=(0, 101))
        ax.legend(loc='lower right', fontsize=9)
    else:
        bars = ax.bar(['Precision', 'Recall', 'F1'], scores[1:], color=[purple, gold, red], width=.55)
        for bar, value in zip(bars, scores[1:]):
            ax.text(bar.get_x()+bar.get_width()/2, value+1.5, f'{value:.2f}%', ha='center', weight='bold')
        ax.set(title='Macro Precision / Recall / F1 (%)', ylabel='Score (%)', ylim=(0, 105), xlabel='Selected checkpoint')
    if base:
        ax = axes[1, 0]
        ax.plot(epochs, [h['train_loss'] for h in base_history], color='#78b4ee', lw=2, label='Training loss')
        ax.plot(epochs, [h['loss'] for h in base_history], color=red, lw=2, label='Validation loss')
        ax.set(title='Loss', xlabel='Epoch', ylabel='Loss')
        ax.legend(fontsize=9)
    else:
        left, right = axes[1, 0].get_position(), axes[1, 1].get_position()
        axes[1, 0].remove()
        axes[1, 1].set_position([left.x0, left.y0, right.x1-left.x0, left.height])
    ax = axes[1, 1]
    lr_epochs = epochs if base else list(range(1, 9))
    lr = [h['lr'] for h in base_history] if base else 2e-5*(1+np.cos(np.pi*(np.arange(1, 9)-1)/8))/2
    ax.plot(lr_epochs, lr, color=teal, lw=2.3)
    ax.set(title='Learning Rate', xlabel='Epoch', ylabel='LR')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    for ax in fig.axes:
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    fig.text(.5, .19, 'Validation Evaluation', ha='center', fontsize=18, weight='bold')
    table_ax = fig.add_axes([.065, .09, .91, .075]); table_ax.axis('off')
    table = table_ax.table(cellText=[[f'{v:.2f}%' for v in [scores[0], top5, *scores[1:]]]],
                          colLabels=['Top-1 accuracy', 'Top-5 accuracy', 'Macro precision', 'Macro recall', 'Macro F1'],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(weight='bold', color='white' if row == 0 else '#111111')
        cell.set_facecolor('#2d3538' if row == 0 else '#e1e7e9')
    note = ('378 validation clips. Precision and recall are available for the selected checkpoint only.' if base else
            'Accuracy curve: original 3,982 clips. Summary: current 3,957 clips. Macro metrics: selected checkpoint only.')
    fig.text(.5, .043, note, ha='center', fontsize=9, color='#505b65')
    if not base:
        fig.text(.5, .023, 'Top-5: selected checkpoint only. Learning rate: recognition-head recipe schedule.', ha='center', fontsize=9, color='#505b65')
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{suffix}', dpi=220, facecolor='white')
    plt.close(fig)
print('Primary:', dict(zip(names, values * 100)))
print('Base:', base_receipt['metrics_percent'])

# Explicitly verify the user's non-replacement requirement.
for name, digest in top5_receipt['original_chart_files_sha256'].items():
    assert hashlib.sha256((OLD / name).read_bytes()).hexdigest() == digest, name
print('Original chart batch unchanged.')
print('Primary Top-5:', top5_receipt['top5_percent'])
print('Base Top-5:', base_top5)
