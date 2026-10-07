"""Review-only figure from the final recognizer's recorded training history."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
history = json.loads((ROOT / 'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe/chain_9683/span_recognizer/history.json').read_text())['history']
assert len(history) == 9 and abs(history[4]['tune']['wer'] * 100 - 31.41592920353982) < 1e-8
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
trained = [r for r in history if r['loss'] is not None]
axes[0].plot([r['epoch'] for r in trained], [r['loss'] for r in trained], 'o-', color='#285b7e')
axes[0].set(xlabel='Training epoch', ylabel='Training objective', title='Training objective')
axes[1].plot([r['epoch'] for r in history], [100*r['tune']['wer'] for r in history], 'o-', color='#238578')
axes[1].set(xlabel='Epoch (0 = initial recognizer)', ylabel='Development gloss-sequence WER (%)', title='Development sequence recognition', ylim=(29, 40))
axes[1].annotate('Selected: epoch 4\n31.42% WER', xy=(4,100*history[4]['tune']['wer']), xytext=(4.5,35.5), arrowprops={'arrowstyle':'->','color':'#555555'}, fontsize=10)
for ax in axes:
    ax.axvline(4, color='#b87135', linestyle='--', linewidth=1.3)
    ax.grid(alpha=.16)
    ax.set_xticks(range(0 if ax is axes[1] else 1,9))
fig.suptitle('Final recognizer: training and development performance', fontsize=14, weight='bold')
for suffix in ('png','svg'):
    fig.savefig(OUT / ('proposed_recognizer_training.'+suffix), dpi=180, bbox_inches='tight')
plt.close(fig)
