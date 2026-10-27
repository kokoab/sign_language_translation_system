"""Reliability diagram for the primary recognizer, before and after cross-fitted temperature scaling."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O = Path(__file__).resolve().parent
r = json.loads((O / 'calibration.json').read_text())
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SERIES = {'before': ('#2a78d6', 'o'), 'after_cross_fit': ('#eb6834', 's')}   # reference palette slots 1-2, fixed order
labels = {'before': f"Uncalibrated (ECE {100 * r['before']['ece']:.2f}%)",
          'after_cross_fit': f"Temperature-scaled, T ≈ {r['temperature_fit_on_all']:.2f} (ECE {100 * r['after_cross_fit']['ece']:.2f}%)"}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2})
fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1, label='Perfect calibration')
for key, (color, marker) in SERIES.items():
    bins = [b for b in r[key]['bins'] if b['n'] >= 10]          # sparse bins are noise; counts listed in calibration.json
    xs = [b['confidence'] for b in bins]; ys = [b['accuracy'] for b in bins]
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=6.5, mec=SURFACE, mew=1.5, label=labels[key], zorder=3)
ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.02)
ax.set_xlabel('Mean predicted confidence in bin (top-1 softmax probability)')
ax.set_ylabel('Observed accuracy in bin')
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(GRID)
ax.legend(loc='lower right', frameon=False, fontsize=8.8, labelcolor=INK)
ax.set_title('Primary recognizer reliability, 3,957 validation clips', loc='left', fontsize=11, color=INK, pad=10)
fig.text(0.01, 0.005, '15 equal-width bins; bins with fewer than 10 clips not drawn. Points above the diagonal = under-confident.',
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ['png', 'svg']:
    fig.savefig(O / f'calibration_reliability.{ext}', facecolor=SURFACE)
print('saved')
