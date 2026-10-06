"""Current manuscript charts. Reads completed measurements; never runs evaluation.

Run this after any historical review-asset rebuild to restore current figures.
"""
import json
from pathlib import Path
from statistics import median
import numpy as np
from build_review_assets import plt, save, BLUE, TEAL, ROOT


def phone_chart():
    source = ROOT / 'artifacts/reports/phone_precision_v17_20261007/device_summary.json'
    summary = json.loads(source.read_text())
    values = [summary[k]['median_of_run_medians_ms'] for k in ['selected_fp16_all', 'all_visual_fp32']]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.barh([0, 1], values, color=[TEAL, BLUE], height=.48)
    ax.set_yticks([0, 1], ['Current FP16\nvisual / recognition models', 'FP32\nvisual / recognition models'])
    ax.invert_yaxis(); ax.set_xlim(0, 130)
    ax.set_xlabel('Preparation and recognition per frame (ms)')
    ax.set_title('On-device processing on iPhone 13', loc='left', weight='bold')
    for i, v in enumerate(values):
        ax.text(v + 2, i, f'{v:.1f}', va='center')
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    save(fig, 'phone_precision')


def mac_chart():
    folder = ROOT / 'artifacts/reports/capstone_mac_comparison_v17_20261007'
    data = {family: json.loads((folder / f'{family}.json').read_text()) for family in ['apple', 'mediapipe']}
    paths = [[r['path'] for r in data[f]['rows']] for f in data]
    assert paths[0] == paths[1] and len(paths[0]) == 378
    assert not any(r['failed'] for d in data.values() for r in d['rows'])
    keys = ['base', 'fusion', 'span']
    times = {f: [median(r['total_ms'][k] for r in d['rows']) for k in keys] for f, d in data.items()}
    scores = {'apple': [95.77, 96.30, 95.24], 'mediapipe': [91.80, 94.44, 94.44]}
    labels = ['Landmark-only', 'Combined inputs', 'Interval-adapted']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    y = np.arange(3)
    for ax, measures, title, xlabel in [(axes[0], scores, 'Recognition performance', 'Validation Top-1 accuracy (%)'), (axes[1], times, 'Processing on Apple M4', 'Median staged processing per clip (ms)')]:
        for offset, f, name, color in [(-.18, 'apple', 'Apple Vision', BLUE), (.18, 'mediapipe', 'MediaPipe', TEAL)]:
            values = measures[f]
            ax.barh(y + offset, values, height=.33, label=name, color=color)
            for yy, v in zip(y + offset, values):
                ax.text(v + max(max(x) for x in measures.values())*.018, yy, f'{v:.2f}' if measures is scores else f'{v:.0f}', va='center', fontsize=9)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel(xlabel)
        ax.set_title(title, loc='left', weight='bold', fontsize=12)
        ax.set_xlim(0, 111 if measures is scores else max(max(x) for x in measures.values())*1.18)
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    save(fig, 'extractors')
    (folder / 'chart_values.json').write_text(json.dumps({'accuracy': scores, 'median_staged_ms': times}, indent=2) + '\n')

if __name__ == '__main__':
    phone_chart()
    mac_chart()
