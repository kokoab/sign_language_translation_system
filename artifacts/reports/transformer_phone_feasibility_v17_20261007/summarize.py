"""Summarize completed, counterbalanced phone runs; fail on missing/confounded data."""
import json
import statistics as stats
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for line in (root / 'phone_test.log').open():
    if 'CLASSIFIER_RESULT ' in line:
        rows.append(json.loads(line.split('CLASSIFIER_RESULT ', 1)[1]))
assert len(rows) == 16, f'Expected 16 measurements, got {len(rows)}'
conversion = json.loads((root / 'conversion_summary.json').read_text())
assert set(r['model'] for r in rows) == set(conversion)
summary = {}
for name, converted in conversion.items():
    runs = sorted((r for r in rows if r['model'] == name), key=lambda r: r['pass'])
    assert [r['pass'] for r in runs] == [0, 1, 2, 3]
    assert all(not r['low_power'] and r['thermal_before'] < 2 and r['thermal_after'] < 2 for r in runs)
    assert all(r['predictions'] == runs[0]['predictions'] and r['top5'] == runs[0]['top5'] for r in runs)
    assert all(len(r['samples_ms']) == r['count'] for r in runs)
    medians = [r['median_ms'] for r in runs]
    summary[name] = {
        'median_of_run_medians_ms': stats.median(medians),
        'run_medians_ms': medians,
        'median_of_run_p90_ms': stats.median(r['p90_ms'] for r in runs),
        'top1_percent': 100 * runs[0]['correct'] / runs[0]['count'],
        'top5_percent': 100 * runs[0]['top5_correct'] / runs[0]['count'],
        'changed_from_pytorch': runs[0]['changed_from_pytorch'],
        'changed_from_mac': runs[0]['changed_from_mac'],
        'package_mib': converted['package_bytes'] / 1024**2,
        'thermal_states': sorted(set(r[k] for r in runs for k in ['thermal_before', 'thermal_after'])),
    }
(root / 'phone_results.json').write_text(json.dumps({'summary': summary, 'runs': rows}, indent=2) + '\n')
lines = ['# Transformer / Squeezeformer phone feasibility', '',
    'Measured on physical iPhone 13 with a separate Release benchmark application. '
    'Existing matched base-classifier checkpoints; same canonical validation inputs, '
    'batch one, 32-frame landmark clips, Core ML compute units ALL. '
    'Four counterbalanced passes, 30 warm-up predictions before each measurement. '
    'No retraining, protected-test access, production-model replacement or paper changes.', '',
    '| Model | Precision | Top-1 | Top-5 | Median inference (ms/clip) | Run medians range | P90 (ms) | Package (MiB) |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for name, r in summary.items():
    family, precision = name[:-4], name[-4:]
    lines.append(f"| {family} | {precision} | {r['top1_percent']:.2f}% | {r['top5_percent']:.2f}% | "
        f"{r['median_of_run_medians_ms']:.2f} | {min(r['run_medians_ms']):.2f}–{max(r['run_medians_ms']):.2f} | "
        f"{r['median_of_run_p90_ms']:.2f} | {r['package_mib']:.2f} |")
lines += ['', 'Timing is synchronous model prediction only, with prepared inputs. It excludes '
    'camera capture, landmark extraction, hand-image encoding, boundary detection, decoding, '
    'English generation and UI. Median and P90 columns are medians of the four respective '
    'per-pass statistics. Package size is export storage, not resident memory.', '',
    'The original PyTorch and exported-wrapper predictions agreed across validation; '
    'empty-input output parity was also checked. Mac FP32 and FP16 conversions preserved '
    'all original predictions for both families. Phone parity and thermal states appear '
    'in `phone_results.json`; repeated phone predictions and top-five rankings were stable. '
    'Low-power mode was off and no pass began or ended in serious/critical thermal state.', '',
    'This compares the existing flat Transformer and part-wise/global Squeezeformer designs. '
    'It is not a pure attention-versus-convolution ablation, a multi-seed accuracy study, '
    'or a comparison of complete multimodal streaming systems. FP32/FP16 speed differences '
    'include runtime execution placement; they cannot be attributed to precision alone. '
    'Do not equate these timings with the manuscript’s historical 28 ms system result.', '',
    'Toolchain: Xcode 27.0 (27A266a), macOS 27.0, PyTorch 2.8.0, coremltools 9.0, NumPy 1.26.4. '
    'Evidence: `PLAN.md`, `prepare.py`, `conversion_summary.json` (checkpoint/package hashes), '
    '`phone/Resources/manifest.json` (input hash/reference predictions), `phone/BenchTests.swift`, '
    '`phone_test.log`, and `phone_results.json`. The initial unsupported boolean-OR conversion '
    'failure is retained in `prepare_failed_mask.log`; an equivalent torch.where mask passed '
    'reference-parity checks before successful export.', '']
(root / 'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(summary, indent=2))
