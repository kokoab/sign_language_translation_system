import json
from pathlib import Path
root=Path(__file__).resolve().parent
data=json.loads((root/'verified_results.json').read_text())
phone=json.loads((root/'phone_results.json').read_text())['summary']
lines=['# Pinned recognition comparison — 2026-10-07','',
 'The historical part-wise checkpoint still reproduces 96.83% validation Top-1. '
 'Its FP32 and FP16 phone exports preserve all original predictions. '
 'These are selected existing checkpoints evaluated on identical inputs, not a new '
 'matched from-scratch training experiment. No protected test was accessed.', '',
 '| Model / stage | Validation Top-1 | Top-5 | Mac CPU inference (ms) |',
 '| --- | ---: | ---: | ---: |']
for name,r in data['models'].items():
    lines.append(f"| {name.replace('_',' ')} | {r['top1']:.2f}% | {r['top5']:.2f}% | {r['median_ms']:.2f} |")
lines+=['','Mac timings use FP32 PyTorch, one CPU thread, batch one, identical prepared '
 'validation inputs, 30 warm-ups and three ordered passes. Values are medians of run '
 'medians. The initial timings overlapped export preparation and are excluded; '
 '`verified_results.json` contains the isolated rerun and exact checkpoint/input hashes. '
 'Timing excludes extraction, camera, image encoding, decoder, English and UI.', '',
 '## Physical iPhone 13','',
 '| Classifier | Top-1, both precisions | FP32 median (ms) | FP16 median (ms) |',
 '| --- | ---: | ---: | ---: |']
for name in ['Transformer','FlatSqueezeformer','PartwiseSqueezeformer']:
    a,b=phone[name+'FP32'],phone[name+'FP16']
    assert a['top1']==b['top1']
    lines.append(f"| {name} | {a['top1']:.2f}% | {a['median_ms']:.2f} | {b['median_ms']:.2f} |")
lines+=['', 'Six rotating orders place every configuration in every position. Core ML ALL, '
 'prepared inputs, 30 warm-ups per measurement, synchronous prediction only. All recorded '
 'thermal states nominal, low-power off, all predictions unchanged from PyTorch and Mac '
 'conversion checks. Small FP16 timing differences should be interpreted with per-pass '
 'variability, available in `phone_results.json`. These are not full-system 28 ms timings.', '',
 '## Correct training-stage interpretation','',
 'Flat Squeezeformer exceeds flat Transformer by 0.26 percentage points in these selected '
 'checkpoints. Part-wise/global Squeezeformer improves on the flat Squeezeformer by 1.06 '
 'points and is the most accurate base classifier here. Different historical training '
 'runs and selection histories mean this is not proof of a universal architecture advantage.', '',
 'The deployed branch initializes from the orientation-robust checkpoint, which is a '
 'separate part-wise model, rather than directly from the historical 96.83% weights. '
 'The local-adapted landmark branch is embedded in the unified multimodal recognizer; '
 'phrase/activity adaptation starts from that unified model, and interval adaptation '
 'starts from phrase/activity adaptation. Source checkpoint hashes in these artifacts '
 'were checked where provided. Preserve this intermediate orientation/local adaptation '
 'in any lineage explanation; do not draw an unsupported direct weight-inheritance arrow.', '',
 'Later stages optimize additional domains or candidate intervals. Their isolated-sign '
 'Top-1 scores are not monotonically increasing; do not transfer 96.83% to those stages '
 'or claim a stream-level gain from this isolated evaluation.', '',
 '## Next authorized experiment','',
 'See `TRAINING_PLAN.md`: current trainer, strict warm-start from the preserved96.83% '
 'checkpoint, mild-roll control versus full-circle rotation treatment, equal20epoch '
 'budgets. Check `finetune/status.json` for completion before assuming results. The '
 'original remains the fallback. Full downstream retraining and manuscript updates '
 'remain pending; this report does not claim those tasks are complete.', '',
 'Phone installation first hit the free-profile app limit. Reusing the previously created '
 'benchmark bundle and a fresh build directory resolved it; production app/models stayed '
 'unchanged. Failed logs are preserved. No manuscript edits were made during this work.', '']
(root/'REPORT.md').write_text('\n'.join(lines))
