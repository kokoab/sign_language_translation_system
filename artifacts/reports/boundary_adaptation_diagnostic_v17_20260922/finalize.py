"""Recompute saved scores/provenance and write the evidence report."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts import evaluate_boundary_expanded_v17 as e
OUT=Path(__file__).parent
load=lambda name:json.loads((OUT/name).read_text())
x=load('evaluation.json');cal=load('calibration.json');audit=load('contract_audit.json');bench=load('benchmark_interleaved.json')
old=json.loads((e.REPORT/'evaluation.json').read_text())
base={r['id']:r for r in old['runs'][0]['records']}
rows=load('evaluation_membership.json')['videos'];assert len(rows)==72 and len({r['video_sha256'] for r in rows})==72
for run in x['runs']:
 assert len(run['records'])==72
 for r in run['records']:
  assert r['reference']==base[r['id']]['reference']
  assert r['hypothesis']==[p['committed_gloss'] for p in r['predictions'] if p['committed_gloss']]
  assert json.loads(json.dumps(e.edit_counts(r['reference'],r['hypothesis'])))==r['metrics']
 assert e.summarize(run['records'],list(base.values()))==run['summaries']['combined']
assert x['runs'][0]['summaries']['combined']['correct']==127
assert x['runs'][0]['summaries']['combined']['wer']==74/186
for row in rows:assert e.digest(ROOT/row['video_path'])==row['video_sha256']
for spec,saved in zip(e.ARMS[1:],load('weight_audit.json')):assert e.digest(spec['checkpoint'])==saved['checkpoint_sha256']
prior=json.loads((e.REPORT/'tcn_comparison/audit.json').read_text())
for path,sha in prior['source_sha256'].items():assert e.digest(ROOT/path)==sha,(path,'historical source changed')
lines=['# Pretrained boundary regression: controlled BIO readout experiment','',
 'No training, deployment or distillation. Existing weights, splits, reports and frozen Reel preserved.',
 'Same72 development videos/186 signs; familiar-signer reused development, not independent generalization.', '',
 '## Results','', '| Backbone / readout | WER | Correct /186 | Retained frozen /127 | Insertions | Exact /72 |',
 '|---|---:|---:|---:|---:|---:|']
for run in old['runs']:
 s=run['summaries']['combined'];retained=e.summarize(run['records'],list(base.values()))['retained_baseline_correct_events']
 lines.append(f"| Existing {run['arm']} | {s['wer']:.2%} | {s['correct']} | {retained} | {s['insertions']} | {s['exact_videos']} |")
for run in x['runs'][1:]:
 s=run['summaries']['combined'];lines.append(f"| {run['arm']} | {s['wer']:.2%} | {s['correct']} | {s['retained_baseline_correct_events']} | {s['insertions']} | {s['exact_videos']} |")
lines += ['', '## Interpretation and minimal change','',
 'All72 frozen hypotheses AND interval boundaries reproduce the original saved evaluation exactly. Each adapted checkpoint has an unchanged temporal CNN, input normalization and original BIO head (exact tensor equality); only attention and the edge head differ. Restoring the original BIO readout/decoder isolates the major regression to the replacement readout/decoding path. This does not distinguish edge-head quality from decoder policy, nor prove absence of smaller representation changes.',
 '', 'The original BIO decoder groups B/I activity, accepts three-frame spans and closes at EOF. The edge decoder requires rising START/END pairs, uses 0.15–4.0s duration limits, replaces unmatched starts and drops unfinished EOF intervals. Training optimizes masked, weighted frame BCE, not these discrete interval/commit decisions. EOF alone is not established as the sole cause.',
 '', 'Small code change: scripts/evaluate_boundary_expanded_v17.py run_arm now accepts readout="bio" independently of whether a checkpoint is loaded. Existing defaults remain unchanged. Regression test compares actual adapted-backbone BIO logits against the direct original-head path; invalid readout is rejected. The diagnostic script retains a separate controlled path and full frozen-output reproduction checks.',
 '', '## Supervision, preprocessing and timing audit','',
 f"Reconstructed every cached target/index/split/parent: {audit['cached_rows_verified']}. Unknown labels have exactly zero gradient. Sampled clean projection parity max absolute error {max(r['max_abs'] for r in audit['cache_parity']):.8f}. Pose/source hashes pass. No negative supervision was found outside approved intervals; positives can occupy the documented ±50ms edge band. No unknown gap was relabeled background.",
 '', 'Training/evaluation share20Hz past-observation sampling,64-frame windows,target53,10 future frames (500ms), bounded-window normalization and explicit zero-confidence padding. CNN caches include temporal context. Raw/cache paths agree. Augmentation changes past observation holds without moving target timestamps. Existing clock/future/padding tests pass. All arms in the new comparison share identical original decoding,100ms classifier context and commit rules.',
 '', '## Training-held calibration and stopping decision','',
 'Only2 existing calibration videos (4 signs) match complete approved locked-vocabulary sequence targets.129 other calibration records provide boundary supervision but do not establish complete locked-vocabulary whole-video transcripts. Splits were not changed and no OOV region was silently scored as blank.',
 '', '| Backbone | Readout | Calibration WER | Correct /4 | Insertions |', '|---|---|---:|---:|---:|']
for r in cal['runs']:
 s=r['summary'];lines.append(f"| {r['arm']} | {r['readout']} | {s['wer']:.2%} | {s['correct']} | {s['insertions']} |")
lines += ['', f"The predeclared calibration rule chooses {cal['selected']['arm']} / {cal['selected']['readout']}, based on one correct sign; it does not select a BIO-adapted checkpoint. No candidate is promoted. Expanded results are descriptive only, not used to override calibration or select an epoch/threshold. This exposes an inadequate recognition-selection signal, so another fine-tune is not justified yet. The bounded experiment used fixed existing weights and zero gradient steps.",
 '', '## Runtime','', f"Controlled72-video three-arm replay: {x['elapsed_seconds']:.2f}s. Calibration five-arm replay: {cal['elapsed_seconds']:.2f}s. These wall times include some overlapping audit/test work and are not isolated throughput benchmarks.",
 '', '| Backbone | Readout | Batch1 median ms | p95 ms |', '|---|---|---:|---:|']
for r in bench['results']:lines.append(f"| {r['arm']} | {r['readout']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} |")
lines += ['', 'Synchronized MPS benchmark,30 measured windows after5 warmups per arm,real cached calibration pose,seeded shuffled interleaving. Initial sequential benchmark (preserved as benchmark.json) drifted from14ms to31ms across equal-size models; the interleaved repeat avoids attributing that order/load drift to model or readout speed. Includes normalization,CNN,attention,readout and CPU output. Excludes MediaPipe frontend,Reel and camera scheduling. Intrinsic future delay remains500ms. This is not a live/iPhone readiness measurement.',
 '', f"Shared72-video Apple Vision frontend: {sum(t['apple_frontend_seconds'] for t in x['timings']):.2f}s; shared bounded normalization/CNN projection: {sum(t['projection_seconds'] for t in x['timings']):.2f}s."]
for r in x['runs']:
 lines.append(f"- {r['arm']}: attention/BIO {sum(v['head_seconds'] for v in r['records']):.2f}s; Reel classification {sum(v['classifier_seconds'] for v in r['records']):.2f}s.")
lines += ['', '## Validation and next safe action','',
 '23 focused tests pass, including the new original-BIO override test (first failed because the option was absent). All saved edit counts, summaries, video/checkpoint hashes and historical comparison source hashes rechecked. No original result files or weights overwritten. Small audit setup error (some combined records lack source_item_id) was corrected by filtering that field before matching; no model/data changes resulted.',
 '', 'Keep untouched pretrained BIO as the reference. Preserve the original head/decoder in future adaptation comparisons. Resolve complete recognition scoring within the existing calibration split before new checkpoint-selection training; do not select on expanded72 or simply train longer. The fixed-weight readout recovery is measured; a deployable adaptation selected by reliable held-out recognition is not established.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
e.atomic(OUT/'verification.json',dict(status='passed',videos=72,references=186,all_metrics_recomputed=True,old_source_hashes_unchanged=True,checkpoints_unchanged=True,
 code_sha256={str(p.relative_to(ROOT)):e.digest(p) for p in [ROOT/'scripts/evaluate_boundary_expanded_v17.py',ROOT/'scripts/diagnose_boundary_adaptation_v17.py',ROOT/'test/test_pretrained_boundary_v17.py']},promoted=False))
print(json.dumps([dict(arm=r['arm'],**r['summaries']['combined']) for r in x['runs']]))
