import json
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).parent
x=json.loads((out/'evaluation.json').read_text());b=json.loads((out/'benchmark_interleaved.json').read_text());c=json.loads((out/'calibration.json').read_text())
a=x['runs'][1]['summaries']['combined'];d=x['runs'][2]['summaries']['combined']
entry=f'''## 2026-09-22 — controlled original-BIO swaps isolate adaptation regression

User authorized fast diagnosis and smallest justified fix after discussion. No new fitting,
deployment, distillation, data/split changes or old artifact overwrites. Same72videos/186signs,
same frozen Reel. Frozen127correct/39.78%WER/15insertions hypotheses AND interval boundaries
reproduced exactly. First adapted backbone + originalBIO: {a['correct']}correct/{a['wer']:.2%}WER/
{a['insertions']}insertions, retains{a['retained_baseline_correct_events']}/127. Augmented+BIO:
{d['correct']}correct/{d['wer']:.2%}WER/{d['insertions']}insertions, retains{d['retained_baseline_correct_events']}/127.
CNN/inputnorm/BIO tensors unchanged exactly; attention and newedgehead are the only changed
tensors. Major regression is replacement readout/decoder, not wholesale representation loss;
head quality versus decoder policy (including EOF) remains unresolved.

Changed scripts/evaluate_boundary_expanded_v17.py: explicit readout='bio' independent of
checkpoint presence, old defaults preserved. Added scripts/diagnose_boundary_adaptation_v17.py
and OriginalBioReadoutTest in test/test_pretrained_boundary_v17.py. Diagnostic/audit/calibration/
benchmark/verification scripts and results in artifacts/reports/boundary_adaptation_diagnostic_v17_20260922/.
Reconstructed all42614cached targets with independent unknown/edge masks, zero unknown loss
gradients, sampled CPU/raw-cache parity <=5.723e-5; shared20Hz/64frames/target53/500msfuture
contract intact.23focusedtests pass; readout test first failed missingkeyword then passed.
Initial audit matching hit missing source_item_id on unrelated combined records; filtering
that field fixed audit setup only. No training/data changes needed.

Fixed-candidate calibration: only2approved complete locked-vocabulary videos/4signs among
131boundarycalibration records. Original/fine-tuned/augmented BIO all0correct/100%WER;
both adaptededge paths1correct/75%WER/0insertions. Predeclared rule chooses firstedgepath,
not a recoveredBIOcheckpoint. Expanded72never used to override it or tune thresholds/epochs.
This is not sufficient recognition-selection evidence; no candidate promoted, no newtraining.
Bounded experiment was fixed-weight readout comparison with zero gradientsteps.

Replay{x['elapsed_seconds']:.2f}s; calibration{c['elapsed_seconds']:.2f}s. Isolated synchronizedMPS
batch1 normalization/CNN/attention/BIO firstadapted median{b['results'][1]['median_ms']:.2f}ms,
p95{b['results'][1]['p95_ms']:.2f}ms, excludesMediaPipe/Reel/camera and intrinsic500msfuture.
Per-stage whole-video timings in REPORT.md; notlive/iPhone evidence. Video/checkpoint/oldsource
hashes and all editcounts/retention recomputed. Next preserve originalBIO reference and resolve
complete recognition scoring inside existing calibration split before another fine-tune.

'''
p=root/'docs/ground_truth/live-streaming/log.md';s=p.read_text();pos=s.index('\n')+1;p.write_text(s[:pos]+'\n'+entry+s[pos:])
p=root/'PROJECT_GROUND_TRUTH.md';s=p.read_text();marker='## Pipeline state\n';pos=s.index(marker)+len(marker)
current=f'''\n\n**Boundary adaptation regression isolated 2026-09-22 — no promotion:**
Controlled72video/186sign swaps reproduce frozenBIO127correct/39.78%WER/15I exactly.
Firstadapted+originalBIO={a['correct']}correct/{a['wer']:.2%}WER/{a['insertions']}I,
retains{a['retained_baseline_correct_events']}/127; augmented+BIO={d['correct']}/{d['wer']:.2%}/{d['insertions']}I,
retains{d['retained_baseline_correct_events']}/127. CNN/norm/BIO tensors unchanged; major loss
comes from replacing readout/decoder, not wholesale representation damage. Offline evaluator
now permits originalBIO independently of checkpoint; defaults unchanged.42614target rows and
mask/clock/cache audit pass;23focusedtests. Existing calibration has only2complete in-vocabulary
videos/4signs: BIO0correct, adaptededges1. It cannot support robust recognition selection;
no expanded-set checkpoint selection, no newtraining/deployment/distillation. Next resolve
complete calibration scoring and preserve originalBIO in future adaptation comparisons.
Report: `artifacts/reports/boundary_adaptation_diagnostic_v17_20260922/REPORT.md`.
'''
p.write_text(s[:pos]+current+s[pos:])
p=root/'docs/ground_truth/live-streaming/high.md';s=p.read_text();pos=s.index('\n')+1
p.write_text(s[:pos]+'''\n\n## 2026-09-22 — boundary adaptation must preserve and control the BIO readout

OriginalBIO swaps recover most of the expanded recognition lost by START/END adaptation;
backbone, readout and decoder must be separated before attributing failure to representation
or capacity. Keep frozen pretrainedBIO as reference. Expanded72 is comparison only. Existing
boundary calibration offers only2complete locked-vocabulary sequences/4signs, insufficient
for robust whole-video checkpoint selection; resolve that scoring contract before new fitting.
No automatic deployment/distillation. Evidence: boundary_adaptation_diagnostic_v17_20260922/REPORT.md.
'''+s[pos:])
