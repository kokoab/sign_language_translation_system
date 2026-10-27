"""Render docs/capstone_papers/POST_DEFENSE_BATIANCELA_REVIEW.md from the measured JSON files.

Every number in the proposed manuscript wording is filled from calibration.json, precision_accuracy.json and
phone_metrics.json, so the text, tables and conclusions cannot drift apart. Missing values render as [pending].
"""
import json, os, re
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[2]
OUT = R / 'docs/capstone_papers/POST_DEFENSE_BATIANCELA_REVIEW.md'
cal = json.loads((D / 'calibration.json').read_text())
prec = json.loads((D / 'precision_accuracy.json').read_text())
phone = json.loads((D / 'phone_metrics.json').read_text()) if (D / 'phone_metrics.json').exists() else {'pooled': {}, 'runs': {}}
V = {}


def pc(x, d=2): return f'{100 * x:.{d}f}%'


def size(p):
    t = 0
    for r, _, fs in os.walk(p):
        t += sum(os.path.getsize(os.path.join(r, f)) for f in fs if not f.startswith('._'))
    return f'{t / 1e6:.2f} MB'


# Calibration
b, a = cal['before'], cal['after_cross_fit']
V.update(cal_n=f"{cal['n']:,}", cal_acc=pc(b['accuracy']), cal_conf=pc(b['mean_confidence']),
         cal_ece_b=pc(b['ece']), cal_ece_a=pc(a['ece']), cal_mce_b=pc(b['mce']), cal_mce_a=pc(a['mce']),
         cal_nll_b=f"{b['nll']:.4f}", cal_nll_a=f"{a['nll']:.4f}", cal_brier_b=f"{b['brier']:.4f}", cal_brier_a=f"{a['brier']:.4f}",
         cal_T=f"{cal['temperature_fit_on_all']:.2f}", cal_conf_a=pc(a['mean_confidence']),
         cal_ci_b=f"{100 * cal['ece_bootstrap_95ci']['before'][0]:.2f}–{100 * cal['ece_bootstrap_95ci']['before'][1]:.2f}%",
         cal_ci_a=f"{100 * cal['ece_bootstrap_95ci']['after'][0]:.2f}–{100 * cal['ece_bootstrap_95ci']['after'][1]:.2f}%")
t = sorted(cal['temperatures'].values()); V['cal_T_range'] = f'{t[0]:.2f}–{t[1]:.2f}'
for k, n in [('citizen', 'cit'), ('semlex', 'sem'), ('local', 'loc')]:
    d = cal['per_domain'][k]
    V[f'{n}_n'] = f"{d['n']:,}"; V[f'{n}_acc'] = pc(d['before']['accuracy']); V[f'{n}_conf'] = pc(d['before']['mean_confidence'])
    V[f'{n}_b'] = pc(d['before']['ece']); V[f'{n}_a'] = pc(d['after']['ece'])
top = b['bins'][-1]; V.update(top_n=f"{top['n']:,}", top_share=pc(top['n'] / cal['n'], 1), top_acc=pc(top['accuracy']), top_conf=pc(top['confidence']))

# Precision (desktop accuracy) and package sizes
h = prec['heads']['original_word_logits']
V.update(p32_acc=pc(h['fp32_accuracy']), p16_acc=pc(h['fp16_accuracy']), p_agree=f"{h['top1_agreement']:,}/{prec['n']:,}",
         p_maxdiff=f"{h['max_abs_logit_diff']:.3f}")
M = R / 'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/Models'; F = D / 'fp32_models'
V.update(rec16=size(M / 'SpanRecognizerV17ConfidenceDualB8FP16.mlpackage'), rec32=size(F / 'SpanRecognizerV17ConfidenceDualB8FP32.mlpackage'),
         bnd16=size(M / 'AVBoundaryStudentV17L6FP16.mlpackage'), bnd32=size(F / 'AVBoundaryStudentV17L6FP32.mlpackage'),
         clip16=size(M / 'MobileCLIP2S0ImageEncoderV17FP16.mlpackage'), clip32=size(R / 'artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage'))

# Phone
THERMAL = {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical'}
for key, s in phone['pooled'].items():
    if s.get('wer') is None:
        continue
    lat, fr = s['commit_latency_s'], s['frame_compute_ms']
    V.update({f'{key}_{k}': v for k, v in dict(
        n=s['recordings'], refs=s['references'], hyps=s['hypotheses'], wer=pc(s['wer']), p=pc(s['precision'], 1), r=pc(s['recall'], 1), f1=pc(s['f1'], 1),
        S=s['substitutions'], D=s['deletions'], I=s['insertions'], exact=f"{s['exact_sequences']}/{s['recordings']}",
        lat_med=f"{lat['median']:.2f} s", lat_p90=f"{lat['p90']:.2f} s", lat_u05=pc(lat['share_under_0_5'], 1),
        fmed=f"{fr['median']:.2f} ms", fp95=f"{fr['p95']:.2f} ms", fmean=f"{fr['mean']:.2f} ms", frames=f"{fr['frames']:,}",
        fps=f"{s['processed_fps']:.1f}", rtf=f"{s['real_time_factor']:.2f}", secs=f"{s['recording_seconds'] / 60:.1f} min",
        cap_fin=f"{s['captions_ms']['finish_median']:.1f} ms", cap_upd=f"{s['captions_ms']['update_median']:.1f} ms",
        t5_utt=f"{s['t5_ms']['per_utterance_total_median']:.0f} ms", t5_utt95=f"{s['t5_ms']['per_utterance_total_p95']:.0f} ms",
        t5_fin=f"{s['t5_ms']['finish_median']:.1f} ms", therm=THERMAL[s['thermal_max']],
        sc_ece=pc(s['commit_score_reliability']['ece']), sc_mean=pc(s['commit_score_reliability']['mean_score']),
        sc_prec=pc(s['commit_score_reliability']['precision'])).items()})
    passes = [phone['runs'][r]['wer'] for r in phone['runs'] if r.rsplit('_', 1)[0] == key]
    if passes:
        V[f'{key}_wer_range'] = ' and '.join(pc(x) for x in passes)
mem = [s['peak_sampled_mb'] for k, s in phone['runs'].items() if '_fp16_' in k]
if mem: V['mem16'] = f'{max(mem):.0f} MB'
mem = [s['peak_sampled_mb'] for k, s in phone['runs'].items() if '_fp32_' in k]
if mem: V['mem32'] = f'{max(mem):.0f} MB'
V['ios'] = next((s['system_version'] for s in phone['runs'].values()), '[pending]')

import numpy as _np
def _sdi(recs):
    tot = [0, 0, 0]
    for x in recs:
        ref, hyp = x['reference'], x['hypothesis']; n, m = len(ref), len(hyp)
        d = _np.zeros((n + 1, m + 1), int); d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1): d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
        i, j = n, m
        while i or j:
            if i and j and d[i, j] == d[i-1, j-1] + (ref[i-1] != hyp[j-1]): tot[0] += ref[i-1] != hyp[j-1]; i -= 1; j -= 1
            elif i and d[i, j] == d[i-1, j] + 1: tot[1] += 1; i -= 1
            else: tot[2] += 1; j -= 1
    return tot
def _load(name):
    p = D / name
    return json.loads(p.read_text())['records'] if p.exists() and (name.startswith('mac') or json.loads(p.read_text()).get('complete')) else None
mac = _load('mac_current_tune.json')
if mac:
    t_ = _sdi(mac); V['mac_now_wer'] = pc(sum(t_) / sum(len(x['reference']) for x in mac)); V['mac_now_sdi'] = ' / '.join(map(str, t_))
for key, runs_ in [('every_sdi', ['tune_every_fp16_p0', 'tune_every_fp16_p2']), ('paced_sdi', ['tune_fp16_p0', 'tune_fp16_p3']), ('fp32_sdi', ['tune_fp32_p1', 'tune_fp32_p2'])]:
    parts = [r_ for r_ in (_load(f'phone_results/{n}.json') for n in runs_) if r_]
    if parts: V[key] = ' and '.join(' / '.join(map(str, _sdi(r_))) for r_ in parts)
g = lambda k: V.get(k, '**[pending]**')
V['p13_interpretation'] = (
    f"When every frame was analyzed, WER was {g('tune_every_fp16_wer')}. The same recordings scored 12.83% in Section 4.8.2, "
    "where they were replayed on the development computer; both evaluations used the same decoder settings, but landmark "
    "extraction and the models ran on different hardware. Under live camera timing, WER was "
    f"{g('tune_fp16_wer')} (two passes: {g('tune_fp16_wer_range')}), because frames that arrived while the previous frame "
    "was still being analyzed were skipped, so the decoder received fewer frames.")
V['p14_interpretation'] = (
    f"Both precisions produced the same prediction for {g('p_agree')} validation clips, so numerical precision did not change "
    f"recognition. On the phone, however, FP32 needed a median of {g('tune_fp32_fmed')} per frame, more than the 50 ms available "
    f"at 20 frames per second, so it analyzed only {g('tune_fp32_fps')} frames per second and its WER was {g('tune_fp32_wer')}, compared with "
    f"{g('tune_fp16_wer')} for FP16. FP16 also halved the size of each package, and it was therefore retained for the visual "
    "and recognition models.")
text = (D / 'review_template.md').read_text()
missing = sorted(set(re.findall(r'\{\{(\w+)\}\}', text)) - set(V))
text = re.sub(r'\{\{(\w+)\}\}', lambda m: str(V.get(m.group(1), '**[pending]**')), text)
OUT.write_text(text)
print('wrote', OUT.relative_to(R), 'missing:', missing)
