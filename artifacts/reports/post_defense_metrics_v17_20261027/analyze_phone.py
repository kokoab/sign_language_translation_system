"""Score the iPhone 13 ABBA replays: sequence metrics, commit latency, frame cost, English timing, commit-score reliability.

Definitions (also written into phone_metrics.json):
- WER = (S + D + I) / reference signs, Levenshtein alignment per recording, pooled over recordings.
- Sequence precision = correct / output signs; sequence recall = correct / reference signs; F1 their harmonic mean.
- Commit latency = commit time - model-estimated sign end + compute time of the committing frame, for signs
  committed during the stream (the end-of-recording flush is excluded, as in score_live_reel_swift_v17.py).
- Frame compute = engine.process wall time per processed frame (landmarks, hand crops, MobileCLIP2, boundary,
  recognizer, decoder); camera capture/decoding, model loading, UI and English are outside it.
- Real-time factor = summed frame compute / recording duration.
"""
import json, subprocess
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
RES = D / 'phone_results'
STAGE = Path.home() / 'slt_post_defense_staging/post_defense/videos'
runs = json.loads((D / 'runs.json').read_text())
if (D / 'runs_every_frame.json').exists():
    runs += json.loads((D / 'runs_every_frame.json').read_text())
duration_cache = D / 'video_durations.json'
durations = json.loads(duration_cache.read_text()) if duration_cache.exists() else {}
for r in runs:
    for c in r['clips']:
        if c['file'] not in durations:
            durations[c['file']] = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                                                         str(STAGE / c['file'])], capture_output=True, text=True).stdout)
duration_cache.write_text(json.dumps(durations, indent=0))


def align(ref, hyp):
    """Levenshtein alignment; returns counts and per-hypothesis-word correctness."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int); d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, s, dl, ins, ok = n, m, 0, 0, 0, [False] * m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]: ok[j - 1] = True
            else: s += 1
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1: dl += 1; i -= 1
        else: ins += 1; j -= 1
    return dict(substitutions=s, deletions=dl, insertions=ins, correct=sum(ok), references=n, hypotheses=m), ok


def pct(a, q): return float(np.percentile(a, q)) if len(a) else None


def summarize(records, files):
    t = dict(substitutions=0, deletions=0, insertions=0, correct=0, references=0, hypotheses=0)
    exact, latency, compute, scores, hits = 0, [], [], [], []
    cap_upd, cap_fin, t5_app, t5_fin, t5_total, dur, proc, offered, therm = [], [], [], [], [], 0.0, 0, 0, []
    for rec, f in zip(records, files):
        c, ok = align(rec['reference'], rec['hypothesis'])
        for k in t: t[k] += c[k]
        exact += rec['reference'] == rec['hypothesis']
        for w, good in zip(rec['words'], ok):
            scores.append(w['score']); hits.append(good)
        for w in rec['words'][:len(rec['words']) - rec['tail_count']]:
            latency.append(w['commit_seconds'] - w['end_seconds'] + w['compute_seconds'])
        compute += [1000 * x for x in rec['compute_seconds']]
        cap_upd += rec['caption_update_ms']; cap_fin.append(rec['caption_finish_ms'])
        t5_app += rec['t5_append_ms']; t5_fin.append(rec['t5_finish_ms']); t5_total.append(sum(rec['t5_append_ms']) + rec['t5_finish_ms'])
        dur += durations[f]; proc += rec['frames_processed']; offered += rec['frames_offered']
        therm.append(max(rec['thermal_before'], rec['thermal_after']))
    p = t['correct'] / max(t['hypotheses'], 1); r = t['correct'] / max(t['references'], 1)
    lat = np.array(latency); comp = np.array(compute)
    return dict(
        recordings=len(records), **t, wer=(t['substitutions'] + t['deletions'] + t['insertions']) / t['references'] if t['references'] else None,
        precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0, exact_sequences=exact,
        commit_latency_s=dict(n=len(lat), median=pct(lat, 50), p90=pct(lat, 90), max=float(lat.max()) if len(lat) else None,
                              share_under_0_5=float((lat < .5).mean()) if len(lat) else None),
        frame_compute_ms=dict(frames=len(comp), median=pct(comp, 50), mean=float(comp.mean()), p90=pct(comp, 90), p95=pct(comp, 95), max=float(comp.max())),
        recording_seconds=dur, processed_fps=proc / dur, offered_fps=offered / dur, real_time_factor=float(comp.sum() / 1000 / dur),
        captions_ms=dict(update_median=pct(cap_upd, 50), update_p95=pct(cap_upd, 95), finish_median=pct(cap_fin, 50), finish_p95=pct(cap_fin, 95)),
        t5_ms=dict(append_median=pct(t5_app, 50), append_p95=pct(t5_app, 95), finish_median=pct(t5_fin, 50), finish_p95=pct(t5_fin, 95),
                   per_utterance_total_median=pct(t5_total, 50), per_utterance_total_p95=pct(t5_total, 95)),
        thermal_max=max(therm)), scores, hits


def reliability(scores, hits, bins=10):
    s, h = np.array(scores), np.array(hits, float)
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(s, edges[1:-1], right=True), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(h[m].mean() - s[m].mean())
            rows.append(dict(low=float(edges[b]), high=float(edges[b + 1]), n=int(m.sum()), precision=float(h[m].mean()), mean_score=float(s[m].mean())))
    return dict(n=len(s), ece=float(ece), mean_score=float(s.mean()), precision=float(h.mean()), bins=rows)


out = dict(definitions=__doc__, device='iPhone 13 (angelo), camera-paced replay through the Live path', runs={}, pooled={}, available=[])
pool = {}
for r in runs:
    p = RES / (r['run'] + '.json')
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    if not data.get('complete'):
        continue
    out['available'].append(r['run'])
    files = [c['file'] for c in r['clips']]
    if r['set'] == 'live':   # reference recorded in my_transition_recovery_v17_20261011/analyze_phone_comparison.py
        for x in data['records']:
            x['reference'] = 'MY FRIEND MY FRIEND MY NAME MY NAME MY FRIEND MY EAT MY FRIEND MY NAME'.split()
    assert [x['id'] for x in data['records']] == [c['id'] for c in r['clips']]
    s, sc, hi = summarize(data['records'], files)
    s.update(load_ms=data['load_ms'], peak_sampled_mb=data['peak_sampled_mb'], system_version=data['system_version'])
    out['runs'][r['run']] = s
    key = f"{r['set']}_{'every_' if r.get('mode') == 'every_frame' else ''}{r['arm']}"
    pool.setdefault(key, dict(records=[], files=[]))
    pool[key]['records'] += data['records']; pool[key]['files'] += files
both = [k for k in ('tune_fp16', 'test_fp16') if k in pool]
if both:
    pool['all_fp16'] = dict(records=sum((pool[k]['records'] for k in both), []), files=sum((pool[k]['files'] for k in both), []))
for key, v in pool.items():
    s, sc, hi = summarize(v['records'], v['files'])
    s['commit_score_reliability'] = reliability(sc, hi)
    out['pooled'][key] = s
(D / 'phone_metrics.json').write_text(json.dumps(out, indent=1) + '\n')
for key, s in out['pooled'].items():
    print(f"{key:10s} n={s['recordings']:3d} WER {100*s['wer'] if s['wer'] is not None else float('nan'):6.2f}  P {100*s['precision']:5.1f} R {100*s['recall']:5.1f} "
          f"exact {s['exact_sequences']}/{s['recordings']} | lat med {s['commit_latency_s']['median']} p90 {s['commit_latency_s']['p90']} | "
          f"frame med {s['frame_compute_ms']['median']:.2f} p95 {s['frame_compute_ms']['p95']:.2f} fps {s['processed_fps']:.1f} RTF {s['real_time_factor']:.3f} | "
          f"cap fin {s['captions_ms']['finish_median']:.2f} t5 utt {s['t5_ms']['per_utterance_total_median']:.1f} | thermal {s['thermal_max']} | score ECE {s['commit_score_reliability']['ece']:.3f}")
for name, s in out['runs'].items():
    print(f"  {name:16s} WER {100*s['wer'] if s['wer'] is not None else float('nan'):6.2f} frame med {s['frame_compute_ms']['median']:.2f} p95 {s['frame_compute_ms']['p95']:.2f} thermal {s['thermal_max']} mem {s['peak_sampled_mb']:.0f}")
