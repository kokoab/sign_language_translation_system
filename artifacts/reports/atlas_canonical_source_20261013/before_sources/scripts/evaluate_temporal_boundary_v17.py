"""Paired whole-video Reel / reviewed-interval / learned-boundary development replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from scripts.train_temporal_boundary_v17 import COMBINED, CURATED, REPORT, atomic


def edit_counts(reference, hypothesis):
    """Levenshtein counts plus matched reference positions for paired retention."""
    table = [[None] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    table[0][0] = (0, 0, 0, [])
    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == j == 0: continue
            choices = []
            if i and j:
                s, d, ins, matched = table[i - 1][j - 1]
                equal = reference[i - 1] == hypothesis[j - 1]
                choices.append((s + int(not equal), d, ins, matched + ([(i - 1, j - 1)] if equal else [])))
            if i:
                s, d, ins, matched = table[i - 1][j]
                choices.append((s, d + 1, ins, matched))
            if j:
                s, d, ins, matched = table[i][j - 1]
                choices.append((s, d, ins + 1, matched))
            table[i][j] = min(choices, key=lambda c: (sum(c[:3]), -len(c[3])))
    s, d, ins, matched = table[-1][-1]
    return dict(substitutions=s, deletions=d, insertions=ins, references=len(reference),
                correct=len(matched), matched_pairs=matched, wer=(s + d + ins) / len(reference) if reference else None)


def evaluation_rows(report=REPORT):
    canonical = verify_manifest()
    combined, curated = (json.loads(p.read_text()) for p in (COMBINED, CURATED))
    events = defaultdict(list)
    for e in curated['events']:
        if e['source'] == 'asllrp_contiguous':
            events[e['item']].append(e)
    rows = [r for r in combined['records'] if r['source'] == 'asllrp_contiguous' and r['role'] == 'validation']
    if len(rows) != 12:
        raise ValueError('frozen comparison requires the12 approved ASLLRP development videos')
    for row in rows:
        row['all_events'] = sorted(events[row['source_item_id']], key=lambda e: e['start'])
        row['events'] = [e for e in row['all_events'] if e['kind'] == 'known']
        if [e['label'] for e in row['events']] != row['target_sequence']:
            raise ValueError('whole-video transcript/annotation mismatch')
        if digest(ROOT / row['video_path']) != row['video_sha256']:
            raise ValueError('evaluation video changed')
    atomic(report / 'evaluation_membership.json', dict(canonical=canonical,
           videos=[{k: r[k] for k in ('source_item_id', 'video_path', 'video_sha256', 'target_sequence')} for r in rows],
           scope='reused12-video development set; not protected test or independent phone evidence',
           inputs={str(p.relative_to(ROOT)): digest(p) for p in (COMBINED, CURATED)}))
    return rows


def arguments(video, output):
    from scripts.app_shell_v17 import parser
    return parser().parse_args(['--video', str(ROOT / video), '--output-root', str(output),
                               '--no-display', '--no-speech', '--naturalizer', 'literal', '--no-ollama',
                               '--no-stage2-arbiter', '--no-finish-gesture', '--realtime-video'])


def observations(row, args, detector):
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    capture = cv2.VideoCapture(str(ROOT / row['video_path']))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        capture.release(); raise ValueError('cannot decode evaluation video')
    obs, index, deadline = [], 0, 0.
    wrists = {'left': None, 'right': None}
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            seconds = index / fps; index += 1
            if seconds + 1e-6 < deadline: continue
            obs.append(observe_stage2_frame(frame, seconds, len(obs), detector, wrists, args))
            deadline = max(deadline + 1 / args.processing_fps, seconds)
    finally:
        capture.release()
    return obs


def summarize(records, baseline=None):
    counts = {k: sum(r['metrics'][k] for r in records) for k in ('substitutions', 'deletions', 'insertions', 'references', 'correct')}
    counts['wer'] = sum(counts[k] for k in ('substitutions', 'deletions', 'insertions')) / counts['references']
    counts['exact_videos'] = sum(r['hypothesis'] == r['reference'] for r in records)
    counts['videos'] = len(records)
    if baseline is not None:
        old = {r['id']: {i for i, _ in r['metrics']['matched_pairs']} for r in baseline}
        counts['baseline_correct_events'] = sum(len(v) for v in old.values())
        counts['retained_baseline_correct_events'] = sum(len(old[r['id']] & {i for i, _ in r['metrics']['matched_pairs']}) for r in records)
        counts['retention_note'] = 'transcript-aligned reference positions; temporal event matching is separately required for repeated identical glosses'
    return counts


def transition_commits(row, predictions):
    """Count only commits wholly inside guarded known-to-known annotation gaps."""
    events = row['all_events']
    gaps = []
    for left, right in zip(events, events[1:]):
        start, end = left['end'] + .02, right['start'] - .02
        if left['kind'] == right['kind'] == 'known' and end > start and not any(e['start'] < end and e['end'] > start for e in events):
            gaps.append((start, end))
    count = sum(bool(p.get('committed_gloss')) and not p.get('ignored_after_reset', False) and
                any(p['start_seconds'] >= start and p['end_seconds'] <= end for start, end in gaps)
                for p in predictions)
    return dict(guarded_annotation_gaps=len(gaps), wholly_gap_contained_commits=count,
                note='Mixed sign/transition intervals are not attributed as pure transition errors.')


def baseline_and_oracle():
    from active.v17.extract_v17 import AppleVisionDetector
    from scripts.live_reel_stage1_v17 import run, build_components
    from scripts.live_boundary_v17 import classify_interval
    rows = evaluation_rows()
    args = arguments(rows[0]['video_path'], REPORT / 'baseline_sessions')
    components = build_components(args)
    detector = AppleVisionDetector(args.minimum_point_confidence)
    results = defaultdict(list)
    for row in rows:
        args.video = ROOT / row['video_path']
        baseline = run(args, prebuilt=components)
        reference = row['target_sequence']
        results['reel'].append(dict(id=row['source_item_id'], reference=reference, hypothesis=baseline['hypothesis'],
                                     history=baseline['history'], metrics=edit_counts(reference, baseline['hypothesis'])))
        obs = observations(row, args, detector)
        for trim in (True, False):
            reel = components['classifier']
            reel.args.no_motion_trim = not trim
            reel.full.args.no_motion_trim = not trim
            for context in (0., .1):
                predictions = [classify_interval(reel, obs, dict(start_seconds=e['start'], end_seconds=e['end']), args, context)
                               for e in row['events']]
                hypothesis = [p['committed_gloss'] for p in predictions if p['committed_gloss']]
                supported = [p for p in predictions if 'verifier' in p]
                label_correct = sum(p.get('verifier', {}).get('candidate_gloss') == e['label'] for p, e in zip(predictions, row['events']))
                name = f'oracle_trim_{int(trim)}_context_{int(context * 1000)}'
                results[name].append(dict(id=row['source_item_id'], reference=reference, hypothesis=hypothesis,
                                         predictions=predictions, verifier_correct=label_correct, supported_intervals=len(supported),
                                         metrics=edit_counts(reference, hypothesis)))
        # Restore the default before the next whole-video baseline.
        components['classifier'].args.no_motion_trim = False
        components['classifier'].full.args.no_motion_trim = False
        atomic(REPORT / 'baseline_oracle_progress.json', dict(completed_videos=len(results['reel']), results=dict(results)))
    summaries = {name: summarize(records, results['reel']) for name, records in results.items()}
    for name, records in results.items():
        if name != 'reel':
            summaries[name]['verifier_correct'] = sum(r['verifier_correct'] for r in records)
            summaries[name]['supported_intervals'] = sum(r['supported_intervals'] for r in records)
    value = dict(summary=summaries, results=dict(results), models=components['classifier'].provenance(),
                 limitations=['reused12-video/24-sign development evaluation', 'oracle uses annotation timing and is diagnostic only',
                              'conditional oracle commit differs from live stability scheduling',
                              'no new low-motion or real hold/repeat coverage', 'no iPhone latency claim'])
    atomic(REPORT / 'baseline_oracle.json', value)
    print(json.dumps(summaries), flush=True)


def learned():
    from scripts.live_boundary_v17 import BoundaryRecognizer, run
    rows = evaluation_rows()
    baseline = json.loads((REPORT / 'baseline_oracle.json').read_text())['results']['reel']
    training = json.loads((REPORT / 'training_results.json').read_text())
    output = []
    shared_reel = None
    for trained in training['runs']:
        if digest(ROOT / trained['checkpoint']) != trained['checkpoint_sha256']:
            raise ValueError('trained checkpoint changed')
        args = arguments(rows[0]['video_path'], REPORT / ('learned_sessions_' + Path(trained['checkpoint']).stem))
        args.boundary_checkpoint = ROOT / trained['checkpoint']
        recognizer = BoundaryRecognizer(args, reel=shared_reel)
        shared_reel = recognizer.reel
        records, latency, compute, boundary_matches, boundary_total = [], [], [], 0, 0
        for row in rows:
            args.video = ROOT / row['video_path']; recognizer.reset()
            result = run(args, model=recognizer)
            history = json.loads(Path(result['history']).read_text())
            metrics = edit_counts(row['target_sequence'], result['hypothesis'])
            committed = [p for p in history['predictions'] if p['committed_gloss'] and not p['ignored_after_reset']]
            for ref_index, hyp_index in metrics['matched_pairs']:
                latency.append(committed[hyp_index]['completed_elapsed_seconds'] - row['events'][ref_index]['end'])
            candidates = [event for tick in history['boundary_trace'] for event in tick['events']]
            # One-to-one paired start/end match, rather than best-IoU double counting.
            unmatched = set(range(len(row['events'])))
            for event in candidates:
                possible = [i for i in unmatched if abs(event['start_seconds'] - row['events'][i]['start']) <= .2 and abs(event['end_seconds'] - row['events'][i]['end']) <= .2]
                if possible:
                    choice = min(possible, key=lambda i: abs(event['start_seconds'] - row['events'][i]['start']) + abs(event['end_seconds'] - row['events'][i]['end']))
                    unmatched.remove(choice); boundary_matches += 1
            boundary_total += len(candidates)
            compute.extend(t['compute_seconds'] for t in history['boundary_trace'])
            records.append(dict(id=row['source_item_id'], reference=row['target_sequence'], hypothesis=result['hypothesis'], metrics=metrics,
                                transitions=transition_commits(row, history['predictions']), history=result['history']))
        summary = summarize(records, baseline)
        summary.update(boundary_candidates=boundary_total, boundary_matches_200ms=boundary_matches,
                       known_region_match_fraction_200ms=boundary_matches / boundary_total if boundary_total else 0.,
                       boundary_recall_200ms=boundary_matches / summary['references'],
                       correct_word_end_to_output_p50_seconds=float(np.median(latency)) if latency else None,
                       correct_word_end_to_output_p95_seconds=float(np.percentile(latency, 95)) if latency else None,
                       wholly_gap_contained_commits=sum(r['transitions']['wholly_gap_contained_commits'] for r in records),
                       boundary_cpu_p95_ms=float(np.percentile(compute, 95) * 1000) if compute else None)
        output.append(dict(training=trained, summary=summary, records=records))
        atomic(REPORT / 'learned_results.json', dict(runs=output, promoted=False,
               limitation='Reused development set; unknown/low-motion/real repeat/hold and independent iPhone gates remain unmeasured.'))
        print(json.dumps(summary), flush=True)


def write_report():
    baseline = json.loads((REPORT / 'baseline_oracle.json').read_text())
    learned_result = json.loads((REPORT / 'learned_results.json').read_text())
    lines = ['# ASL temporal boundary experiment', '',
             'Implemented the approved plan as a separate candidate. Default Reel is unchanged.', '',
             '## Fixed contract', '',
             'Continuous20Hz Apple Vision raw observations,64-hidden temporal convolution,4-frame future context. '
             'Two start/end outputs; no wrist activation, no motion trimming for completed intervals, no fabricated end at EOF. '
             'No physical background or inside classifier is trained from uncertified annotation gaps.', '',
             '4,310deduplicated reviewed-timing events(3,518train/792validation) across1,121raw sequences. '
             'ASLLRP known/OOV timing supplies edges without lexical targets; current strict O5S5 positives remain positive-only. '
             'A parent-held subset of training selects epochs. Two seeds and skeleton/hand-relative-geometry arms share the same inputs.', '',
             '## Paired whole-video results', '',
             'The same12previously used ASLLRP development videos,24known reference signs. '
             'Unchanged Reel uses its actual asynchronous video scheduler. Oracle intervals are diagnostic and bypass proposal stability. '
             'Learned arms use the shared app runtime. All frames of each video are replayed; the last0.2s has no future context and remains unscored by the boundary head.', '',
             '| Arm | Correct | S / D / I | Known WER | Retained baseline correct |',
             '| --- | ---: | --- | ---: | ---: |']
    comparisons = [('Reel', baseline['summary']['reel']), ('Reviewed timing +100ms, no wrist trim', baseline['summary']['oracle_trim_0_context_100'])]
    comparisons.extend((f"Seed{r['training']['seed']}, hand geometry={r['training']['hand_geometry']}", r['summary']) for r in learned_result['runs'])
    for name, s in comparisons:
        lines.append(f"| {name} | {s['correct']}/{s['references']} | {s['substitutions']} / {s['deletions']} / {s['insertions']} | {100*s['wer']:.2f}% | {s.get('retained_baseline_correct_events', s['correct'])}/{s.get('baseline_correct_events', s['correct'])} |")
    lines += ['', 'Retention uses transcript-aligned reference positions; no real repeated-sign recording is present to establish temporal repeat accuracy. '
              'Boundary region match counts, guarded-gap commits and headless end-to-output timings are in learned_results.json. '
              'These are desktop replay timings, not sustained iPhone end-to-display performance.', '',
              '## Decision and remaining gates', '',
              '**No automatic promotion.** This reused tiny development set has no independently reviewed low-motion/hold/repeat/OOV stress coverage. '
              'Unit checks establish decoder mechanics, not recognition of real holds or repeats. Boundary losses do not establish sign accuracy.', '',
              f"With reviewed intervals+100ms the verifier identifies {baseline['summary']['oracle_trim_0_context_100']['verifier_correct']}/24signs; "
              'even idealized timing leaves identity/commit errors. Contextual identity adaptation remains indicated for review, '
              'but the earlier combined-data failure prevents treating another generic mixed run as an established remedy. '
              'Use the saved per-interval proposal/verifier evidence to specify that next bounded intervention; no backbone swap or threshold sweep was launched.', '',
              'All source roles, hashes, generic phrase gates and protected test sealing are preserved. No data acquired.', '',
              '## Run', '',
              '```sh', 'venv/bin/python scripts/app_shell_v17.py --boundary-checkpoint artifacts/models/asl_temporal_boundary_v17_20260922/seed_17621_geometry_1.pth', '```', '',
              'The command explicitly opts into a candidate; it is not a recommendation to replace default Reel. '
              'The representative checkpoint is named by the fixed seed/arm, not picked from these validation outcomes.', '']
    (REPORT / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['baseline', 'learned'])
    torch.set_num_threads(2)
    if parser.parse_args().action == 'baseline': baseline_and_oracle()
    else: learned()
