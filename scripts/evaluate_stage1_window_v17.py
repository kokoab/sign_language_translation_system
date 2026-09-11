#!/usr/bin/env python3
"""Complete-stream Stage-1 window evaluation and fail-closed development gates."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from active.v17.stage1_window_v17 import (
    RAW_FORMAT, Stage1WindowTranscript, load_stage1_window_checkpoint,
    normalize_time_window, window_end_times, NO_EMIT,
)
from active.v17.train_stage_2_other_ctc_v17 import _edit_operations, sha256


def failed_gates(candidate, baseline):
    comparisons = {
        'connected_wer': lambda c, b: c <= .9 * b + 1e-10,
        'connected_insertions': lambda c, b: c <= b,
        'connected_deletions': lambda c, b: c <= b,
        'familiar_wer': lambda c, b: c <= b + 1e-10,
        'citizen_accuracy': lambda c, b: c >= b - .01 - 1e-10,
        'semlex_accuracy': lambda c, b: c >= b - .01 - 1e-10,
        'transition_false_emissions': lambda c, b: c <= b,
    }
    failures = []
    for name, predicate in comparisons.items():
        c, b = candidate.get(name), baseline.get(name)
        if c is None or b is None or not math.isfinite(c) or not math.isfinite(b) or not predicate(c, b):
            failures.append(name)
    delay = candidate.get('median_delay_seconds')
    if delay is None or not math.isfinite(delay) or delay > 1.:
        failures.append('median_delay_seconds')
    for name in ('matched_pool', 'complete_streaming', 'latency_includes_runtime'):
        if candidate.get(name) is not True:
            failures.append(name)
    return failures


def word_latency(events, updates):
    first = [None] * len(events)
    for update in updates:
        # Never align an early output to a future identical sign occurrence.
        available = [i for i, event in enumerate(events)
                     if float(event.get('start_seconds', event['end_seconds'])) <= update['seconds']]
        reference = [events[i]['label'] for i in available]
        for operation in _edit_operations(reference, update['hypothesis']):
            if operation['operation'] == 'match':
                index = available[operation['reference_position']]
                if first[index] is None:
                    first[index] = float(update['seconds'])
    delays = [seconds - float(event['end_seconds']) for event, seconds in zip(events, first) if seconds is not None]
    return dict(total_signs=len(events), missed_signs=first.count(None),
                missed_sign_frequency=first.count(None)/len(events) if events else None,
                median_delay_seconds=float(np.median(delays)) if delays else None,
                p95_delay_seconds=float(np.percentile(delays, 95)) if delays else None,
                first_correct_seconds=first, delays_seconds=delays)


def evaluate(checkpoint, manifest, output):
    torch.set_num_threads(2)
    model, labels, payload = load_stage1_window_checkpoint(checkpoint)
    pool = json.loads(manifest.read_text())
    rows = pool['rows']
    if any(row['role'] != 'validation' for row in rows):
        raise ValueError('evaluation requires frozen development rows only')
    results = []
    with torch.inference_mode():
        for row in rows:
            archive = ROOT / row['archive_path']
            if {'test', 'external_evaluation_reserved'} & set(archive.parts):
                raise ValueError('protected evaluation archive')
            if row.get('archive_sha256') != sha256(archive):
                raise ValueError('raw evaluation archive is not frozen or changed')
            with np.load(archive, allow_pickle=False) as z:
                if str(z['raw_format']) != RAW_FORMAT:
                    raise ValueError('wrong raw observation format')
                raw, times = z['raw_features'], z['timestamps_seconds']
            ends = window_end_times(times, include_final=True)
            features, valid, rejected = [], [], {}
            for index, end in enumerate(ends):
                try:
                    value, _ = normalize_time_window(raw, times, end)
                except ValueError as error:
                    rejected[index] = str(error)
                else:
                    features.append(value); valid.append(index)
            predictions = [NO_EMIT] * len(ends)
            probabilities = [None] * len(ends)
            started = time.perf_counter()
            for begin in range(0, len(features), 64):
                scores = model(torch.from_numpy(np.stack(features[begin:begin+64]))).softmax(-1).numpy()
                for index, score in zip(valid[begin:begin+64], scores):
                    predictions[index] = labels[int(score.argmax())]
                    probabilities[index] = float(score.max())
            elapsed = time.perf_counter() - started
            transcript, updates = Stage1WindowTranscript(), []
            for index, (end, label) in enumerate(zip(ends, predictions)):
                # A source gap is unavailable evidence, never continuous agreement.
                if index and end - ends[index-1] > .13 + 1e-6:
                    transcript.update(NO_EMIT, ends[index-1] + .001)
                hypothesis = transcript.update(label, end)
                updates.append(dict(seconds=end, hypothesis=hypothesis, prediction=label,
                                    probability=probabilities[index], rejection=rejected.get(index)))
            reference = row['reference']
            operations = Counter(o['operation'] for o in _edit_operations(reference, transcript.words))
            intervals = [x for x in row.get('intervals', []) if x['label'] in labels[:100]]
            latency = word_latency(intervals, updates) if row.get('verified_reference_intervals') else None
            results.append(dict(item_id=row['source_item_id'], source=row['source'],
                                reference=reference, final=transcript.words, operations=dict(operations),
                                updates=updates, source_latency=latency, batched_inference_seconds=elapsed))
    metrics = {}
    for source in sorted({r['source'] for r in results}):
        selected = [r for r in results if r['source'] == source]
        counts = Counter()
        for row in selected:
            counts.update(row['operations'])
        total = sum(len(row['reference']) for row in selected)
        edits = sum(counts[k] for k in ('substitution', 'deletion', 'insertion'))
        metrics[source] = dict(samples=len(selected), reference_tokens=total, wer_percent=100*edits/max(1,total),
                               substitutions=counts['substitution'], deletions=counts['deletion'], insertions=counts['insertion'])
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  manifest_sha256=sha256(manifest), epoch=payload['epoch'], metrics=metrics, rows=results,
                  isolated=payload['stage1_window'].get('isolated_validation_current'),
                  isolated_baseline=payload['stage1_window'].get('isolated_validation_baseline'),
                  latency_includes_runtime=False, selection=None, protected_test_accessed=False,
                  limitation='Complete scheduled classifier outputs including Finish tail; batched cached observations exclude live extraction/scheduling latency. Requires paced runtime verification before latency eligibility.')
    transition_root = manifest.parent/'transitions'
    transition_baseline = json.loads((transition_root/'baseline.json').read_text())
    if sha256(transition_root/'windows.npz') != transition_baseline['windows_sha256']:
        raise ValueError('frozen transition windows changed')
    with np.load(transition_root/'windows.npz', allow_pickle=False) as z, torch.inference_mode():
        predictions = model(torch.from_numpy(z['features'])).argmax(-1).tolist()
    report['transition_false_emissions'] = sum(p != 100 for p in predictions)
    report['transition_predictions'] = [labels[p] for p in predictions]
    report['complete_streaming'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)
    return report


def prepare_manifest(output):
    """Freeze all matched raw development inputs; local phrases supply no boundaries."""
    from scripts.diagnose_stage1_window_v17 import recording_observations
    from scripts.live_reel_continuous_v17 import parser as live_parser
    from scripts.extract_stage2_multimodal_v17 import safe_name
    from active.v17.stage1_window_v17 import raw_observation_features
    source_path = ROOT/'artifacts/reports/stage2_v17_live_matched_v1/frozen_inputs.json'
    annotation_path = ROOT/'artifacts/reports/stage2_v17_revisable_v1/supervision.json'
    annotations = json.loads(annotation_path.read_text())['items']
    source = json.loads(source_path.read_text())
    args = live_parser().parse_args(['--no-display', '--no-speech', '--naturalizer', 'literal'])
    rows = []
    for row in source['rows']:
        if row['role'] != 'validation':
            continue
        archive = ROOT/'data/local/stage1_window_v17/raw_observations/validation'/row['source']/f'{safe_name(row)}.stage1_window_raw_v17.npz'
        if not archive.exists():
            if row['source'] != 'local_phrases':
                raise ValueError('checked ASLLRP raw preparation is not complete')
            if sha256(ROOT/row['video_path']) != row['video_sha256']:
                raise ValueError('development video changed')
            observations, _ = recording_observations(dict(video=row['video_path']), args)
            raw, times = raw_observation_features(observations)
            archive.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(archive, raw_features=raw, timestamps_seconds=times,
                                raw_format=np.array(RAW_FORMAT), video_sha256=np.array(row['video_sha256']))
        annotation = annotations.get(row['source_item_id'], {})
        rows.append(dict(source_item_id=row['source_item_id'], source=row['source'], role='validation',
                         signer_id=row['signer_id'], archive_path=str(archive.relative_to(ROOT)),
                         archive_sha256=sha256(archive), video_path=row['video_path'], video_sha256=row['video_sha256'],
                         reference=[label for label in row['target_sequence'] if label != '__OTHER__'],
                         original_reference_including_oov=row['target_sequence'],
                         intervals=annotation.get('sign_intervals', []),
                         verified_reference_intervals=bool(annotation.get('full_clip_alignment_matches'))))
    payload = dict(format='stage1_window_matched_development_v17', rows=rows,
                   frozen_inputs_sha256=sha256(source_path), annotations_sha256=sha256(annotation_path),
                   protected_test_accessed=False)
    if output.exists() and json.loads(output.read_text()) != payload:
        raise ValueError('existing evaluation manifest changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2)+'\n')
    return payload


def prepare_transitions(output):
    """Compare both recognizers on exactly the same verified background support."""
    from active.v17.live_transition_supervision_v17 import interior_gaps
    from scripts.diagnose_stage1_window_v17 import recording_observations
    from scripts.live_reel_continuous_v17 import parser as live_parser
    from scripts.live_stage2_ctc_v17 import LiveStage2CTC
    manifest = ROOT/'artifacts/reports/stage1_window_v17/supervision_manifest.json'
    frozen = json.loads((ROOT/'artifacts/reports/stage2_v17_live_matched_v1/frozen_inputs.json').read_text())
    videos = {r['source_item_id']: r for r in frozen['rows']}
    args = live_parser().parse_args(['--no-display', '--no-speech', '--naturalizer', 'literal'])
    torch.set_num_threads(2)
    classifier = LiveStage2CTC(args)
    features, results = [], []
    for row in json.loads(manifest.read_text())['rows']:
        if row['role'] != 'validation':
            continue
        ends = []
        for left, right in interior_gaps(row['intervals'], .10):
            end = left + .53
            while end <= right + 1e-9:
                ends.append(end); end += .13
        if not ends:
            continue
        source = videos[row['source_item_id']]
        if sha256(ROOT/source['video_path']) != source['video_sha256']:
            raise ValueError('transition video changed')
        observations, _ = recording_observations(dict(video=source['video_path']), args)
        with np.load(ROOT/row['archive_path'], allow_pickle=False) as z:
            raw, times = z['raw_features'], z['timestamps_seconds']
        for end in ends:
            value, _ = normalize_time_window(raw, times, end)
            clip = [o for o in observations if end - .53 - 1e-9 <= o.seconds <= end + 1e-9]
            _, result = classifier.classify_window(clip, [])
            features.append(value)
            results.append(dict(item_id=row['source_item_id'], start_seconds=end-.53, end_seconds=end,
                                ctc_hypothesis=result['hypothesis'], ctc_false_emission=bool(result['hypothesis']),
                                diagnostics=result.get('diagnostics'), rejection_reasons=result.get('rejection_reasons', [])))
    if not features:
        raise ValueError('no checked background subset')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'windows.npz', features=np.stack(features))
    payload = dict(supervision_sha256=sha256(manifest), windows_sha256=sha256(output/'windows.npz'),
                   ctc_checkpoint_sha256=sha256(args.stage2_other_preservation), windows=results,
                   samples=len(results), ctc_false_emissions=sum(r['ctc_false_emission'] for r in results),
                   definition='Any known-sign output on a fresh 0.53-second window wholly within a guarded, all-sign-annotated gap. Both paths see the identical sampled frames; this is window-level rejection, not an independent idle-motion or OOV benchmark.')
    (output/'baseline.json').write_text(json.dumps(payload, indent=2)+'\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prepare-manifest', action='store_true')
    parser.add_argument('--prepare-transitions', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.prepare_transitions:
        prepare_transitions(args.output)
    elif args.prepare_manifest:
        prepare_manifest(args.output)
    elif args.checkpoint and args.manifest:
        evaluate(args.checkpoint, args.manifest, args.output)
    else:
        parser.error('--checkpoint and --manifest are required for evaluation')


if __name__ == '__main__':
    main()
