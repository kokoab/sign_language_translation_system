"""Expanded held-out boundary evaluation; both adapted checkpoints, one frozen Reel.

Adds the 60 genuinely held-out local phrase clips to the frozen 12-video ASLLRP
comparison. Subsets are summarized separately and never pooled across scoring
conventions without an explicit combined label. No training, no promotion.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.pretrained_boundary_v17 import UPSTREAM, PretrainedBoundary, load_pose, window_features, FPS, FRAMES, TARGET
from active.v17.temporal_boundary_v17 import BoundaryDecoder
from active.v17.approved_phrase_data_v17 import digest
from scripts.train_pretrained_boundary_v17 import PREPARED
from scripts.train_temporal_boundary_v17 import atomic, COMBINED
from scripts.evaluate_temporal_boundary_v17 import evaluation_rows, arguments, observations, edit_counts, summarize, transition_commits
from scripts.live_boundary_v17 import classify_interval
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector

REPORT = ROOT / 'artifacts/reports/boundary_expanded_eval_v17_20260922'
SPLIT = ROOT / 'artifacts/reports/local_familiar_signer_v17_20260922/split_manifest.json'
BASELINE = ROOT / 'artifacts/reports/asl_temporal_boundary_v17_20260922/baseline_oracle.json'

ARMS = [
    dict(name='frozen_pretrained_bio', checkpoint=None, recipe=None),
    dict(name='finetune_epoch7',
         checkpoint=ROOT / 'artifacts/models/pretrained_boundary_finetune_v17_20260922/seed_17621.pth',
         recipe=ROOT / 'active/v17/pretrained_boundary_recipe_20260922.json'),
    dict(name='augmented_epoch8',
         checkpoint=ROOT / 'artifacts/models/pretrained_boundary_augmented_v17_20260922/seed_17621.pth',
         recipe=ROOT / 'active/v17/pretrained_boundary_augmented_recipe_20260922_v2.json'),
]


def local_rows():
    """The 60 signer02 clips the current versioned split still holds out."""
    split = json.loads(SPLIT.read_text())
    if digest(ROOT / split['source_manifest']) != split['source_manifest_sha256']:
        raise ValueError('split source manifest changed')
    held = {r['video_sha256'] for r in split['records']
            if r.get('source') == 'local_phrases' and r['experiment_role'] == 'validation'}
    combined = json.loads(COMBINED.read_text())
    rows = [r for r in combined['records']
            if r['source'] == 'local_phrases' and r['video_sha256'] in held]
    if len(rows) != 60:
        raise ValueError('expected the60 held-out local phrase clips, got %d' % len(rows))
    for row in rows:
        if digest(ROOT / row['video_path']) != row['video_sha256']:
            raise ValueError('local evaluation video changed: ' + row['video_path'])
        # No curated boundary annotation for local phrases; interval metrics stay unavailable.
        row['all_events'] = []
        row['events'] = []
    return rows


def cached_inputs(rows, model, args, detector, cache_dir):
    """Frozen-CNN projections and Apple Vision observations, shared by every arm."""
    sys.path.insert(0, str(UPSTREAM))
    from prepare import extract
    prepared = {r['item']: r for r in json.loads(PREPARED.read_text())['records']}
    cache = {}
    with torch.inference_mode():
        for row in rows:
            item = row['source_item_id']
            record = prepared.get(item)
            if record is not None and record['video_sha256'] != row['video_sha256']:
                raise ValueError('cached pose/video identity mismatch: ' + item)
            if record is None:
                record = extract(dict(item=item, role='validation', video_path=row['video_path'],
                                      video_sha256=row['video_sha256'],
                                      intervals=[[e['start'], e['end']] for e in row['events']]), cache_dir)
            if digest(ROOT / record['pose_path']) != record['pose_sha256']:
                raise ValueError('evaluation pose changed: ' + item)
            pose = load_pose(record)
            projected = []
            tick = time.perf_counter()
            for start in range(0, len(pose.body.data), 16):
                x = np.stack([window_features(pose, j)[0]
                              for j in range(start, min(start + 16, len(pose.body.data)))])
                projected.append(model.project(torch.from_numpy(x).to('mps')).cpu())
            cache[item] = dict(projected=torch.cat(projected), observations=observations(row, args, detector),
                               projection_seconds=time.perf_counter() - tick, pose_sha256=record['pose_sha256'])
            print('prepared %d/%d %s' % (len(cache), len(rows), item), flush=True)
    return cache


def run_arm(arm, rows, cache, model, args, reel, times, decode, *, readout=None):
    """One decision per interval; readout can preserve BIO with adapted weights."""
    if readout is None:
        readout = 'edges' if arm['checkpoint'] else 'bio'
    if readout not in ('bio', 'edges'):
        raise ValueError('readout must be bio or edges')
    if arm['checkpoint'] is not None:
        state = torch.load(arm['checkpoint'], map_location='cpu', weights_only=False)
        if state['recipe_sha256'] != digest(arm['recipe']):
            raise ValueError('checkpoint does not match its recipe: ' + arm['name'])
        model.load_state_dict(state['model_state_dict'], strict=True)
        arm = {**arm, 'selected_epoch': state['epoch'], 'checkpoint_sha256': digest(arm['checkpoint'])}
    model.eval()
    records, matched, candidates = [], 0, 0
    for row in rows:
        cached = cache[row['source_item_id']]
        z = cached['projected']
        outputs = []
        tick = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(z), 128):
                h = model.encode_projected(z[start:start + 128].to('mps'), times)
                value = model.edge(h).sigmoid() if readout == 'edges' else model.backbone.sign_bio_head(h).log_softmax(-1)
                outputs.append(value.cpu())
        values = torch.cat(outputs)
        head_seconds = time.perf_counter() - tick
        if readout == 'edges':
            decoder = BoundaryDecoder()
            events = []
            for i, p in enumerate(values.numpy()):
                events.extend(decoder.update(i / FPS, p))
        else:
            events = [dict(start_seconds=s['start'] / FPS, end_seconds=s['end'] / FPS)
                      for s in decode['filter_segments'](decode['likeliest_probs_to_segments'](values))]
        predictions = [classify_interval(reel, cached['observations'], event, args, .1) for event in events]
        for p in predictions:
            p['uses_eof_partial_context'] = p['end_seconds'] + .5 > (len(z) - 1) / FPS
        hyp = [p['committed_gloss'] for p in predictions if p['committed_gloss']]
        unmatched = set(range(len(row['events'])))
        for event in events:
            choices = [i for i in unmatched
                       if abs(event['start_seconds'] - row['events'][i]['start']) <= .2
                       and abs(event['end_seconds'] - row['events'][i]['end']) <= .2]
            if choices:
                unmatched.remove(min(choices))
                matched += 1
        if row['events']:
            candidates += len(events)
        records.append(dict(id=row['source_item_id'], subset=row['subset'], reference=row['target_sequence'],
                            hypothesis=hyp, metrics=edit_counts(row['target_sequence'], hyp),
                            predictions=predictions, transitions=transition_commits(row, predictions),
                            head_seconds=head_seconds, projection_seconds=cached['projection_seconds'],
                            pose_sha256=cached['pose_sha256']))
    return arm, records, matched, candidates


def subset_summary(records, baseline=None, annotated=False):
    summary = summarize(records, baseline)
    summary['wholly_gap_contained_commits'] = sum(r['transitions']['wholly_gap_contained_commits'] for r in records)
    summary['committed_with_eof_partial_context'] = sum(
        bool(p['committed_gloss']) and p['uses_eof_partial_context'] for r in records for p in r['predictions'])
    if not annotated:
        summary['interval_metrics'] = 'unavailable; local phrases carry no curated boundary annotation'
    return summary


def evaluate():
    REPORT.mkdir(parents=True, exist_ok=True)
    cache_dir = REPORT / 'evaluation_poses'
    cache_dir.mkdir(exist_ok=True)
    sys.path.insert(0, str(UPSTREAM))
    from probe import upstream_helpers
    decode = upstream_helpers()

    asllrp = evaluation_rows(report=REPORT)
    for row in asllrp:
        row['subset'] = 'asllrp12'
    local = local_rows()
    for row in local:
        row['subset'] = 'local60'
    rows = asllrp + local
    base = json.loads(BASELINE.read_text())['results']['reel']

    args = arguments(rows[0]['video_path'], REPORT / 'evaluation_sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)
    model = PretrainedBoundary().to('mps').eval()
    times = torch.tensor((np.arange(FRAMES) - TARGET)[None] / FPS, dtype=torch.float32, device='mps')

    cache = cached_inputs(rows, model, args, detector, cache_dir)
    output = []
    for spec in ARMS:
        arm, records, matched, candidates = run_arm(spec, rows, cache, model, args, reel, times, decode)
        by_subset = {name: [r for r in records if r['subset'] == name] for name in ('asllrp12', 'local60')}
        summaries = dict(
            asllrp12=subset_summary(by_subset['asllrp12'], base, annotated=True),
            local60=subset_summary(by_subset['local60'], None, annotated=False),
            combined=subset_summary(records, None, annotated=False))
        summaries['asllrp12'].update(boundary_candidates=candidates, boundary_matches_200ms=matched,
                                     boundary_recall_200ms=matched / summaries['asllrp12']['references'])
        summaries['combined']['scope'] = 'explicitly labelled pooled figure; the two subsets use different sources and annotation coverage'
        output.append(dict(arm=arm['name'], checkpoint=str(arm['checkpoint'] or ''),
                           checkpoint_sha256=arm.get('checkpoint_sha256'), selected_epoch=arm.get('selected_epoch'),
                           summaries=summaries, records=records))
        atomic(REPORT / 'evaluation.json', dict(runs=output, reel=reel.provenance(), promoted=False,
              limitation='12 reused ASLLRP development videos plus the60 held-out local signer02 clips. '
                         'Familiar-signer reused development pool, not unseen-signer and not protected test. '
                         'Cached-pose offline composition, not live latency. Local phrases carry no curated '
                         'boundary annotation, so interval recall and gap commits are ASLLRP-only.'))
        print(json.dumps(dict(arm=arm['name'], **{k: v['wer'] for k, v in summaries.items()})), flush=True)

    if any(digest(ROOT / r['video_path']) != r['video_sha256'] for r in rows):
        raise ValueError('video changed during evaluation')

    lines = ['# Expanded held-out boundary evaluation', '',
             'No promotion. Frozen Reel, identical commit rules in every arm.',
             '12 reused ASLLRP development videos (24 signs) plus 60 held-out local signer02 clips (162 signs).',
             'Familiar-signer reused development pool; not unseen-signer, not protected test, not live latency.', '',
             '| Arm | Epoch | ASLLRP12 WER | local60 WER | Combined WER | Combined correct |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in output:
        s = r['summaries']
        lines.append('| %s | %s | %.2f%% | %.2f%% | %.2f%% | %d/%d |' % (
            r['arm'], r['selected_epoch'] or '—', s['asllrp12']['wer'] * 100, s['local60']['wer'] * 100,
            s['combined']['wer'] * 100, s['combined']['correct'], s['combined']['references']))
    lines += ['', 'Interval recall and gap-contained commits are ASLLRP-only; local phrases carry no curated',
              'boundary annotation. Retention is measured against the recorded Reel baseline on ASLLRP only.',
              'Subsets are summarized separately; the combined column pools two different sources and is labelled as such.',
              'See evaluation.json. Do not compare these figures against the older standalone 211-phrase convention.']
    (REPORT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    evaluate()
