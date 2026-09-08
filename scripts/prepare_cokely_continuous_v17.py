#!/usr/bin/env python3
"""Audit and cut maximal locked100 runs from the public Cokely ASL corpus."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


SIGNERS = {
    1: 'DAVID_HAMILTON',
    2: 'PATRICK_GRAYBILL',
    3: 'PATRICK_GRAYBILL',
    4: 'MARK_MORALES',
    5: 'MJ_BIENVENU',
    6: 'PATRICK_GRAYBILL',
}

STREAMS = {
    1: 'https://s3.amazonaws.com/bepress-streaming-outbox-production/article_files/8e/b1/8b/8eb18b6b2f01a5a179100b0acd84a639b5f2a0ee95fb0195d477b946217c6bb3_2.ts.m3u8',
    2: 'https://s3.amazonaws.com/bepress-streaming-outbox-production/article_files/f3/55/5e/f3555e5d9d2ddea48e8c57b4da72ebca61d47513d1bd39b145aefe2accb7a7aa_2.ts.m3u8',
    3: 'https://s3.amazonaws.com/bepress-streaming-outbox-production/article_files/1c/fc/23/1cfc23608bc5a533a51d5758dacc82c048b2e0797e3bd8371853fc9f82a073d4_2.ts.m3u8',
    4: 'https://s3.amazonaws.com/bepress-streaming-outbox-production/article_files/71/ac/32/71ac328568cbd92e50dccad08944d895515800e1ad40b38aaa947b914848cd47_2.ts.m3u8',
    5: 'https://s3.amazonaws.com/bepress-streaming-outbox-production/article_files/4e/5f/10/4e5f10d2f2b853ddf89576179dd4c8c6814d5772af2d572ddb9338a69290520b_2.ts.m3u8',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate:format=duration',
        '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'frame_rate': stream['avg_frame_rate'],
            'duration_seconds': float(data['format']['duration'])}


def read_events(path):
    root = ET.parse(path).getroot()
    header = root.find('HEADER')
    if root.tag != 'ANNOTATION_DOCUMENT' or header is None or header.get('TIME_UNITS') != 'milliseconds':
        raise ValueError(f'{path.name}: expected millisecond EAF')
    times = {slot.get('TIME_SLOT_ID'): slot.get('TIME_VALUE')
             for slot in root.findall('./TIME_ORDER/TIME_SLOT')}
    tiers = {tier.get('TIER_ID'): tier for tier in root.findall('TIER')}
    if 'ASL-individual-cp' not in tiers:
        raise ValueError(f'{path.name}: expected one ASL-individual-cp tier')
    resolved = {}

    def parse_tier(tier_id):
        events = []
        for wrapper in tiers.get(tier_id, ()).findall('./ANNOTATION') if tier_id in tiers else ():
            annotation = next(iter(wrapper), None)
            if annotation is None:
                continue
            annotation_id = annotation.get('ANNOTATION_ID')
            if annotation.tag == 'ALIGNABLE_ANNOTATION':
                start = times.get(annotation.get('TIME_SLOT_REF1'))
                end = times.get(annotation.get('TIME_SLOT_REF2'))
                interval = (int(start), int(end)) if start and end else (None, None)
            else:
                interval = resolved.get(annotation.get('ANNOTATION_REF'), (None, None))
            resolved[annotation_id] = interval
            events.append({
                'annotation_id': annotation_id,
                'tier_id': tier_id,
                'raw_gloss': annotation.findtext('ANNOTATION_VALUE', '').strip(),
                'start_ms': interval[0],
                'end_ms': interval[1],
            })
        return events

    utterances = parse_tier('ASL-TT')
    events = parse_tier('ASL-individual-cp')
    supplementary = parse_tier('ASL-right-hand') + parse_tier('ASL-left-hand')
    for event in events:
        containing = [item for item in utterances if valid_interval(event) and
                      valid_interval(item) and item['start_ms'] <= event['start_ms'] and
                      item['end_ms'] >= event['end_ms']]
        event['utterance_id'] = containing[0]['annotation_id'] if len(containing) == 1 else None
    media = header.find('MEDIA_DESCRIPTOR')
    return {
        'main': events,
        'supplementary': supplementary,
        'utterances': utterances,
        'annotated_media': media.get('RELATIVE_MEDIA_URL') if media is not None else '',
    }


def valid_interval(event):
    return (event.get('start_ms') is not None and event.get('end_ms') is not None and
            0 <= event['start_ms'] < event['end_ms'])


def overlaps(left, right):
    return (valid_interval(left) and valid_interval(right) and
            left['start_ms'] < right['end_ms'] and right['start_ms'] < left['end_ms'])


def rejection_reason(event, locked, supplementary):
    if event['raw_gloss'] not in locked:
        return 'main_gloss_not_exact_locked100'
    if not valid_interval(event):
        return 'main_timing_missing_or_invalid'
    if 'utterance_id' in event and event['utterance_id'] is None:
        return 'parent_utterance_missing_or_ambiguous'
    conflicts = [item for item in supplementary if overlaps(event, item) and
                 (not valid_interval(item) or item['raw_gloss'] != event['raw_gloss'])]
    if conflicts:
        return 'conflicting_supplementary_hand_annotation'
    return None


def exact_runs(events, locked, max_gap_ms=300, supplementary=()):
    runs, run = [], []
    for event in events:
        valid = rejection_reason(event, locked, supplementary) is None
        boundary = run and (event['start_ms'] - run[-1]['end_ms'] > max_gap_ms or
                            event.get('utterance_id') != run[-1].get('utterance_id') or
                            any(overlaps(item, {
                                'start_ms': run[-1]['end_ms'], 'end_ms': event['start_ms']
                            }) for item in supplementary))
        if not valid or boundary:
            if len(run) >= 2:
                runs.append(run)
            run = []
        if valid:
            run.append(event)
    if len(run) >= 2:
        runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-root', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--clip-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('active/v17/citizen100_manifest.json'))
    parser.add_argument('--max-gap-ms', type=int, default=300)
    args = parser.parse_args()
    classes = json.loads(args.manifest.read_text())['classes']
    locked = {item['citizen_raw_gloss']: item for item in classes}
    if len(classes) != 100 or len(locked) != 100:
        raise ValueError('expected frozen100 manifest')
    rows, decisions = [], []
    for eaf in sorted(args.annotation_root.glob('cokely_*.eaf')):
        recording = int(eaf.stem.split('_')[1])
        parsed = read_events(eaf)
        events = parsed['main']
        runs = exact_runs(events, set(locked), args.max_gap_ms, parsed['supplementary'])
        accepted_ids = {event['annotation_id'] for run in runs for event in run}
        for event in events:
            reason = rejection_reason(event, set(locked), parsed['supplementary'])
            if reason is None and event['annotation_id'] not in accepted_ids:
                reason = 'isolated_locked100_token_not_a_phrase_run'
            decisions.append({
                'annotation_file': eaf.name,
                'recording_id': recording,
                'annotation_id': event['annotation_id'],
                'raw_gloss': event['raw_gloss'],
                'start_ms': event['start_ms'],
                'end_ms': event['end_ms'],
                'status': 'candidate' if event['annotation_id'] in accepted_ids else 'rejected',
                'reason': None if event['annotation_id'] in accepted_ids else reason,
            })
        if not runs:
            continue
        video = args.video_root / f'cokely_{recording}_360p.mp4'
        if not video.is_file():
            raise FileNotFoundError(video)
        for run_index, run in enumerate(runs):
            rows.append({
                    'source': 'cokely_verified',
                    'source_corpus': 'cokely_parallel_corpus_v1',
                    'role': 'candidate_verification',
                    'source_item_id': f'cokely:{recording}:run:{run_index}',
                    'license': 'CC BY-NC-SA 4.0',
                    'recording_id': recording,
                    'signer_id': SIGNERS[recording],
                    'source_page': f'https://encompass.eku.edu/cokely_videos/{recording}/',
                    'source_stream_url': STREAMS[recording],
                    'annotation_file': eaf.name,
                    'annotation_sha256': sha256(eaf),
                    'annotated_media': parsed['annotated_media'],
                    'source_video': str(video),
                    'run_index': run_index,
                    'start_ms': run[0]['start_ms'],
                    'end_ms': run[-1]['end_ms'],
                    'raw_glosses': [event['raw_gloss'] for event in run],
                    'canonical_glosses': [locked[event['raw_gloss']]['canonical_label'] for event in run],
                    'target_sequence': [locked[event['raw_gloss']]['canonical_label'] for event in run],
                    'target_indices': [locked[event['raw_gloss']]['class_index'] for event in run],
                    'target_asl_lex_codes': [
                        locked[event['raw_gloss']]['citizen_asl_lex_code'] for event in run
                    ],
                    'target_token_count': len(run),
                    'token_intervals_ms': [
                        [event['start_ms'], event['end_ms']] for event in run
                    ],
                    'annotation_ids': [event['annotation_id'] for event in run],
                    'source_group': f'cokely_recording_{recording}',
                    'signer_metadata_available': True,
                    'zero_lip_nodes': False,
                    'lip_supervision': 'source_face_visible',
                    'variant_contract': 'exact raw label only; cross-corpus visual variant review pending',
                    'variant_verified': False,
                    'training_eligible': False,
            })
    args.clip_root.mkdir(parents=True, exist_ok=True)
    video_hashes, video_probes = {}, {}
    for index, row in enumerate(rows):
        source = Path(row['source_video'])
        if str(source) not in video_hashes:
            video_hashes[str(source)] = sha256(source)
            video_probes[str(source)] = probe_video(source)
        if row['end_ms'] / 1000 > video_probes[str(source)]['duration_seconds']:
            raise ValueError(f'annotation exceeds source video: {source}')
        output = args.clip_root / f'cokely_{index:03d}_{"_".join(row["canonical_glosses"])}.mp4'
        subprocess.run([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
            '-ss', f'{row["start_ms"] / 1000:.3f}',
            '-i', str(source), '-t', f'{(row["end_ms"] - row["start_ms"]) / 1000:.3f}',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-movflags', '+faststart',
            str(output),
        ], check=True)
        row['source_video_sha256'] = video_hashes[str(source)]
        row['source_video_probe'] = video_probes[str(source)]
        row['clip_file'] = str(output)
        row['clip_sha256'] = sha256(output)
        row['clip_probe'] = probe_video(output)
        row['video_path'] = row['clip_file']
        row['video_sha256'] = row['clip_sha256']
        expected_duration = (row['end_ms'] - row['start_ms']) / 1000
        if abs(row['clip_probe']['duration_seconds'] - expected_duration) > 0.08:
            raise ValueError(f'clip duration mismatch: {output}')
    report = {
        'format': 'slt_cokely_locked100_candidate_manifest_v17',
        'version': 2,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'vocabulary_manifest': args.manifest.as_posix(),
        'manifest_sha256': sha256(args.manifest),
        'class_count': 100,
        'label_to_index': {
            item['canonical_label']: item['class_index'] for item in classes
        },
        'selection': ('maximal runs of case-sensitive exact Citizen raw labels within one '
                      'parent utterance; conflicting supplementary hand annotations, OOV, '
                      'and missing timing break runs'),
        'max_gap_ms': args.max_gap_ms,
        'clips': len(rows),
        'signers': sorted({row['signer_id'] for row in rows}),
        'recordings': sorted({row['recording_id'] for row in rows}),
        'training_eligible': 0,
        'annotation_decisions': decisions,
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('clips', 'signers', 'recordings', 'training_eligible')}, indent=2))


if __name__ == '__main__':
    main()
