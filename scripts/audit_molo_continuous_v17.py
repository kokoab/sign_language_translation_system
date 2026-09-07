"""Screen public MoLo EAFs for timed, exact-raw-label locked100 candidates.

This is a discovery report, never a training manifest: ASL Signbank variants,
video alignment, signer identity and split independence still need verification.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


def read_eaf(path):
    root = ET.parse(path).getroot()
    header = root.find('HEADER')
    if root.tag != 'ANNOTATION_DOCUMENT' or header is None or header.get('TIME_UNITS') != 'milliseconds':
        raise ValueError(f'{path.name}: expected EAF with millisecond timing')
    times = {t.get('TIME_SLOT_ID'): t.get('TIME_VALUE') for t in root.findall('./TIME_ORDER/TIME_SLOT')}
    annotations = {}
    for a in root.findall('./TIER/ANNOTATION/*'):
        key = a.get('ANNOTATION_ID')
        if not key or key in annotations:
            raise ValueError(f'{path.name}: missing or duplicate annotation ID')
        annotations[key] = a
    events = {}
    participants = set()
    for tier in root.findall('TIER'):
        if tier.get('LINGUISTIC_TYPE_REF') != 'ID Gloss':
            continue
        hand = tier.get('TIER_ID')
        if hand not in ('RightHand_IDg', 'LeftHand_IDg'):
            raise ValueError(f'{path.name}: unexpected manual tier {hand}')
        if tier.get('PARTICIPANT'):
            participants.add(tier.get('PARTICIPANT'))
        for a in tier.findall('./ANNOTATION/*'):
            if a.tag != 'ALIGNABLE_ANNOTATION':
                raise ValueError(f'{path.name}: untimed manual annotation')
            try:
                start, end = (int(times[a.get(ref)]) for ref in ('TIME_SLOT_REF1', 'TIME_SLOT_REF2'))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path.name}: unresolved manual timing') from exc
            if not 0 <= start < end:
                raise ValueError(f'{path.name}: invalid manual interval')
            key = a.get('ANNOTATION_ID')
            events[key] = dict(annotation_id=key, hand=hand, start_ms=start, end_ms=end,
                               raw_gloss=a.findtext('ANNOTATION_VALUE', ''),
                               cve_ref=a.get('CVE_REF'), notes=[])
    for key, a in annotations.items():
        value = a.findtext('ANNOTATION_VALUE', '')
        if a.tag != 'REF_ANNOTATION' or not value:
            continue
        seen = {key}
        parent = a.get('ANNOTATION_REF')
        while parent not in events:
            if parent in seen or parent not in annotations:
                raise ValueError(f'{path.name}: invalid annotation reference')
            seen.add(parent)
            node = annotations[parent]
            if node.tag != 'REF_ANNOTATION':
                break
            parent = node.get('ANNOTATION_REF')
        if parent in events:
            events[parent]['notes'].append(value)
    media = header.findall('MEDIA_DESCRIPTOR')
    return dict(events=list(events.values()), participant_fields=sorted(participants),
                media_names=sorted({Path(unquote(urlparse(m.get('MEDIA_URL', '')).path)).name for m in media}),
                media_time_origins_ms=[int(m.get('TIME_ORIGIN', '0')) for m in media],
                external_refs=[r.get('VALUE') for r in root.findall('EXTERNAL_REF')])


def candidate_spans(events, locked, max_gap_ms=300):
    if max_gap_ms < 0:
        raise ValueError('max_gap_ms must be nonnegative')
    groups = []
    end = -1
    for e in sorted(events, key=lambda e: (e['start_ms'], e['end_ms'], e['annotation_id'])):
        if not 0 <= e['start_ms'] < e['end_ms']:
            raise ValueError('invalid manual interval')
        if groups and e['start_ms'] < end:
            groups[-1].append(e)
            end = max(end, e['end_ms'])
        else:
            groups.append([e])
            end = e['end_ms']
    spans, run = [], []
    # ponytail: overlap groups conservatively reject asynchronous two-hand signs;
    # use reviewed alignments if this screening loses too many valid phrases.
    for group in groups:
        first = group[0]
        valid = (first['raw_gloss'] in locked and
                 all(e['raw_gloss'] == first['raw_gloss'] and not e['notes'] for e in group) and
                 len({e['hand'] for e in group}) == len(group) and
                 (len(group) == 1 or (first['cve_ref'] and
                  all(e['cve_ref'] == first['cve_ref'] for e in group))))
        if not valid or (run and first['start_ms'] - run[-1]['end_ms'] > max_gap_ms):
            if len(run) >= 2:
                spans.append(run)
            run = []
        if valid:
            run.append(dict(raw_gloss=first['raw_gloss'], canonical_label=locked[first['raw_gloss']],
                            start_ms=first['start_ms'], end_ms=max(e['end_ms'] for e in group),
                            cve_ref=first['cve_ref'], annotation_ids=[e['annotation_id'] for e in group],
                            hands=[e['hand'] for e in group]))
    if len(run) >= 2:
        spans.append(run)
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-root', type=Path, required=True)
    parser.add_argument('--source-listing', type=Path, required=True, help='v1 molo_audit.json')
    parser.add_argument('--manifest', type=Path, default=Path('active/v17/citizen100_manifest.json'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--max-gap-ms', type=int, default=300)
    args = parser.parse_args()
    if args.max_gap_ms < 0:
        parser.error('--max-gap-ms must be nonnegative')
    classes = json.loads(args.manifest.read_text())['classes']
    locked = {c['citizen_raw_gloss']: c['canonical_label'] for c in classes}
    if len(classes) != 100 or len(locked) != 100:
        raise ValueError('expected frozen100 unique raw glosses')
    sources = {f['name']: f for f in json.loads(args.source_listing.read_text())['files']}
    paths = sorted(args.annotation_root.glob('*.eaf'))
    if set(p.name for p in paths) != set(sources):
        raise ValueError('annotation files do not match complete source listing')
    files, candidates = [], []
    for path in paths:
        data = path.read_bytes()
        source = sources[path.name]
        if len(data) != source['bytes']:
            raise ValueError(f'{path.name}: source size mismatch')
        parsed = read_eaf(path)
        events = parsed.pop('events')
        counts = Counter(e['raw_gloss'] for e in events)
        session, task, signer = path.stem.split('_')[:3]
        info = dict(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    source_url=source['download'], session_group=session,
                    signer_from_filename_unverified=signer, task=task, **parsed,
                    manual_annotations=len(events),
                    exact_raw_label_annotations=sum(n for g, n in counts.items() if g in locked),
                    annotated_gloss_counts=dict(sorted(counts.items())))
        spans = candidate_spans(events, locked, args.max_gap_ms)
        info['candidate_spans'] = len(spans)
        files.append(info)
        for span in spans:
            candidates.append(dict(candidate_id=f'{path.stem}:{span[0]["start_ms"]}',
                                   file=path.name, source_sha256=info['sha256'],
                                   session_group=session, signer_from_filename_unverified=signer,
                                   participant_fields=parsed['participant_fields'],
                                   media_names=parsed['media_names'],
                                   media_time_origins_ms=parsed['media_time_origins_ms'],
                                   start_ms=span[0]['start_ms'], end_ms=span[-1]['end_ms'],
                                   raw_glosses=[e['raw_gloss'] for e in span], signs=span,
                                   eligible_for_training=False,
                                   unresolved=['exact lexical variant crosswalk', 'source video and time alignment',
                                               'signer and session disjointness', 'annotation completeness']))
    summary = dict(files=len(files), bytes=sum(f['bytes'] for f in files),
                   manual_annotations=sum(f['manual_annotations'] for f in files),
                   exact_raw_label_annotations=sum(f['exact_raw_label_annotations'] for f in files),
                   empty_manual_files=sum(f['manual_annotations'] == 0 for f in files),
                   files_with_two_or_more_manual_annotations=sum(f['manual_annotations'] >= 2 for f in files),
                   signers_with_two_or_more_manual_annotations=sorted({f['signer_from_filename_unverified'] for f in files if f['manual_annotations'] >= 2}),
                   candidate_spans=len(candidates),
                   distinct_raw_sequences=len({tuple(c['raw_glosses']) for c in candidates}),
                   candidate_raw_glosses=sorted({g for c in candidates for g in c['raw_glosses']}),
                   candidate_lengths=dict(sorted(Counter(len(c['signs']) for c in candidates).items())),
                   candidate_sessions=sorted({c['session_group'] for c in candidates}),
                   eligible_for_training=0)
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                 max_gap_ms=args.max_gap_ms, matching='case-sensitive exact raw label, no alias/variant approval',
                 summary=summary, files=files)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (args.output_root / 'candidates.json').write_text(json.dumps(candidates, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
