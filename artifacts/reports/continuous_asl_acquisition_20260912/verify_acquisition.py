"""Verify acquired files and count coverage; never admit them to model training."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts.audit_molo_continuous_v17 import read_eaf

DATA = ROOT/'data/local/continuous_asl_acquisition_20260912'
REPORT = ROOT/'artifacts/reports/continuous_asl_acquisition_20260912'
classes = json.loads((ROOT/'active/v17/citizen100_manifest.json').read_text())['classes']
locked = {c['citizen_raw_gloss'] for c in classes}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def video_info(path):
    path = path.resolve()
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration',
        '-of', 'json', str(path)]))['streams'][0]
    assert int(result['nb_read_frames']) == int(result['nb_frames']), path
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                bytes=path.stat().st_size, **result)


def verify_epee():
    directory = DATA/'epee'
    metadata = list(csv.DictReader((directory/'metadata.csv').open()))
    assert len(metadata) == len({r['clip_id'] for r in metadata}) == 1200
    assert len(list((directory/'annotations').glob('*.json'))) == 1200
    assert len(list((directory/'keypoints').glob('*.npy'))) == 1200
    rows, counts, signers, signer_glosses = [], Counter(), Counter(), defaultdict(set)
    errors, repeated, immediate, long_segments, frames, total_bytes = [], 0, 0, 0, 0, 0
    for entry in metadata:
        name = entry['clip_id']
        annotation_path = directory/'annotations'/f'{name}.json'
        pose_path = directory/'keypoints'/f'{name}.npy'
        annotation = json.loads(annotation_path.read_text())
        pose = np.load(pose_path, allow_pickle=False)
        assert annotation['clip_id'] == name and annotation['signer_id'] == entry['signer_id']
        assert annotation['sign_language'] == entry['sign_language'] == 'ASL'
        assert pose.shape == (annotation['n_frames'], 128, 3)
        assert annotation['n_frames'] == int(entry['n_frames']) and np.isfinite(pose).all()
        assert float(entry['fps']) == annotation['fps'] and annotation['fps'] > 0
        duration = annotation['n_frames']/annotation['fps']
        segments = annotation['segments']
        assert [s['gloss'] for s in segments] == entry['gloss_sequence'].split(' | ')
        issues = []
        for index, segment in enumerate(segments):
            a, b = segment['start'], segment['end']
            if not np.isfinite([a, b]).all() or not 0 <= a < b <= duration + 1/annotation['fps']:
                issues.append(dict(segment=index, reason='invalid_or_out_of_video_interval'))
            if index and a < segments[index-1]['end']:
                issues.append(dict(segment=index, reason='overlapping_segments'))
            counts[segment['gloss']] += 1
            signer_glosses[segment['gloss']].add(entry['signer_id'])
            long_segments += int(b-a > 1.07)
        if issues:
            errors.append(dict(clip_id=name, issues=issues))
        labels = [s['gloss'] for s in segments]
        repeated += int(len(set(labels)) < len(labels))
        immediate += sum(a == b for a, b in zip(labels, labels[1:]))
        signers[entry['signer_id']] += 1
        frames += annotation['n_frames']
        total_bytes += pose_path.stat().st_size + annotation_path.stat().st_size
        rows.append(dict(clip_id=name, signer_id=entry['signer_id'], phrase_id=annotation['phrase_id'],
                         seconds=duration, segments=len(segments), structurally_valid=not issues,
                         exact_raw_label_tokens=sum(label in locked for label in labels),
                         annotation_sha256=digest(annotation_path), keypoints_sha256=digest(pose_path)))
    with (REPORT/'epee_locked100_coverage.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=['canonical_label', 'citizen_raw_gloss',
            'citizen_asl_lex_code', 'exact_raw_tokens', 'exact_raw_signers', 'signers'])
        writer.writeheader()
        for c in classes:
            gloss = c['citizen_raw_gloss']
            writer.writerow(dict(canonical_label=c['canonical_label'], citizen_raw_gloss=gloss,
                citizen_asl_lex_code=c['citizen_asl_lex_code'], exact_raw_tokens=counts[gloss],
                exact_raw_signers=len(signer_glosses[gloss]), signers='|'.join(sorted(signer_glosses[gloss]))))
    result = dict(revision=json.loads((directory/'source_api.json').read_text())['sha'],
                  locked_manifest_sha256=digest(ROOT/'active/v17/citizen100_manifest.json'),
                  clips=len(rows), signers=dict(signers), frames=frames,
                  duration_minutes=sum(r['seconds'] for r in rows)/60, bytes=total_bytes,
                  gloss_tokens=sum(counts.values()), unique_glosses=len(counts),
                  exact_raw_labels=len(locked & counts.keys()), exact_raw_tokens=sum(counts[g] for g in locked),
                  hungry_tokens=counts['HUNGRY'], hungry_signers=sorted(signer_glosses['HUNGRY']),
                  repeated_label_clips=repeated, adjacent_identical_label_pairs=immediate,
                  annotated_segments_longer_than_1_07_seconds=long_segments,
                  annotation_issues=errors, rows=rows,
                  raw_video_available=False, apple_vision_compatible=False, training_eligible=False,
                  limitation='Publisher-described linguistically validated temporal annotations; visual variants and raw pixels cannot be independently checked. No alias/numbered-variant merging.')
    (REPORT/'epee_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Epee', {k: v for k, v in result.items() if k not in ('rows', 'annotation_issues')}, flush=True)


def verify_raw():
    rit = DATA/'rit_homework_sample'
    info = video_info(rit/'H01F13U01.mov')
    root = ET.parse(rit/'H01F13U01.eaf').getroot()
    times = {s.get('TIME_SLOT_ID'): int(s.get('TIME_VALUE'))/1000 for s in root.findall('./TIME_ORDER/TIME_SLOT')}
    intervals = [dict(raw_gloss=a.findtext('ANNOTATION_VALUE'), start=times[a.get('TIME_SLOT_REF1')],
                      end=times[a.get('TIME_SLOT_REF2')]) for a in root.findall("./TIER[@TIER_ID='Gloss']/ANNOTATION/ALIGNABLE_ANNOTATION")]
    assert all(0 <= e['start'] < e['end'] <= float(info['duration']) for e in intervals)
    rit_result = dict(video=info, intervals=intervals, annotation_sha256=digest(rit/'H01F13U01.eaf'),
                      source_signer_id='F13', eaf_participant='Kas', training_eligible=False,
                      limitation='Public one-signer sample; EAF participant field differs from source filename identity. Citizen visual variants and pipeline orientation not admitted.')
    molo = DATA/'molo'
    source = video_info(molo/'201102_FrankGriffin_JonHenner_MoLo003_S_4_5_Logi.mp4')
    expected = json.loads((molo/'systems_download.json').read_text())
    assert source['sha256'] == expected['sha256'] and source['bytes'] == expected['bytes']
    transcripts = []
    original_annotations = {r['file']: r for r in json.loads((molo/'annotation_sources.json').read_text())}
    for path in sorted(molo.glob('*.eaf')):
        data = read_eaf(path)
        assert digest(path) == original_annotations[path.name]['sha256']
        assert data['media_names'] == ['201102_FrankGriffin_JonHenner_MoLo003_S_4_5_Logi.mp4']
        assert all(origin == 0 for origin in data['media_time_origins_ms'])
        assert all(e['end_ms']/1000 <= float(source['duration']) for e in data['events'])
        transcripts.append(dict(file=path.name, sha256=digest(path),
                                source_url=original_annotations[path.name]['source_url'], **data))
    result = dict(rit=rit_result, molo=dict(video=source, transcripts=transcripts,
        timed_hand_annotations=sum(len(t['events']) for t in transcripts),
        training_eligible=False, limitation='Human EAFs are work in progress; raw filenames and duration match, but full temporal alignment and exact Citizen variants are not independently approved. Unannotated periods are not background labels.'))
    (REPORT/'raw_video_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Raw video verification complete', flush=True)


if __name__ == '__main__':
    verify_epee()
    verify_raw()
