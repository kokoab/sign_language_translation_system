"""Select exact native-recognized Citizen training originals; preserve all attempts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import build_gloss_examples_v17 as builder

OUT = Path(__file__).resolve().parent
DEST = ROOT / 'data/local/model_class_examples'
HARNESS = Path('/private/tmp/slt-class-examples-independent-harness')
RESOURCES = ROOT / 'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/Models'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def prepare():
    classes = json.loads(builder.MANIFEST.read_text())['classes']
    provenance = builder.provenance_rows()
    rejected = builder.rejected_videos()
    pools = {}
    for cls in classes:
        label = cls['canonical_label']
        candidates = sorted(builder.collect(label, rejected, provenance), key=lambda c: (-c.score, str(c.video)))
        rows = []
        for candidate in candidates:
            p = provenance[candidate.video.name]
            assert p['split'] == 'train' and p['canonical_label'] == label
            assert p['raw_gloss'] == cls['citizen_raw_gloss']
            assert p['asl_lex_code'] == cls['citizen_asl_lex_code']
            assert int(p['class_index']) == cls['class_index']
            assert Path(p['destination']) == candidate.video.relative_to(ROOT)
            rows.append(dict(label=label, class_index=cls['class_index'], source='citizen',
                             source_video=str(candidate.video.relative_to(ROOT)),
                             source_sha256=p['sha256'], participant=p['participant'],
                             raw_gloss=p['raw_gloss'], asl_lex_code=p['asl_lex_code'],
                             split='train', quality_score=candidate.score))
        assert rows, label
        pools[label] = rows
    assert len(pools) == 100
    save('candidate_pools.json', pools)
    print('Citizen training candidates:', sum(map(len, pools.values())), 'classes:', len(pools), flush=True)


def replay(name, chosen):
    videos = [dict(id=label, path=str(ROOT / row['source_video']), reference=[label]) for label, row in chosen.items()]
    save(name + '_videos.json', videos)
    save(name + '_protocol.json', dict(independent_vision=True, harness_sha256=digest(HARNESS)))
    env = os.environ.copy()
    for key in ('LETTERS', 'WORD_SINK', 'FINISH', 'TRACE', 'LIVE_TIMES', 'LIVE_OFFSET', 'SIGNER_LOCK', 'LIVE_REEL_ENCODER'):
        env.pop(key, None)
    env['LIVE_REEL_ENCODER'] = 'MobileCLIP2S0ImageEncoderV17FP16'
    with (OUT / (name + '.log')).open('w') as log:
        subprocess.run([str(HARNESS), 'videos', str(RESOURCES), str(OUT / (name + '_videos.json')),
                        str(OUT / (name + '_replay.json')), 'all'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = json.loads((OUT / (name + '_replay.json')).read_text())['records']
    assert len(records) == len(chosen)
    good = {r['id'] for r in records if r['hypothesis'] == [r['id']]}
    print(name, 'exact:', len(good), '/', len(records), 'misses:', [r['id'] for r in records if r['id'] not in good], flush=True)
    return good, records


def select():
    pools = json.loads((OUT / 'candidate_pools.json').read_text())
    selected, attempts = {}, []
    for n in range(max(map(len, pools.values()))):
        chosen = {label: rows[n] for label, rows in pools.items() if label not in selected and n < len(rows)}
        if not chosen:
            break
        for row in chosen.values():
            assert digest(ROOT / row['source_video']) == row['source_sha256']
        good, records = replay('round_%02d' % n, chosen)
        attempts.extend(dict(**chosen[r['id']], hypothesis=r['hypothesis'], exact=r['id'] in good, round=n) for r in records)
        selected.update({label: chosen[label] for label in good})
        save('selection.json', selected)
        save('attempts.json', attempts)
        if len(selected) == 100:
            break
    save('gaps.json', sorted(set(pools) - set(selected)))
    print('Selected', len(selected), 'classes', flush=True)


def repair():
    selected = json.loads((OUT / 'selection.json').read_text())
    pools = json.loads((OUT / 'candidate_pools.json').read_text())
    attempts = json.loads((OUT / 'attempts.json').read_text())
    pending = json.loads((OUT / 'final_gaps.json').read_text())
    tried = {r['source_video'] for r in attempts}
    index = len(list(OUT.glob('repair_*_videos.json')))
    while pending:
        chosen = {label: next(r for r in pools[label] if r['source_video'] not in tried) for label in pending}
        for row in chosen.values():
            assert digest(ROOT / row['source_video']) == row['source_sha256']
        good, records = replay('repair_%02d' % index, chosen)
        attempts.extend(dict(**chosen[r['id']], hypothesis=r['hypothesis'], exact=r['id'] in good, round='repair_%02d' % index) for r in records)
        tried.update(r['source_video'] for r in chosen.values())
        selected.update({label: chosen[label] for label in good})
        pending = [label for label in pending if label not in good]
        save('selection.json', selected)
        save('attempts.json', attempts)
        index += 1


def deliver(verified=None):
    selected = json.loads((OUT / 'selection.json').read_text())
    assert len(selected) == 100
    if verified is None:
        index = len(list(OUT.glob('final_*_videos.json')))
        good, records = replay('final_%02d' % index, selected)
    else:
        good = set(verified)
    save('final_gaps.json', sorted(set(selected) - good))
    if len(good) != 100:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    assert not list(DEST.iterdir()), 'Refusing to overwrite destination contents'
    rows = []
    for label, row in selected.items():
        source = ROOT / row['source_video']
        target = DEST / (label + source.suffix.lower())
        assert digest(source) == row['source_sha256']
        shutil.copy2(source, target)
        assert digest(target) == row['source_sha256']
        rows.append(dict(row, filename=target.name, recognition=[label], recognition_exact=True, bytes=target.stat().st_size))
    assert len(list(DEST.glob('*.mp4'))) == 100
    save('manifest.json', dict(format='atlas_known_training_class_examples', date='2026-10-19',
                              destination=str(DEST.relative_to(ROOT)), classes=100, videos=100,
                              source_counts={'citizen': 100}, validation_accessed=False, test_accessed=False,
                              verification='Native Swift/CoreML replay on Mac with current bundled FP16 models, .all, EOF flush; fresh Vision state per independent video; exact single-gloss output',
                              caveat='Selected training demonstrations; not independent accuracy or phone-camera evidence.',
                              recognition_evidence=verified,
                              examples=rows))
    with (OUT / 'manifest.csv').open('w', newline='') as stream:
        fields = ['filename', 'label', 'class_index', 'source', 'split', 'source_video', 'source_sha256', 'participant', 'raw_gloss', 'asl_lex_code', 'recognition_exact', 'bytes']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print('Copied and hash-verified:', len(rows), 'videos;', sum(r['bytes'] for r in rows), 'bytes', flush=True)


def finish():
    selected = json.loads((OUT / 'selection.json').read_text())
    verified = {}
    for protocol_path in sorted(OUT.glob('*_protocol.json')):
        protocol = json.loads(protocol_path.read_text())
        assert protocol['independent_vision'] and protocol['harness_sha256'] == digest(HARNESS)
        name = protocol_path.name.removesuffix('_protocol.json') if hasattr(str, 'removesuffix') else protocol_path.name[:-len('_protocol.json')]
        replay_path = OUT / (name + '_replay.json')
        if not replay_path.exists():
            continue
        videos = {r['id']: r for r in json.loads((OUT / (name + '_videos.json')).read_text())}
        for r in json.loads(replay_path.read_text())['records']:
            label = r['id']
            if r['hypothesis'] == [label] and videos[label]['path'] == str(ROOT / selected[label]['source_video']):
                verified[label] = dict(replay=replay_path.name, hypothesis=r['hypothesis'], protocol=protocol_path.name)
    assert set(verified) == set(selected), sorted(set(selected) - set(verified))
    save('independent_verification.json', verified)
    deliver(verified)


if __name__ == '__main__':
    {'prepare': prepare, 'select': select, 'repair': repair, 'deliver': deliver, 'finish': finish}[sys.argv[1]]()
