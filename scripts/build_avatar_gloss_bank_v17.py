#!/usr/bin/env python3
"""Build a train-source motion bank for the frozen Citizen100 blue-avatar vocabulary."""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import retarget_avatar, interpolate_world_hand, bone_length_metrics
from scripts.render_rigged_avatar_v17 import sha256


def exact_training_rows(rows, cls):
    return [r for r in rows if r['split'] == 'train'
            and r['class_index'] == str(cls['class_index'])
            and r['canonical_label'] == cls['canonical_label']
            and r['raw_gloss'] == cls['citizen_raw_gloss']
            and r['asl_lex_code'] == cls['citizen_asl_lex_code']]


def participating_hands(features, sign_type):
    supported = [np.flatnonzero(features[:, start, 3] > 0) for start in (0, 21)]
    if not any(len(x) for x in supported):
        raise ValueError('no observed wrists')
    if sign_type != 'OneHanded':
        return [True, True]
    motion = []
    for start, indices in zip((0, 21), supported):
        points = features[indices, start, :2]
        motion.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.)
    side = max(range(2), key=lambda i: (motion[i], len(supported[i]), i))
    return [side == 0, side == 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--glosses', nargs='+', help='optional exact subset for smoke validation')
    parser.add_argument('--manifest', type=Path, default=Path('active/v17/citizen100_manifest.json'))
    parser.add_argument('--lexicon', type=Path, default=Path('active/v17/citizen100_phonology.json'))
    parser.add_argument('--provenance', type=Path, default=Path('data/local/citizen100_v17/provenance.csv'))
    parser.add_argument('--pilot', type=Path, default=Path('artifacts/reports/signwriting_avatar_pilot_v17_v10_palm_onset'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    lexicon = json.loads(args.lexicon.read_text())
    if lexicon['manifest_sha256'] != sha256(args.manifest):
        raise ValueError('phonology manifest hash mismatch')
    classes = manifest['classes']
    if args.glosses:
        wanted = set(g.upper() for g in args.glosses)
        if wanted - {c['canonical_label'] for c in classes}:
            parser.error('unknown frozen gloss')
        classes = [c for c in classes if c['canonical_label'] in wanted]
    with args.provenance.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    types = next(a for a in lexicon['attributes'] if a['name'] == 'sign_type')
    dictionary_path = Path('artifacts/reports/signwriting_avatar_pilot_v17_v1/dictionary_100_coverage.json')
    dictionary = json.loads(dictionary_path.read_text())
    candidates = {r['canonical_label']: r['candidates'] for r in dictionary['rows']}
    pilot = {r['entry']['gloss']: r for r in json.loads((args.pilot/'report.json').read_text())['items']}
    args.output.mkdir(parents=True, exist_ok=False)
    from active.v17.extract_mediapipe_v17 import MediaPipeHybridDetector, DEFAULT_MODEL_PATH
    from active.v17.schema_mediapipe_v17 import MediaPipeV17Config
    from active.v17.extract_v17 import AppleVisionDetector, assign_hands
    from scripts.compare_avatar_source_v17 import video_frames, match_world_hands
    detector = MediaPipeHybridDetector(DEFAULT_MODEL_PATH, MediaPipeV17Config(include_apple_auxiliary=False))
    apple = AppleVisionDetector()
    report = dict(format='slt_avatar_gloss_bank_v17', items=[], failures=[], training_eligible=False,
        human_accepted=False, manifest_sha256=sha256(args.manifest), lexicon_sha256=sha256(args.lexicon),
        dictionary_sha256=sha256(dictionary_path), code_sha256=sha256(Path(__file__)),
        rig_code_sha256=sha256(Path('active/v17/avatar_rig_v17.py')),
        limitations=['YOU/NEED use accepted symbolic cores; other glosses use source-estimated motion',
                    'Dictionary candidates are references, not verified executable notation',
                    'Fixed review rig lacks facial grammar and measured camera depth',
                    'Source clips are isolated; connected coarticulation remains generated'])
    try:
        for cls in classes:
            gloss = cls['canonical_label']
            try:
                if gloss in pilot:
                    item = dict(pilot[gloss])
                    old_path = args.pilot/f'{gloss.lower()}_audit.npz'
                    path = args.output/f'{gloss.lower()}_audit.npz'
                    path.write_bytes(old_path.read_bytes())
                    item.update(motion_origin='signwriting_pilot', source_audit_sha256=sha256(old_path))
                else:
                    sign_type = types['values'][types['targets_by_class_index'][cls['class_index']]]
                    choices = []
                    for row in exact_training_rows(rows, cls):
                        archive = Path('data/local/citizen100_v17/landmarks/train')/gloss/(Path(row['video']).stem+'.v17.npz')
                        if not archive.exists():
                            continue
                        with np.load(archive) as data:
                            f = data['features'].astype(np.float32)
                        participation = participating_hands(f, sign_type)
                        support = [(f[:, side*21:(side+1)*21, 3] > 0).all(axis=1).mean()
                                   for side in range(2) if participation[side]]
                        choices.append((min(support), row['participant'] == 'P52', str(archive), row, f, participation))
                    if not choices:
                        raise ValueError('no exact training source')
                    _, _, archive_name, row, f, participation = max(choices, key=lambda x: x[:3])
                    archive = Path(archive_name)
                    with np.load(archive) as data:
                        metadata = json.loads(str(data['metadata_json']))
                    video = Path(row['destination'])
                    if sha256(video) != row['sha256']:
                        raise ValueError('source video hash mismatch')
                    images = video_frames(video)
                    source_indices = np.rint(np.linspace(metadata['hand_trim_start_frame'],
                        metadata['hand_trim_end_frame_exclusive']-1, len(f))).astype(int)
                    present = f[:, :, 3] > 0
                    supported = np.ones(len(f), bool)
                    for side in range(2):
                        if participation[side]:
                            supported &= present[:, side*21:(side+1)*21].all(axis=1)
                        else:
                            present[:, side*21:(side+1)*21] = False
                    good = np.flatnonzero(supported)
                    if not len(good):
                        raise ValueError('no frame with every participating hand observed')
                    # Existing extraction includes approach/release; trim only unsupported edges.
                    lo, hi = int(good[0]), int(good[-1])+1
                    normalized_frames = len(f)
                    f, present, source_indices = f[lo:hi], present[lo:hi], source_indices[lo:hi]
                    world = np.full((len(f), 2, 21, 3), np.nan, np.float32)
                    detector.reset_sequence()
                    previous = {'left': None, 'right': None}
                    for i, index in enumerate(source_indices):
                        reference = assign_hands(apple.detect(images[index], False, False).hands, previous)
                        hands = match_world_hands(detector.detect(images[index], False, False).hands, reference)
                        for side, name in enumerate(('left', 'right')):
                            if reference[name] is not None:
                                previous[name] = reference[name].xy[0].copy()
                            if hands[name] is not None and participation[side]:
                                world[i, side] = hands[name].world_xyz
                    observed = np.isfinite(world).all((2, 3))
                    for side in range(2):
                        if not participation[side]:
                            continue
                        known = np.flatnonzero(observed[:, side])
                        if not len(known):
                            raise ValueError(f'no world estimates for participating side{side}')
                        for i in np.flatnonzero(~observed[:, side]):
                            left = known[known < i]; right = known[known > i]
                            if len(left) and len(right):
                                world[i, side] = interpolate_world_hand(world[left[-1], side], world[right[0], side],
                                                                       (i-left[-1])/(right[0]-left[-1]))
                            else:
                                world[i, side] = world[known[np.argmin(abs(known-i))], side]
                    timeline = {'timeline': [dict(kind='gloss', start=0, stop=len(f), gloss=gloss,
                                                 hand_participation=participation)]}
                    rig = retarget_avatar(f[:, :, :3], present, present, timeline, hand_world_xyz=world)
                    path = args.output/f'{gloss.lower()}_audit.npz'
                    np.savez_compressed(path, shoulders=rig.shoulders, elbows=rig.elbows, symbol_hands=rig.hands,
                                        hand_states=rig.hand_states, source_indices=source_indices, world_observed=observed)
                    fps = normalized_frames * metadata['fps'] / metadata['source_frames_processed']
                    item = dict(entry=dict(gloss=gloss), source=str(video), source_sha256=row['sha256'],
                        archive=str(archive), archive_sha256=sha256(archive), participant=row['participant'],
                        motion_origin='source_world_estimate', fps=fps, core_start_frame=0, core_stop_frame=len(f),
                        hand_participation=participation, world_observed_fraction=observed.mean(axis=0).tolist(),
                        **bone_length_metrics(rig))
                item.update(class_index=cls['class_index'], raw_gloss=cls['citizen_raw_gloss'],
                    asl_lex_code=cls['citizen_asl_lex_code'], dictionary_candidates=candidates[gloss],
                    audit_sha256=sha256(path), audit=path.name)
                report['items'].append(item)
                print(f'{gloss}: {item["motion_origin"]}', flush=True)
            except (ValueError, OSError, KeyError) as error:
                report['failures'].append(dict(gloss=gloss, error=str(error)))
                print(f'{gloss}: FAILED {error}', flush=True)
            report['coverage'] = len(report['items'])
            report['requested_classes'] = len(classes)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    finally:
        detector.close()
    if report['failures']:
        raise SystemExit(f'{len(report["failures"])} glosses failed; see report')


if __name__ == '__main__':
    main()
