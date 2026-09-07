#!/usr/bin/env python3
"""Render exact gloss combinations from a blue-avatar pilot or source-grounded bank."""
from pathlib import Path
from dataclasses import fields
import argparse
import json
import subprocess
import sys

import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import (RetargetedAvatar, connect_avatar_clips,
    prepend_isolated_approach, append_isolated_release, bone_length_metrics,
    _fallback_rest_hand, _solve_elbow, mirror_avatar)
from scripts.render_rigged_avatar_v17 import _load_makehuman, render, sha256
from scripts.compare_avatar_source_v17 import video_frames, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('artifacts/reports/signwriting_avatar_pilot_v17_v10_palm_onset'))
    parser.add_argument('--glosses', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transition-seconds', type=float, default=.3)
    parser.add_argument('--reference-motion', action='store_true',
                        help='explicitly render source-estimated bank motions for comparison only')
    args = parser.parse_args()
    if not np.isfinite(args.transition_seconds) or not .1 <= args.transition_seconds <= 1:
        parser.error('transition duration must be between0.1 and1seconds')
    report = json.loads((args.pilot/'report.json').read_text())
    if not report.get('symbol_motion') and report.get('format') != 'slt_avatar_gloss_bank_v17':
        parser.error('input must be a notation pilot or avatar gloss bank')
    items = {item['entry']['gloss']: item for item in report['items']}
    glosses = [g.upper() for g in args.glosses]
    if any(g not in items for g in glosses):
        parser.error('every gloss must have a generated pilot core')
    nonsymbolic = [g for g in glosses if not str(items[g].get('motion_origin',
        'signwriting_pilot' if report.get('symbol_motion') else '')).startswith('signwriting_')]
    if nonsymbolic and not args.reference_motion:
        parser.error('SignWriting motion unavailable for: ' + ', '.join(nonsymbolic))
    fps, approach, release = 30, 14, 10
    join = round(args.transition_seconds * fps)
    clips, timeline, source_frames, provenance, references = [], [], [], [], {}
    mirrored = {}
    cursor = approach
    for gloss in glosses:
        item = items[gloss]
        path = args.pilot/item.get('audit', f'{gloss.lower()}_audit.npz')
        if item.get('audit_sha256') and sha256(path) != item['audit_sha256']:
            raise ValueError('cached gloss motion hash mismatch')
        with np.load(path) as data:
            start, stop = item['core_start_frame'], item['core_stop_frame']
            count = round((stop-start) / item['fps'] * fps)
            indices = np.rint(np.linspace(start, stop-1, count)).astype(int)
            states = data['hand_states'][indices].astype('<U16')
            states[~np.char.startswith(states, 'rest')] = ('source-estimate'
                if item.get('motion_origin') == 'source_world_estimate' else 'symbolic')
            clip = RetargetedAvatar(data['shoulders'][indices], data['elbows'][indices],
                data['symbol_hands'][indices], states, np.zeros((count, 2), bool))
            source_indices = data['source_indices'][indices]
        # Refresh only the generic rest preset in cached cores, preserving active hands.
        for side in range(2):
            for i in np.flatnonzero(states[:, side] == 'rest-uncertain'):
                rest = _fallback_rest_hand(side)
                clip.elbows[i, side], wrist = _solve_elbow(clip.shoulders[i, side], rest[0], side)
                clip.hands[i, side] = rest + wrist-rest[0]
        mirrored[gloss] = bool(np.all(np.char.startswith(states[:, 1], 'rest'))
                               and np.any(~np.char.startswith(states[:, 0], 'rest')))
        if mirrored[gloss]:
            clip = mirror_avatar(clip)
        if clips:
            timeline.append(dict(kind='transition', start=cursor, stop=cursor+join))
            source_frames.extend([None]*join)
            cursor += join
        clips.append(clip)
        timeline.append(dict(kind='gloss', gloss=gloss, start=cursor, stop=cursor+count))
        cursor += count
        source_frames.extend((gloss, int(i)) for i in source_indices)
        provenance.append(dict(gloss=gloss, audit=str(path), sha256=sha256(path),
                               motion_origin=item.get('motion_origin', 'signwriting_pilot'),
                               mirrored_to_right_hand=mirrored[gloss]))
        if gloss not in references:
            video = Path(item['source'])
            if sha256(video) != item['source_sha256']:
                raise ValueError('source video changed since pilot render')
            references[gloss] = video_frames(video)
            if mirrored[gloss]:
                references[gloss] = [cv2.flip(frame, 1) for frame in references[gloss]]
    core = connect_avatar_clips(clips, join)
    rig = append_isolated_release(prepend_isolated_approach(core, approach), release)
    first, last = source_frames[0], source_frames[-1]
    prefix = [(first[0], int(i)) for i in np.linspace(0, first[1], approach, endpoint=False)]
    suffix = [(last[0], min(last[1]+i+1, len(references[last[0]])-1)) for i in range(release)]
    source_frames = prefix + source_frames + suffix
    timeline = [dict(kind='approach', start=0, stop=approach)] + timeline + [dict(kind='release', start=cursor, stop=cursor+release)]
    args.output.mkdir(parents=True, exist_ok=False)
    asset = _load_makehuman(Path('artifacts/tools/makehuman_cc0'))
    temporary = [args.output/'avatar_temporary.mp4', args.output/'comparison_temporary.mp4']
    writers = [cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
               for p, size in zip(temporary, ((720, 900), (1440, 900)))]
    for i, reference in enumerate(source_frames):
        segment = next(s for s in timeline if s['start'] <= i < s['stop'])
        label = segment.get('gloss', segment['kind'])
        avatar = render(rig, i, label, asset)
        writers[0].write(avatar)
        source = fit(references[reference[0]][reference[1]], 720, 900) if reference else np.zeros((900, 720, 3), np.uint8)
        title = f'Source {reference[0]} frame{reference[1]}' if reference else 'Generated join: no paired source footage'
        if reference and mirrored[reference[0]]:
            title += ' (mirrored)'
        cv2.putText(source, title, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 230, 255), 1, cv2.LINE_AA)
        writers[1].write(np.concatenate((source, avatar), axis=1))
    for writer in writers:
        writer.release()
    for source, name in zip(temporary, ('avatar.mp4', 'comparison.mp4')):
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(args.output/name)], check=True)
        source.unlink()
    np.savez_compressed(args.output/'motion.npz', **{f.name: getattr(rig, f.name) for f in fields(rig)})
    result = dict(glosses=glosses, frames=len(rig.hands), fps=fps, timeline=timeline,
        reference_motion=args.reference_motion,
        source_frames=source_frames, source_cores=provenance, training_eligible=False, human_accepted=False,
        pilot_report_sha256=sha256(args.pilot/'report.json'), script_sha256=sha256(Path(__file__)),
        rig_code_sha256=sha256(Path('active/v17/avatar_rig_v17.py')), **bone_length_metrics(rig),
        limitations=['Only exact covered glosses are supported; no English grammar conversion',
                    'Core phases resampled to30fps by nearest sample; source timing is approximate',
                    'Transitions and boundaries are inferred; no paired continuous source validates them'])
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('glosses', 'frames', 'fps')}, indent=2))


if __name__ == '__main__':
    main()
