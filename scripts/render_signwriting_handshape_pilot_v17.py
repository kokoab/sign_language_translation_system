#!/usr/bin/env python3
"""Controlled dictionary-handshape comparison; wrist/orientation remain video-derived."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import (retarget_avatar, constrain_signwriting_handshape, bone_length_metrics,
                                     animate_signwriting_pilot, signwriting_pilot_symbols, _solve_elbow,
                                     prepend_isolated_approach, append_isolated_release)
from active.v17.extract_mediapipe_v17 import MediaPipeHybridDetector, DEFAULT_MODEL_PATH
from active.v17.schema_mediapipe_v17 import MediaPipeV17Config
from active.v17.extract_v17 import AppleVisionDetector, assign_hands
from scripts.compare_avatar_source_v17 import video_frames, fit, match_world_hands
from scripts.render_rigged_avatar_v17 import _load_makehuman, render, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entries', type=Path, default=Path('artifacts/reports/signwriting_avatar_pilot_v17_v1/pilot_entries.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lexicon', type=Path, default=Path('active/v17/citizen100_phonology.json'))
    parser.add_argument('--calibration', type=Path, help='training-only measured placement and duration report')
    parser.add_argument('--symbol-motion', action='store_true',
                        help='also execute the two supported dictionary orientation/movement pairs')
    args = parser.parse_args()
    if args.calibration and not args.symbol_motion:
        parser.error('--calibration requires --symbol-motion')
    args.output.mkdir(parents=True, exist_ok=False)
    entries = json.loads(args.entries.read_text())['entries']
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if calibration:
        for row in calibration['rows']:
            if '/train/' not in row['archive'] or sha256(Path(row['archive'])) != row['archive_sha256']:
                raise ValueError('calibration must pin unchanged training archives')
    lexicon=json.loads(args.lexicon.read_text())
    repetition=next(a for a in lexicon['attributes'] if a['name']=='repeated_movement')
    repeated={c['canonical_label']:repetition['values'][repetition['targets_by_class_index'][c['class_index']]]=='1'
              for c in lexicon['classes']}
    pairs={entry['gloss']:signwriting_pilot_symbols(entry['fsw'],repeated=repeated[entry['gloss']])
           for entry in entries}
    sources = {'YOU': '17987836792170242-YOU', 'NEED': '6613716577421236-NEED'}
    asset = _load_makehuman(Path('artifacts/tools/makehuman_cc0'))
    detector = MediaPipeHybridDetector(DEFAULT_MODEL_PATH, MediaPipeV17Config(include_apple_auxiliary=False))
    apple = AppleVisionDetector()
    reports = []
    for entry in entries:
        gloss = entry['gloss']
        archive = Path('data/local/citizen100_v17/landmarks/train') / gloss / (sources[gloss] + '.v17.npz')
        with np.load(archive) as data:
            features = data['features'].astype(np.float32)
            metadata = json.loads(str(data['metadata_json']))
        video = Path(metadata['video_path'])
        frames = video_frames(video)
        hand_symbol,movement_symbol=pairs[gloss]
        indices = np.rint(np.linspace(metadata['hand_trim_start_frame'],
            metadata['hand_trim_end_frame_exclusive'] - 1, len(features))).astype(int)
        world = np.full((len(features), 2, 21, 3), np.nan, np.float32)
        detector.reset_sequence()
        previous = {'left': None, 'right': None}
        for i, index in enumerate(indices):
            frame = frames[index]
            reference = assign_hands(apple.detect(frame, False, False).hands, previous)
            hands = match_world_hands(detector.detect(frame, False, False).hands, reference)
            for side, name in enumerate(('left', 'right')):
                if hands[name] is not None:
                    world[i, side] = hands[name].world_xyz
                    previous[name] = hands[name].xy[0].copy()
        valid = np.isfinite(world[:, 1]).all((1, 2))
        if not valid.any():
            raise ValueError('no right-hand world estimates for source')
        known = np.flatnonzero(valid)
        for i in np.flatnonzero(~valid):
            world[i, 1] = world[known[np.argmin(abs(known - i))], 1]
        timeline = {'timeline': [dict(kind='gloss', gloss=gloss, start=0, stop=len(features),
                                     hand_participation=[False, True])]}
        presence = features[:, :, 3] > 0
        before = retarget_avatar(features[:, :, :3], presence, presence, timeline, hand_world_xyz=world)
        after = copy.deepcopy(before)
        after.hands[:, 1] = constrain_signwriting_handshape(before.hands[:, 1], hand_symbol, side=1)
        if args.symbol_motion:
            reference_index = int(known[np.argmin(abs(known - len(features)//2))])
            motion_parameters={}
            if calibration:
                center=np.asarray(calibration['summary'][gloss]['wrist_center'])
                motion_parameters['wrist_position']=(center[0]*.335425,1.285-center[1]*.335425,.28)
            after.hands[:, 1] = animate_signwriting_pilot(before.hands[reference_index, 1],
                hand_symbol, movement_symbol, frames=len(features), **motion_parameters)
            for i in range(len(features)):
                elbow, wrist = _solve_elbow(after.shoulders[i,1], after.hands[i,1,0], 1)
                after.elbows[i,1] = elbow
                after.hands[i,1] += wrist-after.hands[i,1,0]
        fps = len(features) * metadata['fps'] / metadata['source_frames_processed']
        if calibration:
            fps=len(features)/calibration['summary'][gloss]['duration_percentiles'][1]
        # Include the source lead-in instead of presenting the first trimmed signing pose at frame0.
        # A clip with no recorded lead-in still needs a visible isolated-avatar approach.
        lead_frames = max(2, round(max(metadata['hand_trim_start_frame'] / metadata['fps'], .4) * fps))
        before = prepend_isolated_approach(before, lead_frames)
        after = prepend_isolated_approach(after, lead_frames)
        indices = np.r_[np.rint(np.linspace(0, metadata['hand_trim_start_frame'],
                                           lead_frames, endpoint=False)).astype(int), indices]
        release_frames = max(2, round(.3 * fps))
        before = append_isolated_release(before, release_frames)
        after = append_isolated_release(after, release_frames)
        indices = np.r_[indices, np.minimum(len(frames) - 1, np.rint(indices[-1] +
                          np.arange(1, release_frames + 1) * metadata['fps'] / fps).astype(int))]
        temporary = args.output / f'{gloss.lower()}_temporary.mp4'
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1600, 600))
        avatar_temporary = args.output / f'{gloss.lower()}_avatar_temporary.mp4'
        avatar_writer = cv2.VideoWriter(str(avatar_temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps, (720, 900))
        for i, index in enumerate(indices):
            avatar = render(after, i, gloss, asset)
            avatar_writer.write(avatar)
            panels = [fit(frames[index], 640, 600), fit(render(before, i, gloss, asset), 480, 600),
                      fit(avatar, 480, 600)]
            for panel, title in zip(panels, ('Source ' + gloss, 'Detector handshape',
                'SignWriting motion' if args.symbol_motion else 'SignWriting handshape')):
                panel[:36] = 25
                cv2.putText(panel, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,230,255), 1, cv2.LINE_AA)
            canvas = np.concatenate(panels, axis=1)
            writer.write(canvas)
            if i in (0, lead_frames//2, lead_frames, lead_frames + len(features)//2, len(indices)-1):
                cv2.imwrite(str(args.output / f'{gloss.lower()}_{i:02d}.png'), canvas)
        writer.release()
        avatar_writer.release()
        destination = args.output / f'{gloss.lower()}.mp4'
        avatar_destination = args.output / f'{gloss.lower()}_avatar.mp4'
        for source, target in ((temporary, destination), (avatar_temporary, avatar_destination)):
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
                '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(target)], check=True)
            source.unlink()
        np.savez_compressed(args.output / f'{gloss.lower()}_audit.npz', source_indices=indices,
                            original_hands=before.hands, symbol_hands=after.hands,
                            shoulders=after.shoulders, elbows=after.elbows,
                            world_observed=np.r_[np.zeros(lead_frames, dtype=bool), valid,
                                                 np.zeros(release_frames, dtype=bool)],
                            core_start_frame=lead_frames, hand_states=after.hand_states)
        reports.append(dict(entry=entry, source=str(video), source_sha256=sha256(video),
            hand_symbol=hand_symbol, movement_symbol=movement_symbol, world_imputed_frames=int((~valid).sum()),
            presentation_duration_seconds=len(indices)/fps, fps=fps,
            core_start_frame=lead_frames, approach_duration_seconds=lead_frames/fps,
            core_stop_frame=lead_frames + len(features), release_duration_seconds=release_frames/fps,
            approach_provenance='inferred neutral-to-first-core pose; source lead-in displayed for comparison',
            source_trim_duration_seconds=metadata['source_frames_processed']/metadata['fps'],
            video=str(destination), avatar_video=str(avatar_destination), **bone_length_metrics(after)))
    detector.close()
    report = dict(items=reports, training_eligible=False, human_accepted=False,
        symbol_motion=args.symbol_motion,
        calibration_sha256=sha256(args.calibration) if args.calibration else None,
        entries_sha256=sha256(args.entries), code_sha256=sha256(Path(__file__)),
        lexicon_sha256=sha256(args.lexicon),
        rig_code_sha256=sha256(Path('active/v17/avatar_rig_v17.py')),
        limitations=(['Only three exact right-handed symbol pairs execute; all others are rejected',
                     'A static source hand supplies proportions/thumb; reference clip supplies duration',
                     'Wrist location, 6cm travel and 65degree flex are explicit animation assumptions'] if args.symbol_motion else
                    ['Only S100/S106 handshape bases are decoded; fill/rotation and movement are not decoded',
                     'Wrist, palm orientation, thumb and timing remain source-derived']) + [
                     'Fixed flexion angles are animation assumptions, not measured or native-approved',
                     'Isolated source clips provide no natural inter-sign transitions'] +
                    ['Opening approach and final release are inferred, not measured; core poses are preserved'] +
                    (['Comparison source phases retimed to training-median duration; not a source-speed assessment']
                     if calibration else []))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
