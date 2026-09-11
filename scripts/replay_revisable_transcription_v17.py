#!/usr/bin/env python3
"""Replay frozen development clips and the user's diagnostic recording in revisable mode."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def recording_times(timestamps, frame_count, fps, exclude_final_frames=0):
    if frame_count <= 0 or not 0 <= exclude_final_frames < frame_count:
        raise ValueError('invalid recording length or final-frame exclusion')
    if timestamps is None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('invalid recording FPS')
        timestamps = [i / fps for i in range(frame_count)]
    times = [float(t) for t in timestamps]
    if len(times) < frame_count or any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError('missing or invalid frame timestamps')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('frame timestamps must strictly increase')
    return times[:frame_count - exclude_final_frames]


def validate_recording(row):
    key = str(row['item_id'])
    if '/' in key or '\\' in key or '..' in key:
        raise ValueError('unsafe item id')
    if row.get('role') not in {'validation', 'diagnostic'}:
        raise ValueError('only development or diagnostic recordings permitted')
    if {'test', 'external_evaluation_reserved'} & {p.casefold() for p in Path(row['video']).parts}:
        raise ValueError('protected recording')
    if row['role'] == 'diagnostic' and row.get('reference') is not None:
        raise ValueError('diagnostic recording cannot supply evaluation truth')
    if row.get('video_sha256'):
        import hashlib
        digest = hashlib.sha256()
        with (ROOT / row['video']).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != row['video_sha256']:
            raise ValueError('recording changed after manifest freeze')
    return row


def phase_windows(timestamps, period, origin):
    if period <= 0 or not math.isfinite(period) or not math.isfinite(origin):
        raise ValueError('invalid window period/origin')
    groups = {}
    for index, seconds in enumerate(timestamps):
        key = math.floor((seconds - origin) / period)
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='JSON recording list with video, role, optional frame_timestamps and exclusions')
    parser.add_argument('--backend', choices=('ctc', 'reel', 'stage1-window'), default='ctc')
    parser.add_argument('--diagnose', action='store_true', help='matched Stage1 proposal/verifier and CTC evidence, four window origins')
    args = parser.parse_args()
    if args.manifest:
        rows = json.loads(args.manifest.read_text())
        if isinstance(rows, dict):
            rows = rows['recordings']
        rows = [validate_recording(row) for row in rows]
    else:
        rows = default_recordings()
    if args.diagnose:
        from scripts.diagnose_stage1_window_v17 import diagnose
        diagnose(rows, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        target = args.output/row['item_id'].replace(':', '_').replace('.mp4', '')
        target.mkdir(exist_ok=True)
        entrypoint = 'scripts/live_reel_stage1_v17.py' if args.backend == 'reel' else 'scripts/live_reel_continuous_v17.py'
        command = [str(ROOT/'venv/bin/python'), entrypoint,
            '--video', str(row['video']), '--realtime-video',
            '--finish-at-eof', '--no-display', '--no-speech', '--no-finish-gesture', '--naturalizer', 'literal',
            '--output-root', str(target)]
        if args.backend == 'ctc':
            command += ['--revisable-transcript']
        elif args.backend == 'stage1-window':
            if not args.checkpoint:
                parser.error('stage1-window requires --checkpoint')
            command += ['--transcript-backend', 'stage1-window', '--stage1-window-checkpoint', str(args.checkpoint)]
        if args.checkpoint and args.backend == 'ctc':
            command += ['--stage2-live-checkpoint', str(args.checkpoint)]
        if row.get('frame_timestamps'):
            command += ['--frame-timestamps', str(row['frame_timestamps'])]
        if row.get('exclude_final_frames'):
            command += ['--exclude-final-frames', str(row['exclude_final_frames'])]
        with (target/'process.log').open('w') as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        result = dict(**row, command=command, exit_code=proc.returncode)
        if proc.returncode == 0:
            path = sorted(target.glob('*/history.json'))[-1]
            data = json.loads(path.read_text())
            finishes = [e for e in data['events'] if e['type'] == 'finished_sequence_selected']
            assert len(finishes) == 1
            result.update(history=str(path), finish=finishes[0])
        results.append(result)
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(row['item_id'], result.get('finish', {}).get('selected'), proc.returncode, flush=True)
        assert proc.returncode == 0


def default_recordings():
    rows = json.loads((ROOT/'artifacts/reports/stage2_v17_live_lock_diagnosis_v1/matched_manifest.json').read_text())
    rows += [dict(item_id='local_hello', video='data/raw_videos/PHRASES/HELLO_HOW_YOU/HELLO_HOW_YOU_20.mp4', reference=['HELLO', 'HOW', 'YOU']),
             dict(item_id='local_please', video='data/raw_videos/PHRASES/PLEASE_HELP_ME/3e65d621.mp4', reference=['PLEASE', 'HELP', 'I'])]
    history = ROOT/'artifacts/reports/live_reel_continuous_v17/20260910_071844_645159/history.json'
    recorded = json.loads(history.read_text())
    rows.append(dict(item_id='user_diagnostic', video=recorded['video'], reference=None,
        limitation='Saved low-resolution constant-rate video lacks original source timing and verified intended transcript; no WER.'))
    return rows


if __name__ == '__main__':
    main()
