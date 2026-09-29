"""Teacher (DGS on MediaPipe) and student (Apple Vision raw) caches for YouTube-ASL segments.

Unlabeled continuous signing only; feeds scripts/train_av_boundary_v17.py. Shard with
--shard i --shards n to run several processes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import segmental_lab_v17 as lab
from active.v17.pretrained_boundary_v17 import load_pose
from active.v17.approved_phrase_data_v17 import digest

CLIPS = ROOT / 'data/local/youtube_asl_boundary_distill_v17/clips'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='mps')
    args = ap.parse_args()
    sys.path.insert(0, str(lab.UPSTREAM))
    from prepare import extract
    from scripts.evaluate_temporal_boundary_v17 import arguments, observations
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.stage1_window_v17 import raw_observation_features
    dgs = lab.DGS(args.device)
    run_args = arguments('data', lab.REPORT / 'sessions')
    detector = AppleVisionDetector(run_args.minimum_point_confidence)
    pose_dir = lab.CACHE / 'yt_poses'
    pose_dir.mkdir(parents=True, exist_ok=True)
    (lab.CACHE / 'av_raw').mkdir(parents=True, exist_ok=True)
    done = 0
    tick = time.perf_counter()
    while True:
        clips = sorted(CLIPS.glob('*.mp4'))
        todo = [c for i, c in enumerate(clips) if i % args.shards == args.shard
                and not (lab.CACHE / 'av_raw' / (lab.key('yt:' + c.stem) + '.npz')).exists()]
        if not todo:
            break
        for clip in todo:
            item = 'yt:' + clip.stem
            k = lab.key(item)
            row = dict(item=item, source_item_id=item, role='distill', video_path=str(clip.relative_to(ROOT)),
                       video_sha256=digest(clip), intervals=[])
            try:
                target = lab.CACHE / 'dgs' / (k + '.npz')
                if not target.exists():
                    record = extract(row, pose_dir)
                    pose = load_pose(dict(pose_path=record['pose_path'], pose_fps=record['pose_fps']))
                    sign, phrase = dgs.all_windows(pose)
                    np.savez_compressed(target, sign=sign, phrase=phrase, frames=len(sign))
                obs = observations(row, run_args, detector)
                raw, times = raw_observation_features(obs)
                np.savez_compressed(lab.CACHE / 'av_raw' / (k + '.npz'), raw=raw.astype(np.float16), times=times)
            except Exception as error:  # a bad download must not stop the shard
                print('skip', clip.name, repr(error)[:200], flush=True)
                (lab.CACHE / 'av_raw' / (k + '.npz')).write_bytes(b'')
                continue
            done += 1
            print('yt shard %d done %d (%.0fs)' % (args.shard, done, time.perf_counter() - tick), flush=True)
    print('SHARD DONE', args.shard, done, flush=True)


if __name__ == '__main__':
    main()
