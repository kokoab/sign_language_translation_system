"""Extract signing data for the spelling-vs-signing gate through the live Apple Vision contract (20 Hz).

- O5S5 narratives (continuous signing, fingerspelling glossed 'FS' on the hand tiers): whole videos ->
  data/local/open_asl_alternatives_20260913/o5s5/live_features/<video>.npz. LG is evaluation-only.
- ASL Citizen 100-sign clips, official train and val signers -> data/local/citizen100_v17/live_features/
  <split>/<sign>__<clip>.npz. The official test split is never read.
Uses scripts/extract_fsboard_v17.py Extractor (ffmpeg decode, autorotate, live image sides).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
O5 = ROOT / 'data/local/open_asl_alternatives_20260913/o5s5'
CIT = ROOT / 'data/local/citizen100_v17'


def jobs():
    for v in sorted((O5 / 'videos').iterdir()):
        if v.name.startswith('.') or v.name.startswith('._'):
            continue
        yield v, O5 / 'live_features' / (v.name.split('.')[0] + '.npz')
    for split in ('val', 'train'):
        for d in sorted((CIT / 'raw' / split).iterdir()):
            for v in sorted(d.glob('*.mp4')):
                if not v.name.startswith('._'):
                    yield v, CIT / 'live_features' / split / f'{d.name}__{v.stem}.npz'


def main():
    from scripts.extract_fsboard_v17 import Extractor
    ex = Extractor('ffmpeg', letters=False)
    started, done, failed = time.time(), 0, []
    for video, target in jobs():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            ex.clip(video, dict(clipFilename=video.name, signerId='', phrase='', clipStartTimeS=0.,
                                annotationStartTimeS=0., annotationEndTimeS=1e9), target)
            done += 1
        except Exception as e:
            failed.append((str(video), repr(e)[:120]))
        if done and done % 100 == 0:
            print(f'{done} done, {len(failed)} failed, {(time.time() - started) / 60:.1f} min', flush=True)
    print(f'finished: {done} extracted, {len(failed)} failed {failed[:5]}, {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
