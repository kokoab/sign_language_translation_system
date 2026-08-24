"""Extract a small set of test videos to .npy once, reuse for all conversion tests.

Picks one isolated sign per a small set of classes + one or two phrases.
"""
from __future__ import annotations
import os, sys, json, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, ARTIFACTS, extract_video

ISOLATED_CLASSES = ['HELLO', 'THANKYOU', 'I', 'YOU', 'HOW', 'GOOD', 'BAD', 'FRIEND', 'NAME', 'LOVE']
PHRASE_CLASSES   = ['HELLO_HOW_YOU', 'MY_NAME']


def pick_one(dirpath: Path) -> Path | None:
    if not dirpath.exists():
        return None
    vids = sorted([p for p in dirpath.iterdir() if p.suffix.lower() == '.mp4'])
    return vids[0] if vids else None


def main():
    isolated_root = REPO / 'data' / 'raw_videos' / 'ASL VIDEOS'
    phrases_root  = REPO / 'data' / 'raw_videos' / 'PHRASES'
    out_dir = ARTIFACTS / 'test_features'
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {'isolated': [], 'phrases': []}

    for cls in ISOLATED_CLASSES:
        vid = pick_one(isolated_root / cls)
        if vid is None:
            print(f'  [skip] {cls}: no videos')
            continue
        out = out_dir / f'{cls}__{vid.stem}.npy'
        if out.exists():
            arr = np.load(out)
            print(f'  [cache] {cls} -> {arr.shape}')
        else:
            print(f'  [extract] {cls}: {vid.name}')
            arr = extract_video(vid)
            if arr is None:
                print(f'    failed (no hands detected)')
                continue
            np.save(out, arr)
            print(f'    saved {out.name} shape={arr.shape}')
        manifest['isolated'].append({'class': cls, 'video': str(vid), 'npy': str(out), 'shape': list(arr.shape)})

    for cls in PHRASE_CLASSES:
        vid = pick_one(phrases_root / cls)
        if vid is None:
            print(f'  [skip] phrase {cls}')
            continue
        out = out_dir / f'PHRASE__{cls}__{vid.stem}.npy'
        if out.exists():
            arr = np.load(out)
            print(f'  [cache] phrase {cls} -> {arr.shape}')
        else:
            print(f'  [extract phrase] {cls}: {vid.name}')
            # Use continuous extraction for phrases
            from src_v16.extract_v16 import extract_continuous_v16
            import cv2
            cap = cv2.VideoCapture(str(vid))
            frames = []
            while True:
                ret, f = cap.read()
                if not ret: break
                frames.append(f)
            cap.release()
            arr = extract_continuous_v16(frames)
            if arr is None:
                print('    failed')
                continue
            arr = arr.astype(np.float32)
            np.save(out, arr)
            print(f'    saved {out.name} shape={arr.shape}')
        manifest['phrases'].append({'class': cls, 'video': str(vid), 'npy': str(out), 'shape': list(arr.shape)})

    (ARTIFACTS / 'test_features_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nDone. Manifest: {ARTIFACTS/"test_features_manifest.json"}')
    print(f'Isolated: {len(manifest["isolated"])}, Phrases: {len(manifest["phrases"])}')


if __name__ == '__main__':
    main()
