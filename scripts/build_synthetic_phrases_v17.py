"""Build synthetic continuous phrases by stitching one signer's own isolated clips.

Experimental, personal-use only. ASL Citizen's documentation cautions against treating
concatenated clips as continuous signing; the user authorized this explicitly for
experimentation. Nothing here is evidence of real continuous recognition and nothing is
promoted. Every output records its provenance and is marked synthetic.

Within-signer only: each synthetic phrase is one real person performing each sign, so no
identity changes mid-phrase. Rest frames are trimmed at clip edges so signs abut the way
they do in continuous signing; no coarticulation is synthesized, and that limitation is
the point of the validity check that follows.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import hashlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import digest

COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
OUT_VIDEO = ROOT / 'data/local/synthetic_phrases_v17_20260922'
REPORT = ROOT / 'artifacts/reports/synthetic_phrases_v17_20260922'
PHRASES = [['TOMORROW', 'SCHOOL', 'GO'], ['PLEASE', 'HELP', 'I'], ['HELLO', 'HOW', 'YOU'],
           ['MY', 'NAME'], ['THANKYOU', 'FRIEND'], ['GOOD', 'MORNING']]
FPS = 30.0
SIZE = (640, 480)
MOTION_QUANTILE = 0.35   # frames below this share of peak motion at the edges are rest
MINIMUM_KEPT = 8         # never trim a clip below this many frames


def read_clip(path):
    capture = cv2.VideoCapture(str(ROOT / path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
    capture.release()
    return frames


def trim_rest(frames):
    """Drop the still lead-in/lead-out so signs abut instead of pausing."""
    if len(frames) <= MINIMUM_KEPT:
        return frames, 0, len(frames)
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    motion = np.array([0.] + [float(np.abs(grey[i] - grey[i - 1]).mean()) for i in range(1, len(grey))])
    if motion.max() <= 1e-6:
        return frames, 0, len(frames)
    threshold = motion.max() * MOTION_QUANTILE
    moving = np.flatnonzero(motion >= threshold)
    if len(moving) == 0:
        return frames, 0, len(frames)
    start, end = int(moving[0]), int(moving[-1]) + 1
    if end - start < MINIMUM_KEPT:
        pad = (MINIMUM_KEPT - (end - start) + 1) // 2
        start, end = max(0, start - pad), min(len(frames), end + pad)
    return frames[start:end], start, end


def build(limit_per_phrase=None):
    OUT_VIDEO.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    combined = json.loads(COMBINED.read_text())
    isolated = [r for r in combined['records']
                if r['source'] in ('citizen', 'semlex', 'stem') and len(r.get('target_sequence') or []) == 1]
    by_signer = defaultdict(lambda: defaultdict(list))
    for row in sorted(isolated, key=lambda r: r['video_sha256']):
        by_signer[(row['source'], row.get('signer_id'))][row['target_sequence'][0]].append(row)

    records, skipped = [], []
    for phrase in PHRASES:
        made = 0
        for signer, glosses in sorted(by_signer.items(), key=lambda kv: str(kv[0])):
            if not all(g in glosses for g in phrase):
                continue
            if limit_per_phrase is not None and made >= limit_per_phrase:
                break
            sources, frames, intervals = [], [], []
            failed = False
            for gloss in phrase:
                row = glosses[gloss][0]
                if digest(ROOT / row['video_path']) != row['video_sha256']:
                    failed = True
                    break
                clip = read_clip(row['video_path'])
                if not clip:
                    failed = True
                    break
                kept, s, e = trim_rest(clip)
                start_seconds = len(frames) / FPS
                frames.extend(kept)
                intervals.append(dict(gloss=gloss, start=start_seconds, end=len(frames) / FPS,
                                      frames=len(kept), source_frames=len(clip), trimmed=(s, e),
                                      source_video=row['video_path'], source_sha256=row['video_sha256']))
                sources.append(row['video_path'])
            if failed or not frames:
                skipped.append(dict(signer=list(signer), phrase=phrase, reason='unreadable or hash mismatch'))
                continue
            name = hashlib.sha256(('|'.join(sources)).encode()).hexdigest()[:12]
            target = OUT_VIDEO / ('_'.join(phrase) + '__' + name + '.mp4')
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), FPS, SIZE)
            for frame in frames:
                writer.write(frame)
            writer.release()
            if not target.exists() or target.stat().st_size == 0:
                skipped.append(dict(signer=list(signer), phrase=phrase, reason='writer produced no file'))
                continue
            records.append(dict(item='synthetic:' + '_'.join(phrase) + ':' + name,
                                synthetic=True, provenance='within-signer stitched isolated clips; '
                                'rest frames trimmed; NO coarticulation synthesized',
                                source=signer[0], signer_id=signer[1], target_sequence=phrase,
                                video_path=str(target.relative_to(ROOT)), video_sha256=digest(target),
                                frames=len(frames), fps=FPS, intervals=intervals))
            made += 1
        print('%-22s %d built' % (' '.join(phrase), made), flush=True)

    manifest = dict(
        format='synthetic_phrases_v17', synthetic=True, promoted=False, training_ready=False,
        caution='ASL Citizen documentation cautions against treating concatenated clips as continuous '
                'signing. Built for personal experimentation at explicit user request. Not evidence of '
                'real continuous recognition; never mix into a real-data evaluation.',
        method='within-signer concatenation of that signer\'s own isolated clips, rest frames trimmed '
               'by frame-difference motion threshold; hard cuts, no transition synthesis',
        parameters=dict(fps=FPS, size=list(SIZE), motion_quantile=MOTION_QUANTILE, minimum_kept=MINIMUM_KEPT),
        source_manifest=str(COMBINED.relative_to(ROOT)), source_manifest_sha256=digest(COMBINED),
        counts=dict(records=len(records), signers=len({r['signer_id'] for r in records}),
                    reference_signs=sum(len(r['target_sequence']) for r in records)),
        skipped=skipped, records=records)
    (REPORT / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('\n%d phrases, %d distinct signers, %d reference signs' % (
        len(records), len({r['signer_id'] for r in records}),
        sum(len(r['target_sequence']) for r in records)))
    print('wrote', REPORT / 'manifest.json')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-per-phrase', type=int, default=None)
    build(parser.parse_args().limit_per_phrase)
