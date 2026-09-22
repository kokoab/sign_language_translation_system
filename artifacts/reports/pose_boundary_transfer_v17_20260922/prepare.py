"""Cache native MediaPipe poses from the existing admitted boundary source manifest.

No training. Raw reduced poses preserve the option to normalize a bounded window later;
whole-clip normalization is intentionally not baked into fine-tuning inputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE/'dependencies')]
import cv2
import numpy as np
from pose_format import Pose
from pose_format.utils.holistic import load_holistic
from pose_format.utils.generic import pose_hide_legs, reduce_holistic
from active.v17.approved_phrase_data_v17 import digest, verify_manifest


def atomic(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def extract(row, cache):
    video=ROOT/row['video_path']
    if digest(video)!=row['video_sha256']:
        raise ValueError('source video hash mismatch: '+str(video))
    capture=cv2.VideoCapture(str(video))
    fps=capture.get(cv2.CAP_PROP_FPS)
    width,height=(int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or not np.isfinite(fps) or fps<=0 or min(width,height)<=0:
        capture.release();raise ValueError('invalid source video')
    count=0
    def frames():
        nonlocal count
        while True:
            ok,frame=capture.read()
            if not ok: break
            count+=1
            yield cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    try:
        pose=load_holistic(frames(),fps=fps,width=width,height=height,pose_workers=1,reuse=False,
                           additional_holistic_config={'static_image_mode':False,'model_complexity':1})
    finally:
        capture.release()
    pose=pose.get_components(['POSE_LANDMARKS','LEFT_HAND_LANDMARKS','RIGHT_HAND_LANDMARKS'])
    pose_hide_legs(pose);pose=reduce_holistic(pose)
    assert pose.body.data.shape==(count,1,50,3)
    assert np.isfinite(pose.body.data.filled(0)).all()
    for start,end in row['intervals']:
        if start<-.05 or end>(count-1)/fps+.05:
            raise ValueError('annotation exceeds decoded video coverage')
    if digest(video)!=row['video_sha256']:raise ValueError('video changed during extraction')
    target=cache/(hashlib.sha256(row['item'].encode()).hexdigest()[:20]+'.pose')
    temp=target.with_suffix('.pose.tmp')
    with temp.open('wb') as f:pose.write(f)
    temp.replace(target)
    return dict(**row,pose_path=str(target.relative_to(ROOT)),pose_sha256=digest(target),
                pose_frames=count,pose_fps=fps,width=width,height=height)


def resume_entries(previous, manifest, source_hash):
    if previous['source_sha256'] != source_hash:
        raise ValueError('resume source manifest changed')
    source_rows = {r['item']: r for r in manifest['records']}
    entries = {}
    for row in previous['records']:
        original = source_rows.get(row['item'])
        if original is None or any(row.get(k) != v for k, v in original.items()):
            raise ValueError('resume source record changed')
        if row['item'] in entries:
            raise ValueError('duplicate resumed source')
        if digest(ROOT / row['video_path']) != row['video_sha256'] or digest(ROOT / row['pose_path']) != row['pose_sha256']:
            raise ValueError('resume video/pose hash mismatch')
        with (ROOT / row['pose_path']).open('rb') as f:
            pose = Pose.read(f)
        if pose.body.data.shape != (row['pose_frames'], 1, 50, 3):
            raise ValueError('resume pose shape mismatch')
        if abs(pose.body.fps - row['pose_fps']) > .001 or not np.isfinite(pose.body.data.filled(0)).all():
            raise ValueError('resume pose clock/values mismatch')
        entries[row['item']] = row
    return entries


def main(smoke=False, workers=3):
    if not 1 <= workers <= 3:
        raise ValueError('use one to three independent video workers')
    started=time.perf_counter()
    source=ROOT/'active/v17/temporal_boundary_manifest_20260922.json'
    source_hash=digest(source);manifest=json.loads(source.read_text())
    verify_manifest()
    assert len(manifest['records'])==1121
    assert {r['role'] for r in manifest['records']}=={'train','validation'}
    cache=ROOT/'artifacts/cache/pose_boundary_dgs_2026'
    if smoke: cache=cache/'parallel_smoke'
    cache.mkdir(parents=True,exist_ok=True)
    pins={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),*sorted((HERE/'dependencies').rglob('*.py'))]}
    selection=manifest['records'][:workers] if smoke else manifest['records']
    target=HERE/('preparation_parallel_smoke.json' if smoke else 'prepared_manifest.json')
    entries={};resume_provenance=None
    if not smoke and target.exists():
        previous=json.loads(target.read_text())
        old_code=previous['code_sha256'][str(Path(__file__).relative_to(ROOT))]
        if old_code not in (digest(Path(__file__)), digest(HERE/'prepare_serial.py')):
            raise ValueError('unrecognized extraction code in resume manifest')
        entries=resume_entries(previous,manifest,source_hash)
        resume_provenance=dict(manifest_sha256=digest(target),code_sha256=old_code,records=len(entries))
    resumed=len(entries)
    def checkpoint(status):
        atomic(target,dict(status=status,training_ready=False,
            training_blocker='Review bounded-context normalization and supervised start/end transfer recipe; no automatic BIO background labels from unknown gaps.',
            source_manifest=str(source.relative_to(ROOT)),source_sha256=source_hash,
            records=[entries[r['item']] for r in selection if r['item'] in entries],
            code_sha256=pins,resume_provenance=resume_provenance,workers=workers,
            elapsed_seconds=time.perf_counter()-started,
            contract='MediaPipe0.10.14 Holistic complexity1; native fps/resolution; reduced50 XYZ/confidence; no normalization; original roles/intervals preserved.'))
    todo=[r for r in selection if r['item'] not in entries]
    checkpoint('preparing')
    print(f'Reusing {resumed} verified poses; extracting {len(todo)} videos with {workers} workers.',flush=True)
    # Processes isolate MediaPipe trackers; frames within each video remain serial.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(extract,row,cache) for row in todo]
        try:
            for future in as_completed(futures):
                entry=future.result();entries[entry['item']]=entry
                checkpoint('preparing')
                print(f'{len(entries)}/{len(selection)} {entry["item"]}',flush=True)
        except BaseException:
            for future in futures: future.cancel()
            checkpoint('interrupted')
            raise
    assert digest(source)==source_hash
    assert all(digest(ROOT/path)==sha for path,sha in pins.items())
    checkpoint('complete')
    return dict(status='complete',records=len(entries),resumed=resumed,newly_extracted=len(todo),
                workers=workers,elapsed_seconds=time.perf_counter()-started,training_started=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    smoke=args.smoke
    message='ASL pretrained boundary pose preparation failed; inspect preparation_completion.json.'
    try:
        result=main(smoke,args.workers)
        if not smoke:atomic(HERE/'preparation_completion.json',result)
        message='ASL pretrained boundary pose preparation complete; inputs ready for recipe review. No training started.'
    except BaseException:
        if not smoke:atomic(HERE/'preparation_completion.json',dict(status='failed',traceback=traceback.format_exc(),training_started=False))
        raise
    finally:
        if not smoke:
            subprocess.run(['osascript','-e','display notification '+json.dumps(message)+' with title "SLT pretrained boundary"'],check=False)
