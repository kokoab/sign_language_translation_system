#!/usr/bin/env python3
"""Cache real train/development videos using the actual live Stage2 input path."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.live_reel_continuous_v17 import parser as live_parser
from scripts.live_stage2_ctc_v17 import LiveStage2CTC, ElapsedWindowBuffer, observe_stage2_frame, stage2_landmarks_from_observations
from scripts.live_isolated_v17 import hand_inputs
from active.v17.extract_v17 import AppleVisionDetector,orient_frame
from active.v17.train_stage_2_other_ctc_v17 import sha256
from scripts.extract_stage2_multimodal_v17 import safe_name

def input_contract(args):
    functions=[observe_stage2_frame,stage2_landmarks_from_observations,hand_inputs,ElapsedWindowBuffer]
    return dict(processing_fps=args.processing_fps,detection_image_side=args.detection_image_side,
        maximum_image_side=args.maximum_image_side,minimum_point_confidence=args.minimum_point_confidence,
        window_seconds=args.stage2_window_seconds,dense_model_auxiliary=False,
        observer_sha256=hashlib.sha256(''.join(inspect.getsource(f) for f in functions).encode()).hexdigest(),
        timebase='source timestamps, same live processing scheduler; no synthetic rate change',
        lip_nodes='observed as in live input; no source-specific masking',
        encoder_sha256='1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b')

def extract_live_features(row,args,classifier):
    capture=cv2.VideoCapture(str(ROOT/row['video_path']))
    if not capture.isOpened():raise ValueError('cannot open '+row['video_path'])
    fps=capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=1:raise ValueError('invalid video FPS')
    detector=AppleVisionDetector(args.minimum_point_confidence)
    wrists={'left':None,'right':None};buffer=ElapsedWindowBuffer(args.stage2_window_seconds)
    processed=frame_index=0;next_process=0.;features=[];windows=[];last_result=None
    def encode(window):
        nonlocal last_result
        if len(features)>=8:raise ValueError('source exceeds eight-window training context')
        feature,result=classifier.classify_window(window,features)
        windows.append(dict(start=window[0].seconds,end=window[-1].seconds,frames=len(window),accepted=feature is not None,diagnostics=result.get('diagnostics')))
        if feature is not None:features.append(feature.copy());last_result=result
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            seconds=frame_index/fps;frame_index+=1
            if seconds+1e-6<next_process:continue
            canonical=orient_frame(frame,args.rotation,args.input_mirrored)
            item=observe_stage2_frame(canonical,seconds,processed,detector,wrists,args)
            processed+=1;next_process=max(next_process+1./args.processing_fps,seconds)
            for side,hand in item.assigned.items():
                if hand is not None and hand.confidence[0]>0:wrists[side]=hand.xy[0].copy()
            window=buffer.add(item)
            if window is not None:encode(window)
        tail=buffer.finish()
        if tail is not None:encode(tail)
    finally:capture.release()
    return np.asarray(features,dtype=np.float16),dict(windows=windows,observations=processed,source_frames=frame_index,source_fps=fps,baseline_prediction=[] if last_result is None else last_result['hypothesis'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/local/stage2_v17_live_matched_v1')
    parser.add_argument('--report',type=Path,default=ROOT/'artifacts/reports/stage2_v17_live_matched_v1')
    args=parser.parse_args();torch.set_num_threads(2)
    live=live_parser().parse_args(['--no-display','--no-speech','--naturalizer','literal','--sequence-preview'])
    contract=input_contract(live)
    paths=[ROOT/'active/v17/stage2_training_manifest_v17.json',ROOT/'active/v17/stage2_asllrp_other_ctc_manifest_v17.json']
    rows=[r for p in paths for r in json.loads(p.read_text())['rows'] if r['role'] in {'train','validation'}]
    rows.sort(key=lambda r:(r['role']!='validation',r['source']!='asllrp_contiguous',r['source'],r['source_item_id']))
    assert len({r['source_item_id'] for r in rows})==len(rows)
    assert all('test' not in Path(r['video_path']).parts and 'external_evaluation_reserved' not in Path(r['video_path']).parts for r in rows)
    args.output.mkdir(parents=True,exist_ok=True);args.report.mkdir(parents=True,exist_ok=True)
    frozen=dict(contract=contract,manifests={str(p.relative_to(ROOT)):sha256(p) for p in paths},rows=rows,counts={str(k):v for k,v in Counter((r['source'],r['role']) for r in rows).items()})
    manifest_path=args.report/'frozen_inputs.json'
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==frozen,'input contract changed; use new output'
    else:manifest_path.write_text(json.dumps(frozen,indent=2)+'\n')
    classifier=LiveStage2CTC(live);results=[];started=time.monotonic()
    for index,row in enumerate(rows):
        output=args.output/'frozen_features'/row['role']/row['source']/(safe_name(row)+'.stage2_frozen_v17.npz')
        diagnostic=args.output/'diagnostics'/row['role']/(safe_name(row)+'.json')
        if output.exists() and diagnostic.exists():
            with np.load(output,allow_pickle=False) as z:
                md=json.loads(str(z['metadata_json']));assert md['input_contract']==contract and md['video_sha256']==row['video_sha256']
            result=json.loads(diagnostic.read_text())
        else:
            assert sha256(ROOT/row['video_path'])==row['video_sha256'],'source video changed'
            features,details=extract_live_features(row,live,classifier)
            result=dict(item_id=row['source_item_id'],role=row['role'],source=row['source'],reference=row['target_sequence'],feature_windows=len(features),**details)
            if len(features):
                output.parent.mkdir(parents=True,exist_ok=True)
                metadata=dict(source=row['source'],source_item_id=row['source_item_id'],role=row['role'],video_sha256=row['video_sha256'],target_sequence=row['target_sequence'],stage1_checkpoint_sha256=contract['encoder_sha256'],input_contract=contract)
                temp=output.with_suffix('.tmp.npz');np.savez_compressed(temp,frozen_features=features,target_indices=np.asarray(row['target_indices'],dtype=np.int64),metadata_json=np.array(json.dumps(metadata)));temp.replace(output)
            diagnostic.parent.mkdir(parents=True,exist_ok=True);diagnostic.write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        if index%10==0 or index+1==len(rows):
            (args.report/'cache_progress.json').write_text(json.dumps(dict(completed=len(results),total=len(rows),elapsed_seconds=time.monotonic()-started,results=results),indent=2)+'\n')
            print(index+1,'/',len(rows),row['role'],row['source'],'windows',result['feature_windows'],'seconds',round(time.monotonic()-started,1),flush=True)
    (args.report/'cache.json').write_text(json.dumps(dict(format='slt_stage2_live_matched_cache_v17',contract=contract,rows=results,protected_test_accessed=False),indent=2)+'\n')

if __name__=='__main__':main()
