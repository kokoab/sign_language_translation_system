"""Frozen pretrained DGS boundaries + Reel on the pinned ASLLRP development videos."""
import __future__
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE/'dependencies')]
import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from pose_format.utils.holistic import load_holistic
from check import model_from_upstream
from scripts.evaluate_temporal_boundary_v17 import evaluation_rows, arguments, observations, edit_counts, summarize
from scripts.live_reel_stage1_v17 import build_components
from scripts.live_boundary_v17 import classify_interval
from active.v17.extract_v17 import AppleVisionDetector


def upstream_helpers():
    namespace = {'BIO': {'UNK':0,'O':1,'B':2,'I':3}}
    for name in ('utils/pose.py', 'metrics.py'):
        text = (HERE/'source/sign_language_segmentation'/name).read_text()
        text = '\n'.join(s for s in text.splitlines() if not s.startswith('from sign_language_segmentation.'))
        exec(compile(text,name,'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace


def main():
    start = time.perf_counter()
    helpers = upstream_helpers()
    weights = ROOT/'artifacts/models/pose_boundary_dgs_2026'
    model = model_from_upstream(json.loads((weights/'config.json').read_text())).float().eval()
    model.load_state_dict(load_file(str(weights/'model.safetensors')),strict=True)
    model.to('mps')
    rows = evaluation_rows()
    args = arguments(rows[0]['video_path'],HERE/'sessions');args.no_motion_trim=True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)
    records=[]
    for row in rows:
        capture = cv2.VideoCapture(str(ROOT/row['video_path']))
        fps=capture.get(cv2.CAP_PROP_FPS); frames=[]
        while True:
            ok,frame=capture.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        capture.release()
        assert frames and fps>0
        tick=time.perf_counter()
        height,width=frames[0].shape[:2]
        pose=load_holistic(frames,fps=fps,width=width,height=height,pose_workers=1,reuse=False,
                           additional_holistic_config={'static_image_mode':False,'model_complexity':1})
        processed=helpers['preprocess_pose'](pose)
        data=processed.body.data.filled(0)[:,0,:,:3].astype('float32')
        times=np.arange(len(data),dtype='float32')/fps
        features=np.concatenate([data,helpers['compute_velocity'](data,times)],axis=-1)
        assert features.shape[1:]==(50,6) and np.isfinite(features).all()
        frontend=time.perf_counter()-tick
        tick=time.perf_counter()
        with torch.inference_mode():
            logits=model(torch.from_numpy(features)[None].to('mps'),timestamps=torch.from_numpy(times)[None].to('mps'))['sign'][0].cpu()
        elapsed=time.perf_counter()-tick
        segments=helpers['filter_segments'](helpers['likeliest_probs_to_segments'](logits))
        obs=observations(row,args,detector)
        predictions=[classify_interval(reel,obs,dict(start_seconds=s['start']/fps,end_seconds=s['end']/fps),args,.1) for s in segments]
        hypothesis=[p['committed_gloss'] for p in predictions if p['committed_gloss']]
        record=dict(id=row['source_item_id'],reference=row['target_sequence'],hypothesis=hypothesis,
                    metrics=edit_counts(row['target_sequence'],hypothesis),predictions=predictions,
                    segments=segments,frames=len(frames),fps=fps,frontend_seconds=frontend,model_seconds=elapsed,
                    bio_counts=np.bincount(logits.argmax(-1).numpy(),minlength=4).tolist())
        records.append(record)
        value=dict(records=records,summary=summarize(records),elapsed_seconds=time.perf_counter()-start,
                   scope='Full-video noncausal frozen transfer; reused development only; no training or default changes.',
                   frontend='MediaPipe0.10.14 Holistic model_complexity1, native fps/resolution, aspect preserved; pose-format0.15.0 and pose-anonymization0.0.1 upstream normalization',
                   input_code_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'source').rglob('*.py')})
        (HERE/'probe_results.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(dict(id=record['id'],reference=record['reference'],hypothesis=hypothesis,segments=len(segments))),flush=True)
    assert len(records)==12 and value['summary']['references']==24
    print(json.dumps(value['summary']),flush=True)

if __name__=='__main__':main()
