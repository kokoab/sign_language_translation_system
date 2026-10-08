"""Validation-only per-frame raw for HAVE/ANGRY/TIRED/HAPPY: SemLex val and local val (to set the hand-shape
threshold). Output CACHE/isolated_val_raw/{semlex,local}/<label>/<clip>.npz."""
import glob,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.stage1_window_v17 import raw_observation_features
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
OUT=Path('/Users/frnzlo/slt_go_split_cache/isolated_val_raw')
args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
items=[]
for c in ('HAVE','ANGRY','TIRED','HAPPY','GOOD','THANKYOU','BAD'):
    items+=[('semlex',c,v) for v in sorted(glob.glob(str(ROOT/f'data/local/semlex_citizen100_val_audit/raw/{c}/*')))]
    items+=[('local',c,v) for v in sorted(glob.glob(str(ROOT/f'data/local/local_deep_clean_v17/raw/val/{c}/*')))]
for src,c,v in items:
    p=OUT/src/c/(Path(v).stem+'.npz')
    if p.exists():continue
    p.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(v);fps=cap.get(cv2.CAP_PROP_FPS) or 30
    if fps>200 or fps<1:fps=30
    wr={'left':None,'right':None};i=0;nt=0.0;pr=0;obs=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;obs.append(observe_stage2_frame(f,t,pr,det,wr,args));pr+=1
    if len(obs)<4:continue
    raw,times=raw_observation_features(obs);np.savez_compressed(p,raw=raw.astype(np.float16),times=times)
print('done',len(items))
