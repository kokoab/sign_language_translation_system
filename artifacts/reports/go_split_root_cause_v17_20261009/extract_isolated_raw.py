"""Per-frame Apple Vision raw observation features (20 Hz, the live sampling) for isolated training clips:
Citizen train (all), SemLex clean train candidates (all), local Citizen-variant v2 train (15/class, seeded).
No validation/test clip is read here. Output: CACHE/isolated_raw/<source>/<label>/<stem>.npz (raw f16, times,
trim_start/trim_end from the recognizer's motion trim). Resumable; low priority."""
import glob,json,random,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
OUT=Path('/Users/frnzlo/slt_go_split_cache/isolated_raw')
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.stage1_window_v17 import raw_observation_features
from active.v17.schema_v17 import V17Config
from scripts.live_isolated_v17 import landmarks_from_observations
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
def clips():
    out=[('citizen',Path(v).parent.name,v) for v in sorted(glob.glob(str(ROOT/'data/local/citizen100_v17/raw/train/*/*')))]
    assert not any('/val/' in v or '/test/' in v for _,_,v in out)
    s=json.loads((ROOT/'data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json').read_text())
    out+=[('semlex',v['canonical_label'],str(ROOT/v['raw_path'])) for v in s['videos']]
    m=json.loads((ROOT/'data/local/web_variant_supplement_v17_20261008/local_train_citizen_variant_v2.json').read_text())
    by={}
    for v in m['videos']:
        if v['consensus_tier']=='owner_approved_v16_deep_clean' and v.get('training_eligible',True):by.setdefault(v['canonical_label'],[]).append(v)
    rng=random.Random(20261009)
    for c in sorted(by):
        vs=sorted(by[c],key=lambda v:v['item_id']);rng.shuffle(vs)
        out+=[('local',c,str(ROOT/v['raw_path'])) for v in vs[:15]]
    return out
def main():
    args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence);cfg=V17Config()
    items=clips();print('clips',len(items),flush=True)
    for n,(src,label,vid) in enumerate(items):
        path=OUT/src/label/(Path(vid).stem+'.npz')
        if path.exists():continue
        path.parent.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(vid);fps=cap.get(cv2.CAP_PROP_FPS) or 30;wr={'left':None,'right':None};i=0;nt=0.0;pr=0;obs=[]
        if fps>200 or fps<1:fps=30
        while True:
            ok,f=cap.read()
            if not ok:break
            t=i/fps;i+=1
            if t+1e-9<nt:continue
            nt+=0.05;obs.append(observe_stage2_frame(f,t,pr,det,wr,args));pr+=1
        if len(obs)<4:
            np.savez_compressed(path,raw=np.zeros((0,61,5),np.float16),times=np.zeros(0),trim_start=-1,trim_end=-1);continue
        raw,times=raw_observation_features(obs)
        try:
            _,d=landmarks_from_observations(obs,cfg);a,b=int(d['trim_start']),int(d['trim_end_exclusive'])
        except ValueError:a=b=-1
        np.savez_compressed(path,raw=raw.astype(np.float16),times=times,trim_start=a,trim_end=b)
        if n%100==0:print(n,len(items),flush=True)
    print('done',flush=True)
main()
