"""Cache decoder inputs once per video: Apple Vision observations without pixels + per-frame hand-crop
embeddings (the installed MobileCLIP2 encoder). Sets: takes (user phone takes), citizen_val (378, never
test), tune (89 local tuning phrases, never test). Output: CACHE/<set>/<key>.pkl = dict(obs, hands, meta).
Resumable; low priority."""
import glob,hashlib,json,pickle,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
CACHE=Path('/Users/frnzlo/slt_go_split_cache')
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.segmental_runtime_v17 import build_runtime
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
CFG=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/user_take_replay/config_v2_installed.json'
def videos(name):
    if name=='takes':
        return [(Path(v).stem,v,None) for v in sorted(glob.glob(str(ROOT/'data/local/phone_session_pull_v17_20261009/live_reel_videos/20261009_082836_0*.mp4')))]
    if name=='takes2':
        return [(Path(v).stem,v,None) for v in sorted(glob.glob(str(ROOT/'data/local/phone_session_pull_v17_20261009b/live_reel_videos/20261009_135024_*.mp4')))]
    if name=='takes3':
        return [(Path(v).stem,v,None) for v in sorted(glob.glob(str(ROOT/'data/local/phone_session_pull_v17_20261009c/live_reel_videos/20261009_201823_*.mp4')))]
    if name=='takes4':
        return [(Path(v).stem,v,None) for v in sorted(glob.glob(str(ROOT/'data/local/phone_session_pull_v17_20261010/live_reel_videos/20261010_095533_*.mp4')))]
    if name=='pair_val':
        vs=[]
        for c in ('HAVE','ANGRY','TIRED','HAPPY'):
            vs+=[(f'semlex__{c}__'+Path(v).stem,v,[c]) for v in sorted(glob.glob(str(ROOT/f'data/local/semlex_citizen100_val_audit/raw/{c}/*')))]
            vs+=[(f'local__{c}__'+Path(v).stem,v,[c]) for v in sorted(glob.glob(str(ROOT/f'data/local/local_deep_clean_v17/raw/val/{c}/*')))]
        assert not any('/test/' in v for _,v,_ in vs);return vs
    if name=='citizen_val':
        vs=sorted(glob.glob(str(ROOT/'data/local/citizen100_v17/raw/val/*/*')));assert vs and not any('/test/' in v for v in vs)
        return [(Path(v).parent.name+'__'+Path(v).stem,v,[Path(v).parent.name]) for v in vs]
    if name=='tune':
        from scripts import segmental_lab_v17 as lab
        rows=lab.rows_for('tune');test={r['video_sha256'] for r in lab.rows_for('test')}
        assert not any(r['video_sha256'] in test for r in rows)
        return [(lab.key(r['source_item_id']),str(ROOT/r['video_path']),(r['target_sequence'] if isinstance(r['target_sequence'],list) else __import__('ast').literal_eval(r['target_sequence']))) for r in rows]
    raise ValueError(name)
def main(name):
    import os
    phase=float(os.environ.get('PHASE','0'))
    out=CACHE/(name+(f'_phase{int(round(phase*1000))}' if phase else ''));out.mkdir(parents=True,exist_ok=True)
    rec=build_runtime(CFG,device='mps').words.recognizer
    args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
    vids=videos(name)
    for n,(key,vid,target) in enumerate(vids):
        path=out/f'{key}.pkl'
        if path.exists():continue
        cap=cv2.VideoCapture(vid);fps=cap.get(cv2.CAP_PROP_FPS) or 30;wr={'left':None,'right':None};i=0;nt=phase;pr=0;obs=[];hands=[]
        while True:
            ok,f=cap.read()
            if not ok:break
            t=i/fps;i+=1
            if t+1e-9<nt:continue
            nt+=0.05;o=observe_stage2_frame(f,t,pr,det,wr,args);pr+=1
            hands.append(rec.frame_hand(o));o.frame=None;obs.append(o)
        tmp=path.with_suffix('.tmp');pickle.dump(dict(obs=obs,hands=hands,meta=dict(video=vid,target=target)),open(tmp,'wb'));tmp.replace(path)
        if n%20==0:print(name,n,len(vids),flush=True)
    print(name,'done',flush=True)
if __name__=='__main__':
    for s in sys.argv[1:]:main(s)
