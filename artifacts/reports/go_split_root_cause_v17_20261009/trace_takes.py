"""Instrumented Mac replay of the user's takes: per-frame BIO, frame seconds, every scored span's raw
logits (words+letters) and the committed words, for the installed v2 recognizer in phone word mode
(letter gate 0.95). Output: trace/<take>.pkl. Cached observations: obs/<take>.pkl (Vision pass once)."""
import copy,pickle,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(ROOT))
from active.v17.av_boundary_v17 import AVBoundary
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.segmental_runtime_v17 import SegmentalRuntime,build_runtime
from scripts.live_reel_stage1_v17 import parser
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
HERE=Path(__file__).resolve().parent;(HERE/'trace').mkdir(exist_ok=True);(HERE/'obs').mkdir(exist_ok=True)
CFG=ROOT/'artifacts/reports/citizen_variant_local_filter_v17_20261008/user_take_replay/config_v2_installed.json'
words=build_runtime(CFG,device='mps').words
def boundary():
    p=torch.load(ROOT/words.config['boundary'],map_location='cpu',weights_only=False);b=AVBoundary(**p['config']);b.load_state_dict(p['state_dict']);return b.eval()
def observations(vid):
    path=HERE/'obs'/(Path(vid).stem+'.pkl')
    if path.exists():return pickle.load(open(path,'rb'))
    args=parser().parse_args([]);det=AppleVisionDetector(args.minimum_point_confidence)
    cap=cv2.VideoCapture(vid);fps=cap.get(cv2.CAP_PROP_FPS) or 30;wr={'left':None,'right':None};i=0;nt=0.0;pr=0;obs=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        t=i/fps;i+=1
        if t+1e-9<nt:continue
        nt+=0.05;obs.append(observe_stage2_frame(f,t,pr,det,wr,args));pr+=1
    pickle.dump(obs,open(path,'wb'));return obs
def run(obs,min_span,tag,hands=None):
    cfg=copy.deepcopy(words.config);cfg['stream']['min_span_frames']=min_span
    rt=SegmentalRuntime(boundary(),words.recognizer,cfg,mode='both',letter_gate=0.95);cache={};rt.raw_cache=cache
    bio={};out=[];H=[]
    for k,o in enumerate(obs):
        out+=rt.observe(o,hands=None if hands is None else hands[k]);H.append(rt.hands[-1])
        for j,row in enumerate(rt.bio):bio[rt.bio_off+j]=np.asarray(row)
    out+=rt.finish()
    for j,row in enumerate(rt.bio):bio[rt.bio_off+j]=np.asarray(row)
    return dict(tag=tag,bio=bio,cache={k:v for k,v in cache.items()},words=out,seconds=[o.seconds for o in obs]),H
if __name__=='__main__':
    for vid in sys.argv[1:]:
        obs=observations(vid);res={};H=None
        for m in (3,6):
            r,H2=run(obs,m,f'min{m}',H);H=H or H2;res[f'min{m}']=r
            print(Path(vid).name,m,' '.join(w['gloss'] for w in r['words'] if not w['gloss'].startswith('FS_')),flush=True)
        res['labels']=words.labels;res['hands']=H
        pickle.dump(res,open(HERE/'trace'/(Path(vid).stem+'.pkl'),'wb'))
