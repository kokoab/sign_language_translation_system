"""Finger spread (fingers together vs claw) for HAVE vs ANGRY (and TIRED/HAPPY): isolated train clips (per-frame
raw), Citizen val, and the user's in-sentence HAVE spans. spread = mean adjacent fingertip distance (8-12,12-16,
16-20) / hand size (wrist-middle MCP); median over the active frames' middle half, mean over visible hands."""
import glob,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Volumes/secret/SLT/SLT')
def spread_frames(raw):
    out=[]
    for f in raw:
        vals=[]
        for st in (0,21):
            h=f[st:st+21];
            if h[:,4].min()<.3:continue
            xy=h[:,:2].astype(np.float64);size=np.linalg.norm(xy[9]-xy[0])
            if size<1e-3:continue
            tips=[np.linalg.norm(xy[a]-xy[b]) for a,b in ((8,12),(12,16),(16,20))]
            vals.append(np.mean(tips)/size)
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)
def clip_value(raw,a=None,b=None):
    s=spread_frames(raw if a is None else raw[a:b+1]);s=s[~np.isnan(s)]
    if len(s)<2:return None
    k=np.convolve(s,np.ones(3)/3,mode="valid") if len(s)>=3 else s;return float(k.min())
R={}
for c in ('HAVE','ANGRY','TIRED','HAPPY'):
    for src in ('citizen','semlex','local'):
        v=[clip_value(np.load(f)['raw'].astype(np.float32)) for f in glob.glob(f'/Users/frnzlo/slt_go_split_cache/isolated_raw/{src}/{c}/*.npz')]
        v=[x for x in v if x is not None]
        if v:R[(c,src)]=v;print(f'{c:6s} {src:8s} n={len(v):3d} spread median {np.median(v):.3f}  p10 {np.percentile(v,10):.3f} p90 {np.percentile(v,90):.3f}')
    v=[clip_value(np.load(f)['raw'].astype(np.float32)) for f in glob.glob(f'/Users/frnzlo/slt_go_split_cache/isolated_val_raw/citizen/{c}/*.npz')]
    v=[x for x in v if x is not None];print(f'{c:6s} cit-val  n={len(v):3d} values {np.round(sorted(v),3)}')
# user spans
from active.v17.stage1_window_v17 import raw_observation_features
import cv2
item=pickle.load(open('/Users/frnzlo/slt_go_split_cache/takes2/20261009_135024_11.pkl','rb'))
ok,fr=cv2.VideoCapture(item['meta']['video']).read();view=np.broadcast_to(np.zeros((1,1,1),np.uint8),fr.shape)
for o in item['obs']:o.frame=view
raw,t=raw_observation_features(item['obs'])
words=json.load(open('have_probe_words.json'))
for w in words:
    if w['g'] in ('ANGRY','HAVE','TIRED'):
        print(f"user {w['g']:6s}@{t[w['s']]:5.2f} (truth HAVE) spread {clip_value(raw,w['s'],w['e']):.3f}")
