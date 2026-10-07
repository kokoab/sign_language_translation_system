"""Landmark screen of GOOD / THANKYOU variants across local isolated datasets (read-only).

Two-handed GOOD (user's target): the dominant flat hand starts at the chin and finishes on the
non-dominant palm held in front of the torso. One-handed GOOD (ASL-LEX B_01_052, the pinned
Citizen code) and THANKYOU (H_02_053) move from the chin/mouth outward without a base hand.
This is a screen; every decision is visually checked on contact sheets before any use.
Citizen test is excluded by construction.
"""
import csv,json,glob
from pathlib import Path
import numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent
L=ROOT/'data/local'
SOURCES={
 'citizen_train':(L/'citizen100_v17/landmarks/train',L/'citizen100_v17/raw/train'),
 'citizen_val':(L/'citizen100_v17/landmarks/val',L/'citizen100_v17/raw/val'),
 'semlex_train':(L/'semlex_citizen100_train_audit/full_clean_landmarks_v17',L/'semlex_citizen100_train_audit/raw'),
 'semlex_val':(L/'semlex_citizen100_val_audit/landmarks_v17',L/'semlex_citizen100_val_audit/raw'),
 'local_train':(L/'local_deep_clean_v17/landmarks/train',L/'local_deep_clean_v17/raw/train'),
 'local_val':(L/'local_deep_clean_v17/landmarks/val',L/'local_deep_clean_v17/raw/val'),
 'msasl_audit':(L/'msasl_citizen100_gap_audit/landmarks',L/'msasl_citizen100_gap_audit/raw'),
 'popsign_audit':(L/'popsign_citizen100_variant_audit/landmarks',L/'popsign_citizen100_variant_audit/raw'),
}
PALM=[0,5,9,13,17];CHIN=54
def palm(f,start):
    xy=f[:,[start+i for i in PALM],:2].mean(1);pres=f[:,start,3]>0.5
    return xy,pres
def features(path):
    with np.load(path,allow_pickle=False) as p:f=p['features'].astype(np.float32)
    chin=f[:,CHIN,:2];chin_ok=f[:,CHIN,3]>0.5
    hands=[palm(f,0),palm(f,21)]
    def chin_dist(h):
        xy,pres=h;ok=pres&chin_ok
        return float(np.min(np.linalg.norm(xy[ok]-chin[ok],axis=1))) if ok.any() else np.inf
    d=[chin_dist(h) for h in hands]
    dom=int(np.argmin(d)) if np.isfinite(min(d)) else int(np.argmax([h[1].mean() for h in hands]))
    (dxy,dp),(bxy,bp)=hands[dom],hands[1-dom]
    both=dp&bp;late=np.zeros(len(f),bool);late[int(len(f)*0.5):]=True
    contact=float(np.min(np.linalg.norm(dxy[both&late]-bxy[both&late],axis=1))) if (both&late).any() else np.inf
    base_y=float(np.median(bxy[bp,1])) if bp.any() else np.nan
    base_motion=float(np.linalg.norm(bxy[bp].max(0)-bxy[bp].min(0))) if bp.sum()>1 else np.nan
    dom_drop=float(dxy[dp,1][-max(1,dp.sum()//4):].mean()-dxy[dp,1][:max(1,dp.sum()//4)].mean()) if dp.any() else np.nan
    return dict(dominant='left' if dom==0 else 'right',dom_presence=float(dp.mean()),base_presence=float(bp.mean()),
                dom_min_chin=d[dom],late_contact=contact,base_y=base_y,base_motion=base_motion,dom_drop=dom_drop)
def screen(r):
    if r['dom_min_chin']>0.75:return 'unclear_no_chin'
    if r['base_presence']>=0.35 and r['late_contact']<=0.45 and (np.isnan(r['base_y']) or r['base_y']>-0.1):
        return 'two_handed_base'
    if r['base_presence']>=0.35 and r['late_contact']<=0.8:return 'borderline'
    if r['base_presence']>=0.5 and r['late_contact']>0.8 and r['base_motion']>0.4:return 'two_handed_symmetric'
    return 'one_handed'
rows=[]
for source,(lm,raw) in SOURCES.items():
    for gloss in ('GOOD','THANKYOU'):
        for path in sorted(glob.glob(str(lm/gloss/'*.npz'))):
            assert '/test/' not in path
            r=dict(source=source,gloss=gloss,archive=str(Path(path).relative_to(ROOT)),**features(path))
            stem=Path(path).name.removesuffix('.v17.npz')
            vids=sorted(glob.glob(str(raw/gloss/(stem+'.*'))))
            r['raw']=str(Path(vids[0]).relative_to(ROOT)) if vids else ''
            r['screen']=screen(r);rows.append(r)
with (OUT/'screen.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for r in rows:
    summary.setdefault(f"{r['source']}:{r['gloss']}",{}).setdefault(r['screen'],0)
    summary[f"{r['source']}:{r['gloss']}"][r['screen']]+=1
(OUT/'screen_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for k,v in summary.items():print(f'{k:28s}',v)
