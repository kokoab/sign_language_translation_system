"""Movement audit of SIGN/WHEN and BIG/LANGUAGE across every training source (v4 local pool).

two_hand_ratio: path length of the less-moving wrist / more-moving wrist (SIGN ~ both hands circle,
WHEN ~ one static base hand). reversals: direction changes of inter-wrist distance (LANGUAGE repeats,
BIG is one spread). Writes pair_audit.json with per-clip values and raw video paths.
"""
import glob,json
from pathlib import Path
import numpy as np
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent
def load(f):
    with np.load(f,allow_pickle=False) as p:return p['features'].astype(np.float32)
def two_hand_ratio(a):
    L=a[:,0,:2];R=a[:,21,:2];ok=(a[:,0,3]>0.5)&(a[:,21,3]>0.5)
    if ok.sum()<6:return None
    l=np.linalg.norm(np.diff(L[ok],axis=0),axis=1).sum();r=np.linalg.norm(np.diff(R[ok],axis=0),axis=1).sum()
    return float(min(l,r)/max(l,r,1e-6))
def reversals(a):
    ok=(a[:,0,3]>0.5)&(a[:,21,3]>0.5)
    if ok.sum()<8:return None
    d=np.linalg.norm(a[ok,0,:2]-a[ok,21,:2],axis=1);d=np.convolve(d,np.ones(3)/3,mode='same');amp=d.max()-d.min()
    if amp<1e-3:return 0
    n=0;last=d[0];trend=0
    for x in d[1:]:
        if trend>=0 and x<last-0.15*amp:n+=trend>0;trend=-1;last=x
        elif trend<=0 and x>last+0.15*amp:n+=trend<0;trend=1;last=x
        elif (trend>=0 and x>last) or (trend<=0 and x<last):last=x
    return n
rows=[]
CLASSES=('SIGN','WHEN','BIG','LANGUAGE')
for c in CLASSES:
    for f in sorted(glob.glob(str(ROOT/f'data/local/citizen100_v17/landmarks/train/{c}/*.npz'))):
        rows.append(dict(source='citizen',label=c,feature=f,raw=next(iter(glob.glob(f.replace('/landmarks/','/raw/').replace('.v17.npz','.*'))),'')))
sem=json.loads((ROOT/'data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json').read_text())
for v in sem['videos']:
    if v['canonical_label'] in CLASSES:rows.append(dict(source='semlex',label=v['canonical_label'],feature=str(ROOT/v['feature_path']),raw=str(ROOT/v['raw_path'])))
loc=json.loads((ROOT/'data/local/web_variant_supplement_v17_20261008/local_train_citizen_variant_v4.json').read_text())
for v in loc['videos']:
    if v['canonical_label'] in CLASSES:
        tier=v['consensus_tier'];src={'owner_approved_v16_deep_clean':'local','visual_review_web_citizen_variant_20261008':'web','asllex_citation_20261009':'asllex'}[tier]
        rows.append(dict(source=src,label=v['canonical_label'],feature=str(ROOT/v['feature_path']),raw=str(ROOT/v['raw_path'])))
for r in rows:
    a=load(r['feature']);r['two_hand_ratio']=two_hand_ratio(a);r['reversals']=reversals(a)
(OUT/'pair_audit.json').write_text(json.dumps(rows,indent=1)+'\n')
import collections
for c in CLASSES:
    for s in ('citizen','semlex','local','web','asllex'):
        x=[r for r in rows if r['label']==c and r['source']==s]
        if not x:continue
        t=[r['two_hand_ratio'] for r in x if r['two_hand_ratio'] is not None];v=[r['reversals'] for r in x if r['reversals'] is not None]
        print(f"{c:9s} {s:8s} n={len(x):3d} two-hand ratio median {np.median(t) if t else float('nan'):.2f} (>0.6: {sum(q>0.6 for q in t)}) | repeated(2+) {sum(q>=2 for q in v)}/{len(v)}")
