"""For the recognizer's most frequent confusions on validation (full model, Citizen/SemLex/local val): can a
pairwise hand-crop classifier (like HAVE/ANGRY) separate the pair? Train on train caches, report val AUC and
how many of the model's X->Y errors it would flag (P(X) >= 0.5 on the misread clip)."""
import json,collections,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
L=json.load(open('/Volumes/secret/SLT/SLT/artifacts/coreml/live_reel_labels.json'))
C='/Volumes/secret/SLT/SLT/artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant/'
d=np.load('sweep/feature_sweep.npz');T=d['targets'];fu=d['full'].argmax(1)
def feat(he,hv):
    he=he.astype(np.float32);hv=hv.astype(np.float32)[...,None]
    return np.concatenate([(he[:,:2]*hv[:,:2]).sum((0,1))/max(hv[:,:2].sum(),1),(he[:,2]*hv[:,2]).sum(0)/max(hv[:,2].sum(),1)])
def load(split):
    F,Y=[],[]
    for s in ('citizen','semlex','local'):
        z=np.load(C+f'{s}_{split}.npz',mmap_mode='r');F.append(np.stack([feat(z['hand_embeddings'][i],z['hand_valid'][i]) for i in range(len(z['targets']))]));Y.append(np.asarray(z['targets']))
    return np.concatenate(F),np.concatenate(Y)
Ft,Yt=load('train');Fv,Yv=load('val');assert (Yv==T).all()
errs=collections.Counter((int(t),int(p)) for t,p in zip(T,fu) if t!=p)
pairs={}
for (a,b),c in errs.items():k=tuple(sorted((a,b)));pairs[k]=pairs.get(k,0)+c
rows=[]
for (a,b),c in sorted(pairs.items(),key=lambda x:-x[1])[:30]:
    mt=np.isin(Yt,[a,b]);mv=np.isin(Yv,[a,b])
    sc=StandardScaler().fit(Ft[mt]);clf=LogisticRegression(C=0.05,max_iter=3000,class_weight='balanced').fit(sc.transform(Ft[mt]),Yt[mt]==a)
    p=clf.predict_proba(sc.transform(Fv[mv]))[:,1];auc=roc_auc_score(Yv[mv]==a,p)
    pv=np.full(len(Yv),np.nan);pv[mv]=p
    fixed=sum(1 for i in np.flatnonzero((T==a)&(fu==b)) if pv[i]>=.5)+sum(1 for i in np.flatnonzero((T==b)&(fu==a)) if pv[i]<.5)
    broken=sum(1 for i in np.flatnonzero((T==a)&(fu==a)) if pv[i]<.5)+sum(1 for i in np.flatnonzero((T==b)&(fu==b)) if pv[i]>=.5)
    r=dict(a=L[a],b=L[b],errors=c,ab=errs[(a,b)],ba=errs[(b,a)],train=(int((Yt==a).sum()),int((Yt==b).sum())),crop_auc=round(float(auc),3),would_fix=fixed,would_break=broken)
    rows.append(r);print(f"{L[a]:10s} <-> {L[b]:10s} errors {c:2d} ({errs[(a,b)]}/{errs[(b,a)]})  crop AUC {auc:.3f}  pair check would fix {fixed}, break {broken}")
json.dump(rows,open('sweep/pair_separability.json','w'),indent=1)
