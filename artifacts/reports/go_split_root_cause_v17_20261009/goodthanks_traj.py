"""One-handed GOOD vs THANKYOU from the dominant hand's path and final shape (per-frame raw landmarks)."""
import glob,json,numpy as np,sys
sys.path.insert(0,'.');from handshape_probe import canon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def feat(raw):
    raw=np.asarray(raw,np.float32);ok=lambda f,st:(f[st:st+21,4]>=.3).sum()>=15
    best=None
    for st,left in ((0,True),(21,False)):
        idx=[i for i,f in enumerate(raw) if ok(f,st)]
        if len(idx)<4:continue
        w=np.array([raw[i,st,:2] for i in idx]);path=np.linalg.norm(np.diff(w,axis=0),axis=1).sum()
        if best is None or path>best[0]:best=(path,st,left,idx,w)
    if best is None:return None
    path,st,left,idx,w=best
    size=np.median([np.linalg.norm(raw[i,st+9,:2]-raw[i,st,:2]) for i in idx])+1e-6
    face=[raw[i,42:57,:2][raw[i,42:57,4]>0].mean(0) for i in idx if (raw[i,42:57,4]>0).any()]
    fy=np.mean(face,0) if face else w[0]
    d=(w[-1]-w[0])/size;top=(w[:,1].min()-fy[1])/size;low=(w[:,1].max()-fy[1])/size
    c0=canon(raw[idx[0],st:st+21],left);c1=canon(raw[idx[-1],st:st+21],left)
    if c0 is None or c1 is None:return None
    tip_end=c1.reshape(20,2)[[3,7,11,15,19]].mean(0)    # fingertips in canonical frame at the end
    return np.concatenate([[d[0]*(-1 if left else 1),d[1],path/size,top,low,len(idx)/20.],tip_end,c1])
def load(split_glob,c):
    X=[]
    for f in sorted(glob.glob(split_glob.format(c=c))):
        v=feat(np.load(f)['raw'])
        if v is not None:X.append(v)
    return np.array(X)
G,T='GOOD','THANKYOU'
Xg=load('/Users/frnzlo/slt_go_split_cache/isolated_raw/*/{c}/*.npz',G);Xt=load('/Users/frnzlo/slt_go_split_cache/isolated_raw/*/{c}/*.npz',T)
X=np.vstack([Xg,Xt]);y=np.r_[np.ones(len(Xg)),np.zeros(len(Xt))]
sc=StandardScaler().fit(X);clf=LogisticRegression(C=0.1,max_iter=3000,class_weight='balanced').fit(sc.transform(X),y)
for c in (G,T):
    V=load('/Users/frnzlo/slt_go_split_cache/isolated_val_raw/*/{c}/*.npz',c);p=clf.predict_proba(sc.transform(V))[:,1]
    print(c,'val n',len(V),'P(GOOD) >0.5:',int((p>.5).sum()),' >0.8:',int((p>.8).sum()),' <0.2:',int((p<.2).sum()))
import pickle;pickle.dump(dict(scaler=sc,clf=clf),open('goodthanks_traj.pkl','wb'))
