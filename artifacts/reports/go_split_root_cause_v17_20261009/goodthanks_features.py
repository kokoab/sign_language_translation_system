"""GOOD (chin down onto the other palm, two hands) vs THANKYOU (chin outward, one hand): ending features from
per-frame raw landmarks. features(raw) -> [both_end, contact_end, other_still] or None."""
import numpy as np
def hand_ok(f,st):return (f[st:st+21,4]>=.3).sum()>=15
def features(raw):
    raw=np.asarray(raw,np.float32)
    act=[i for i,f in enumerate(raw) if hand_ok(f,0) or hand_ok(f,21)]
    if len(act)<4:return None
    raw=raw[act[0]:act[-1]+1];n=len(raw)
    def path(st):
        pts=[f[st,:2] for f in raw if hand_ok(f,st)]
        return float(np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1).sum()) if len(pts)>1 else 0.
    pl,pr=path(0),path(21);dom,non=(0,21) if pl>=pr else (21,0)
    end=raw[n//2:]
    both=np.mean([hand_ok(f,0) and hand_ok(f,21) for f in end])
    d=[]
    for f in end:
        if hand_ok(f,dom) and hand_ok(f,non):
            size=np.linalg.norm(f[dom+9,:2]-f[dom,:2])+1e-6
            tips=f[[dom+8,dom+12],:2].mean(0);palm=f[[non+0,non+5,non+9,non+13,non+17],:2].mean(0)
            d.append(np.linalg.norm(tips-palm)/size)
    contact=float(min(d)) if d else 5.0
    still=min(pl,pr)/max(pl,pr,1e-6)
    return np.array([both,min(contact,5.0),still],np.float64)
