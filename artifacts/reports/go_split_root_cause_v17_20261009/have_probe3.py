"""Session 201823 (user: I HAVE BIG throughout): installed word decoder (fragment recognizer + dip bridging +
HAVE/ANGRY check). For each committed I, the gap up to the next BIG/LANGUAGE: best recognizer reads of spans in
the gap and the two handshape cues."""
import json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_variants as ev
from active.v17.segmental_runtime_v17 import CropPair,HandshapePair
from active.v17.stage1_window_v17 import raw_observation_features
G='artifacts/reports/go_split_root_cause_v17_20261009/'
v=dict(recognizer=G+'run_frag_0.3/span_recognizer/best_model.pth',letter_head=G+'deploy_frag/letter_head_a/model.pth',decoder={'inner_floor':0.5},
       stream={'max_inner_o':8,'lag':6,'crop_pair':'artifacts/models/crop_have_angry_v17/model.json','handshape_pair':'artifacts/models/handshape_have_angry_v17/model.json'})
rt=ev.runtime(v);rec=rt.recognizer;L=rec.labels
cp=CropPair(ev.ROOT/'artifacts/models/crop_have_angry_v17/model.json');hp=HandshapePair(ev.ROOT/'artifacts/models/handshape_have_angry_v17/model.json')
item=pickle.load(open('/Users/frnzlo/slt_go_split_cache/takes3/20261009_201823_01.pkl','rb'));ev.restore_frames(item)
cache={};words=ev.run_video(rt,item,cache);obs,hands=item['obs'],item['hands'];sec=np.array([o.seconds for o in obs])
print('MAC:',' '.join(w['g'] for w in words))
json.dump(words,open('have_probe3_words.json','w'))
H,A=L.index('HAVE'),L.index('ANGRY')
for i,w in enumerate(words):
    if w['g']!='I':continue
    nxt=next((x for x in words[i+1:] if x['g'] in ('BIG','LANGUAGE')),None)
    if nxt is None or nxt['s']-w['e']>40:continue
    a0,b0=w['e']+1,nxt['s']-1
    between=[x['g'] for x in words[i+1:] if x['s']<nxt['s']]
    best=None
    for a in range(a0-2,b0):
        for b in range(a+3,min(b0+4,a+16)):
            ids=[k for k in range(len(obs)) if sec[a]-.1<=sec[k]<=sec[b]+.1];x=rec.inputs([obs[k] for k in ids],[hands[k] for k in ids])
            if x is None:continue
            l=rec.logits([x])[0][:100];p=np.exp(l-l.max());p/=p.sum()
            if best is None or p[H]>best[0]:
                raw,_=raw_observation_features([obs[k] for k in ids])
                best=(p[H],sec[a],sec[b],L[int(p.argmax())],p.max(),p[A],cp.score(x[1],x[2]),hp.score(raw))
    if best:print(f"I@{sec[w['s']]:6.2f} committed between: {between or '-'}; best HAVE span {best[1]:.2f}-{best[2]:.2f} P(HAVE) {best[0]:.2f} (top {best[3]} {best[4]:.2f}, ANGRY {best[5]:.2f}) crop {best[6]:.2f} shape {best[7]:.2f}")
