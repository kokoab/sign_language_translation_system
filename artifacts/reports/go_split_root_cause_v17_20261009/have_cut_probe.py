"""For each in-sentence HAVE of take 135024_11: recognizer P(HAVE)/P(ANGRY) for spans that end at different points
(contact only vs including the upward exit into BIG), with the runtime's +-0.1 s context and with none; plus BIO."""
import json,pickle,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_variants as ev
from active.v17.av_boundary_v17 import clip_bio
from active.v17.stage1_window_v17 import raw_observation_features
rec=ev.words.recognizer;L=rec.labels;H,A=L.index('HAVE'),L.index('ANGRY')
item=pickle.load(open('/Users/frnzlo/slt_go_split_cache/takes2/20261009_135024_11.pkl','rb'));ev.restore_frames(item)
obs,hands=item['obs'],item['hands'];sec=np.array([o.seconds for o in obs])
raw,t=raw_observation_features(obs);P=np.exp(clip_bio(ev.boundary(),raw,t))
words=json.load(open('have_probe_words.json'))
for w in [w for w in words if w['g']=='ANGRY']:
    s,e=w['s'],w['e'];print(f"\n== committed ANGRY {sec[s]:.2f}-{sec[e]:.2f}")
    lo,hi=s-4,e+6
    print('   t   :',' '.join(f'{sec[i]:5.2f}' for i in range(lo,hi)))
    print('   act :',' '.join(f'{P[i,2]+P[i,3]:5.2f}' for i in range(lo,hi)))
    print('   B   :',' '.join(f'{P[i,2]:5.2f}' for i in range(lo,hi)))
    for ctx in (0.1,0.0):
        for a in (s-2,s,s+1):
            row=[]
            for b in range(a+3,e+3):
                ids=[i for i in range(len(obs)) if sec[a]-ctx-1e-9<=sec[i]<=sec[b]+ctx+1e-9]
                x=rec.inputs([obs[i] for i in ids],[hands[i] for i in ids]) if len(ids)>=4 else None
                if x is None:row.append(f'{sec[b]:.2f}:  -  ');continue
                l=rec.logits([x])[0][:100];p=np.exp(l-l.max());p/=p.sum()
                row.append(f'{sec[b]:.2f}:H{p[H]:.2f}/A{p[A]:.2f}/{L[int(p.argmax())][:5]}')
            print(f'   ctx{ctx} start {sec[a]:.2f} ->',' '.join(row))
