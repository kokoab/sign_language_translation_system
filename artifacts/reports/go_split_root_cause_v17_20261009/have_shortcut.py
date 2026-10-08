"""Does the hand-image branch drive HAVE -> ANGRY? Re-score each committed ANGRY/HAVE span of take 135024_11
with (a) full inputs, (b) hand views masked (landmarks only), (c) landmarks masked (hands only)."""
import pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_variants as ev,json
rec=ev.words.recognizer;labels=rec.labels
item=pickle.load(open('/Users/frnzlo/slt_go_split_cache/takes2/20261009_135024_11.pkl','rb'));ev.restore_frames(item)
obs,hands=item['obs'],item['hands'];sec=np.array([o.seconds for o in obs])
words=json.load(open('have_probe_words.json'))
def top(l,n=3):
    w=np.asarray(l[:100],np.float64);p=np.exp(w-w.max());p/=p.sum();o=np.argsort(-p)[:n];return ' '.join(f'{labels[i]}:{p[i]:.2f}' for i in o)
for w in words:
    if w['g'] not in ('ANGRY','HAVE','TIRED'):continue
    lo,hi=sec[w['s']]-.1,sec[w['e']]+.1;ids=[i for i in range(len(obs)) if lo<=sec[i]<=hi]
    x=rec.inputs([obs[i] for i in ids],[hands[i] for i in ids])
    no_hand=(x[0],x[1]*0,np.zeros_like(x[2]),x[3]*0)
    no_lm=(np.zeros_like(x[0]),x[1],x[2],x[3])
    L=rec.logits([x,no_hand,no_lm])
    print(f"{w['g']:6s} {sec[w['s']]:5.2f}  full[{top(L[0])}]  landmarks-only[{top(L[1])}]  hands-only[{top(L[2])}]")
