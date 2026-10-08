"""User-domain HAVE rows from take 20261009_135024_11, FIRST HALF only (t < 16 s): the decoder spans of the
in-sentence HAVEs (committed as ANGRY) and the lone HAVEs, each with +-2 frame start/end jitter, labelled HAVE.
The second half (t >= 16 s) stays held out for evaluation."""
import json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_variants as ev
rec=ev.words.recognizer;labels=rec.labels
item=pickle.load(open('/Users/frnzlo/slt_go_split_cache/takes2/20261009_135024_11.pkl','rb'));ev.restore_frames(item)
obs,hands=item['obs'],item['hands'];sec=np.array([o.seconds for o in obs])
words=json.load(open('have_probe_words.json'))
spans=[(w['s'],w['e']) for w in words if w['g'] in ('ANGRY','HAVE','TIRED') and sec[w['e']]<16]
print('spans',[(round(sec[s],2),round(sec[e],2)) for s,e in spans])
rows={k:[] for k in ('landmarks','hand_embeddings','hand_valid','hand_boxes')}
for s,e in spans:
    for ds in (-2,-1,0,1,2):
        for de in (-2,-1,0,1,2):
            a,b=s+ds,e+de
            if b-a<3:continue
            ids=[i for i in range(len(obs)) if sec[a]-.1<=sec[i]<=sec[b]+.1]
            x=rec.inputs([obs[i] for i in ids],[hands[i] for i in ids])
            if x is None:continue
            for k,v in zip(rows,x):rows[k].append(v)
n=len(rows['landmarks']);out={k:np.stack(v) for k,v in rows.items()}
out['hand_embeddings']=out['hand_embeddings'].astype(np.float16);out['hand_valid']=out['hand_valid'].astype(bool)
out['targets']=np.full(n,labels.index('HAVE'),np.int64);out['item_ids']=np.asarray([f'user135024_11:HAVE:{i}' for i in range(n)])
np.savez('/Volumes/secret/SLT/SLT/artifacts/generated/user_have_firsthalf_v17_20261009.npz',**out);print('rows',n)
