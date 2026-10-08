"""Take 135024_11 through the installed word decoder: committed words with recognizer top-3, and the
best-scoring spans around each 'I <X> BIG' to see what HAVE spans are read as."""
import json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_variants as ev
v={'stream':{'max_inner_o':8,'lag':6},'decoder':{'inner_floor':0.5}}
rt=ev.runtime(v);labels=rt.labels
item=pickle.load(open('/Users/frnzlo/slt_go_split_cache/takes2/20261009_135024_11.pkl','rb'));ev.restore_frames(item)
cache={};words=ev.run_video(rt,item,cache);sec=[o.seconds for o in item['obs']]
def top(k,n=4):
    l=cache.get(k)
    if l is None:return '-'
    w=np.asarray(l[:100],np.float64);p=np.exp(w-w.max());p/=p.sum();o=np.argsort(-p)[:n]
    return ' '.join(f'{labels[i]}:{p[i]:.2f}' for i in o)
for w in words:print(f"{w['g']:8s} {sec[w['s']]:5.2f}-{sec[w['e']]:5.2f} q{w['q']:.2f} [{top((w['s'],w['e']))}]")
print('--- spans inside 1.15-1.65 s (the HAVE of the first I HAVE BIG):')
for k in sorted(cache):
    s,e=k
    if 1.1<=sec[s]<=1.35 and 1.35<=sec[e]<=1.75:print(f'  {sec[s]:.2f}-{sec[e]:.2f}',top(k))
json.dump(words,open('have_probe_words.json','w'))
