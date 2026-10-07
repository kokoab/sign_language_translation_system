"""Split local I clips into ME (index extended, pinky folded) vs fingerspelled letter I (pinky up)."""
import json,csv,numpy as np
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent
rows=[]
for split,f in (('train','train_final_manifest.json'),('val','val_final_manifest.json')):
    m=json.loads((ROOT/'data/local/local_deep_clean_v17'/f).read_text())
    for v in m['videos']:
        if v['canonical_label']!='I':continue
        with np.load(ROOT/v['feature_path'],allow_pickle=False) as p:a=p['features'].astype(np.float32)
        scores=[]
        for t in range(len(a)):
            best=None
            for s in (0,21):
                if a[t,s,3]<=0.5:continue
                xy=a[t,s:s+21,:2];w=xy[0]
                r=lambda tip,mcp:np.linalg.norm(xy[tip]-w)/max(np.linalg.norm(xy[mcp]-w),1e-4)
                sc=r(20,17)-r(8,5)
                best=sc if best is None or abs(sc)>abs(best) else best
            if best is not None:scores.append(best)
        med=float(np.median(scores)) if scores else float('nan')
        rows.append(dict(split=split,item_id=v['item_id'],feature_path=v['feature_path'],raw_path=v['raw_path'],frames=len(scores),pinky_minus_index=med,
                         call='letter_I' if med>0.25 else 'ME' if med<-0.25 else 'unclear'))
with (OUT/'i_clip_calls.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
import collections;print(collections.Counter((r['split'],r['call']) for r in rows))
