"""Local train/val manifests with clips whose form differs from the pinned Citizen variant removed.

Excluded (visual review, similar_sign_local_audit_v17_20261008): all local HOME (=HOUSE), CHILD,
GOODBYE, HEAR, WHAT, BIG, SIGN, ASK, COME; local I except clips classified as ME (index to chest).
User decision 2026-10-08: "for all confusions, use the Citizen one". Originals unchanged.
"""
import csv,json,hashlib,collections
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent
DROP={'HOME','CHILD','GOODBYE','HEAR','WHAT','BIG','SIGN','ASK','COME'}
keep_i={r['item_id'] for r in csv.DictReader((OUT/'i_clip_calls.csv').open()) if r['call']=='ME'}
summary={}
for split,f in (('train','train_final_manifest.json'),('val','val_final_manifest.json')):
    src=ROOT/'data/local/local_deep_clean_v17'/f;m=json.loads(src.read_text())
    removed=collections.Counter()
    kept=[]
    for v in m['videos']:
        c=v['canonical_label']
        if c in DROP or (c=='I' and v['item_id'] not in keep_i):removed[c]+=1
        else:kept.append(v)
    m['videos']=kept;m['selected_clips']=len(kept)
    if 'class_counts' in m:m['class_counts']=dict(collections.Counter(v['canonical_label'] for v in kept))
    if 'selected_classes' in m:m['selected_classes']=len({v['canonical_label'] for v in kept})
    m['derived_from']=dict(path=str(src.relative_to(ROOT)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        removed=dict(removed),rule='Citizen variant wins; see build_manifests.py docstring')
    out=OUT/f'local_{split}_citizen_variant.json';out.write_text(json.dumps(m)+'\n')
    summary[split]=dict(kept=len(kept),removed=dict(removed),sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(OUT/'manifest_summary.json').write_text(json.dumps(summary,indent=1)+'\n');print(json.dumps(summary,indent=1))
