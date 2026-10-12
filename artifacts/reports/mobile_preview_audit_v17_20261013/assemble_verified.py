"""Assemble measured reference choices; leaves model/runtime and datasets untouched."""
import json,shutil,collections,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2]
cat=json.loads((O/'corrected_preview/catalog.json').read_text());choices=json.loads((O/'alternatives.json').read_text());records=json.loads((O/'alternative_replay.json').read_text())['records'];passed={r['id'] for r in records if r['reference']==r['hypothesis']}
by=collections.defaultdict(list)
for ident,choice in choices.items():by[choice['row']['canonical_label']].append((ident,choice))
missing=[k for k,v in by.items() if not any(i in passed for i,_ in v)]
assert not missing,missing
out=O/'verified_preview';out.mkdir(exist_ok=True);selected={}
for label,row in cat['examples'].items():
 folder=O/'corrected_preview'
 if label in by:
  ident,choice=next((i,v) for i,v in by[label] if i in passed);row=choice['row'];folder=Path(choice['directory']);cat['examples'][label]=row;selected[label]=ident
 for name in [row['assets']['poster'],row['assets']['loop']]:shutil.copy2(folder/name,out/name)
 row['source_sha256']=hashlib.sha256((R/row['source_video']).read_bytes()).hexdigest()
counts=collections.Counter(row['source'] for row in cat['examples'].values());cat['classes_from_local']=counts['local'];cat['classes_from_citizen']=counts['citizen'];cat['classes_with_edge_notes']=[k for k,v in cat['examples'].items() if v['notes']]
cat['selection']='quality-ranked eligible training sources, with native recognition-checked alternatives; final replay required'
cat['local_training_manifest_sha256']=hashlib.sha256((R/cat['local_training_manifest']).read_bytes()).hexdigest()
(out/'catalog.json').write_text(json.dumps(cat,indent=1)+'\n');(O/'selected_alternatives.json').write_text(json.dumps(selected,indent=2)+'\n');(O/'final_videos.json').write_text(json.dumps([dict(id=k,path=str(out/v['assets']['loop']),reference=[k]) for k,v in cat['examples'].items()],indent=1)+'\n');print('selected',len(selected),'counts',dict(counts))
