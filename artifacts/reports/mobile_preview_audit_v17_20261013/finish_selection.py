"""Choose only four-for-four preview/source alternatives for the eight remaining failures."""
import json,collections,hashlib,shutil
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2]
choices={};groups=collections.defaultdict(list)
for stem in ['last']:
 choices.update(json.loads((O/(stem+'_choices.json')).read_text()))
 for row in json.loads((O/(stem+'_replay.json')).read_text())['records']:groups[row['id'].rsplit('__',2)[0]].append(row)
passed={i:rs for i,rs in groups.items() if len(rs)==4 and all(r['reference']==r['hypothesis'] for r in rs)}
cat=json.loads((O/'stable_preview/catalog.json').read_text());selected={};out=O/'delivery_preview';out.mkdir(exist_ok=True)
for label in ['COME','GIVE','WOMAN']:
 eligible=[i for i in passed if choices[i]['row']['canonical_label']==label];assert eligible,label
 ident=max(eligible,key=lambda i:min(float(r['words'][0]['score']) for r in passed[i]));selected[label]=dict(id=ident,min_score=min(r['words'][0]['score'] for r in passed[ident]));cat['examples'][label]=choices[ident]['row']
for label,row in cat['examples'].items():
 folder=Path(choices[selected[label]['id']]['directory']) if label in selected else O/'stable_preview'
 for name in [row['assets']['poster'],row['assets']['loop']]:shutil.copy2(folder/name,out/name)
 row['source_sha256']=hashlib.sha256((R/row['source_video']).read_bytes()).hexdigest()
counts=collections.Counter(row['source'] for row in cat['examples'].values());cat['classes_from_local']=counts['local'];cat['classes_from_citizen']=counts['citizen'];cat['classes_with_edge_notes']=[k for k,v in cat['examples'].items() if v['notes']]
(out/'catalog.json').write_text(json.dumps(cat,indent=1)+'\n');(O/'last_selected.json').write_text(json.dumps(selected,indent=2)+'\n')
for name,folder in [('latest',out),('latest_source',R)]:
 rows=[dict(id=k,path=str(folder/v['assets']['loop']) if name=='latest' else str(R/v['source_video']),reference=[k]) for k,v in cat['examples'].items()]
 (O/(name+'_videos.json')).write_text(json.dumps(rows,indent=1)+'\n')
print('counts',dict(counts),'selected',selected)
