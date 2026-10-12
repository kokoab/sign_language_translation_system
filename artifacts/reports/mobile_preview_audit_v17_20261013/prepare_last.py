"""Check candidate originals and previews twice under the unchanged native runtime."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from scripts import build_gloss_examples_v17 as b
O=Path(__file__).resolve().parent;cat=json.loads((O/'stable_preview/catalog.json').read_text());failed=set()
for f in ['stable_replay.json','stable_source_replay.json']:
 for row in json.loads((O/f).read_text())['records']:
  if row['reference']!=row['hypothesis']:failed.add(row['id'])
local=b.local_candidates();prov=b.provenance_rows();skip=b.rejected_videos();choices={};videos=[]
for label in sorted(failed):
 cs=local.get(label,[])[:2]+sorted(b.collect(label,skip,prov),key=lambda c:c.score,reverse=True)[:3]
 for n,c in enumerate(cs):
  ident=f'{label}__{c.source}__{n}';out=O/'last_candidates'/ident;assets=b.render(c,out,480,.25)
  row=dict(cat['examples'][label],source=c.source,source_video=str(c.video.relative_to(R)),source_sha256=c.sha256,split='train',seconds=round(c.seconds,3),score=round(c.score,4),components=c.components,notes=c.notes,participant=c.participant,assets=assets)
  choices[ident]=dict(row=row,directory=str(out))
  for repeat in range(2):
   for kind,path in [('preview',out/assets['loop']),('source',c.video)]:videos.append(dict(id=f'{ident}__{kind}__{repeat}',path=str(path),reference=[label]))
(O/'last_choices.json').write_text(json.dumps(choices,indent=1)+'\n');(O/'last_videos.json').write_text(json.dumps(videos,indent=1)+'\n');print('failed labels',sorted(failed),'candidate replays',len(videos))
