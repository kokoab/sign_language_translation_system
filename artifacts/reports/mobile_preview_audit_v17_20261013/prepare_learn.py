import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from scripts import build_gloss_examples_v17 as b
O=Path(__file__).resolve().parent;cat=json.loads((O/'verified_preview/catalog.json').read_text());choices={};videos=[]
for n,c in enumerate(sorted(b.collect('LEARN',b.rejected_videos(),b.provenance_rows()),key=lambda c:c.score,reverse=True)[3:9],3):
 ident=f'LEARN__citizen_extra__{n}';out=O/'repeat_candidates'/ident;a=b.render(c,out,480,.25);row=dict(cat['examples']['LEARN'],source=c.source,source_video=str(c.video.relative_to(R)),source_sha256=c.sha256,split='train',seconds=round(c.seconds,3),score=round(c.score,4),components=c.components,notes=c.notes,participant=c.participant,assets=a);choices[ident]=dict(row=row,directory=str(out))
 for i in range(2):
  for kind,p in [('preview',out/a['loop']),('source',c.video)]:videos.append(dict(id=f'{ident}__{kind}__{i}',path=str(p),reference=['LEARN']))
(O/'learn_choices.json').write_text(json.dumps(choices,indent=1)+'\n');(O/'learn_videos.json').write_text(json.dumps(videos,indent=1)+'\n');print(len(videos))
