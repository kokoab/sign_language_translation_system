"""Render admitted train alternatives for the audited reference failures only."""
import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from scripts import build_gloss_examples_v17 as b
O=Path(__file__).resolve().parent
original=json.loads((O/'preview_replay.json').read_text())['records']
corrected=json.loads((O/'corrected_replay.json').read_text())['records']
labels={v['id'] for v in original+corrected if v['reference']!=v['hypothesis']}
catalog=json.loads((O/'corrected_preview/catalog.json').read_text());local=b.local_candidates();skip=b.rejected_videos();prov=b.provenance_rows();rows=[];choices={}
for label in sorted(labels):
 candidates=local.get(label,[])[:3]+sorted(b.collect(label,skip,prov),key=lambda c:c.score,reverse=True)[:4]
 chosen=catalog['examples'][label]['source_video']
 for n,c in enumerate(candidates):
  if str(c.video.relative_to(R))==chosen:continue
  ident=f'{label}__{c.source}__{n}';out=O/'alternatives'/ident
  assets=b.render(c,out,480,.25)
  row=dict(catalog['examples'][label],source=c.source,source_video=str(c.video.relative_to(R)),source_sha256=c.sha256 or hashlib.sha256(c.video.read_bytes()).hexdigest(),split='train',seconds=round(c.seconds,3),score=round(c.score,4),components=c.components,notes=c.notes,participant=c.participant,assets=assets)
  choices[ident]={'row':row,'directory':str(out)};rows.append(dict(id=ident,path=str(out/assets['loop']),reference=[label]))
(O/'alternatives.json').write_text(json.dumps(choices,indent=1)+'\n');(O/'alternative_videos.json').write_text(json.dumps(rows,indent=1)+'\n');print('labels',len(labels),'alternatives',len(rows))
