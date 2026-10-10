"""Read preserved training membership and cached label metadata; no model inference."""
from pathlib import Path
import json,hashlib,collections
ROOT=Path(__file__).resolve().parents[3];N=Path(__file__).resolve().parent
rows=json.loads((ROOT/'data/local/combined_dataset_v17_20260922/manifest.json').read_text())['records']
recipe=json.loads((ROOT/'active/v17/phrase_segment_recipe_manifest_20261007.json').read_text())
paths={'old':'artifacts/models/span_recognizer_v17_local_a/history.json','new':'artifacts/reports/citizen_variant_local_filter_v17_20261008/run/span_recognizer/history.json'}
result={};members={};my_inputs={}
for name,path in paths.items():
 prov=json.loads((ROOT/path).read_text())['provenance'];admitted=set(prov['train_video_sha256']);members[name]=admitted;c=collections.Counter();videos=collections.defaultdict(set);seen=set();missing=[];my_inputs[name]=[]
 for row in rows:
  h=row['video_sha256']
  if row['source']!='local_phrases' or h not in admitted or h in seen:continue
  seen.add(h);key=hashlib.sha256(row['source_item_id'].encode()).hexdigest()[:20];p=ROOT/'artifacts/cache/segmental_decoder_v17/train_labels'/(key+'.json')
  if not p.exists():missing.append(row['source_item_id']);continue
  spans=[s for s in json.loads(p.read_text())['spans'] if not s.get('prefix')]
  for span in spans:c[span['gloss']]+=1;videos[span['gloss']].add(h)
  n=sum(s['gloss']=='MY' for s in spans)
  if n:my_inputs[name].append({'source_item_id':row['source_item_id'],'MY_spans':n,'video_sha256':h})
 assert sum(c.values())==prov['span_examples'] and not missing
 result[name]={'historical_total':prov['span_examples'],'metadata_count':sum(c.values()),'videos':len(seen),'missing':missing,'counts':dict(c),'videos_per_gloss':{g:len(v) for g,v in videos.items()},'freeze_encoders':prov['freeze_encoders']}
removed=[]
for r in my_inputs['old']:
 if r['video_sha256'] not in members['new']:
  matches=[{k:v for k,v in x.items() if k not in ['rgb','hand','rgb_sha256','hand_sha256']} for x in recipe['entries'] if x['video_sha256']==r['video_sha256']]
  removed.append(dict(r,recipe_records=matches))
assert len(removed)==8 and sum(r['MY_spans'] for r in removed)==84
assert all(len(r['recipe_records'])==1 and r['recipe_records'][0]['split']=='validation' for r in removed)
(N/'phrase_supervision_audit.json').write_text(json.dumps(result,indent=2)+'\n');(N/'removed_MY_phrase_inputs.json').write_text(json.dumps(removed,indent=2)+'\n')
print('Historical totals reproduced; eight MY NAME validation videos account for84removedMYspans.')
