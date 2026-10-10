from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
arms=['current','rollback_none','rollback_geometry','rollback_roll','rollback_both','ordered','ordered_ends','fragment_rollback_both','fragment_ordered','local_a_rollback_both','local_a_ordered']
cases=['first_miss','long_word_run','repeat_opening','take11','take0955','live1006']
out={}
for arm in arms:
 r={}
 for case in cases:
  f=P/f'{arm}_{case}.json'
  if not f.exists():continue
  d=json.loads(f.read_text());h=d['hypothesis'];w=d['words']
  r[case]=dict(hypothesis=h,have=h.count('HAVE'),my=h.count('MY'),like=h.count('LIKE'),angry=h.count('ANGRY'),tired=h.count('TIRED'),backwards=sum(b['start_seconds']<a['start_seconds'] for a,b in zip(w,w[1:])))
 f=P/f'{arm}_videos.json'
 if f.exists():
  d=json.loads(f.read_text());r['videos']=[]
  for v in d['records']:
   contact=[w for w in v['words'] if 9.05<w['start_seconds']<9.4]
   r['videos'].append(dict(id=v['id'],hypothesis=v['hypothesis'],first_mode=v['spell_events'][:2],contact9=contact,backwards=sum(b['start_seconds']<a['start_seconds'] for a,b in zip(v['words'],v['words'][1:]))))
 out[arm]=r
(P/'summary.json').write_text(json.dumps(out,indent=2))
for arm,r in out.items():
 print(arm,'HAVE',[r.get(k,{}).get('have') for k in cases[-3:]],'first',r.get('first_miss',{}).get('hypothesis'),'9s',[w['gloss'] for w in r.get('videos',[{}])[0].get('contact9',[])])
manifest=json.loads((P/'source_manifest.json').read_text());assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest.items())
print('Production source hashes unchanged.')
