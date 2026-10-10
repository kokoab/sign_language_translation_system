from pathlib import Path
import json,hashlib,collections
O=Path(__file__).resolve().parent
summary={'scope':'Mac native FP16 video replay and FP32 checkpoint diagnostics; development evidence; no training or installation','matrix':{},'mode_ablations':{},'source_checks':[]}
for key in ['historical_old','historical_new','current_old','current_new','no_name','uniform_word']:
 d=json.load(open(O/(key+'.json')));rows=d['records'];expected=8 if key in ['no_name','uniform_word'] else 17
 assert len(rows)==expected
 old=[r for r in rows if r['id'].startswith('sept30')];new=[r for r in rows if r['id'].startswith('oct11')]
 q={'replays':len(rows),'sept30_exact':sum(r['hypothesis']==r['reference'] for r in old),'sept30_n':len(old),'my_contact_regions':[]}
 for lo,hi in [(2.7,3.15),(5.4,7.15),(8.9,9.55)]:
  outputs=[[{'gloss':w['gloss'],'start_seconds':w['start_seconds'],'end_seconds':w['end_seconds']} for w in r['words'] if lo<=w['start_seconds']<hi and w['gloss'] in ['MY','LIKE','PLEASE']] for r in new]
  q['my_contact_regions'].append({'start_range':[lo,hi],'phase_outputs':outputs,'MY':sum(any(w['gloss']=='MY' for w in a) for a in outputs),'LIKE':sum(any(w['gloss']=='LIKE' for w in a) for a in outputs)})
 for r in rows:
  if r['id']=='have_take11':q['have_take11_counts']=dict(collections.Counter(r['hypothesis']));q['have_take11_hypothesis']=r['hypothesis']
 summary['mode_ablations' if expected==8 else 'matrix'][key]=q
for row in json.load(open(O/'source_manifest.json')):
 p=Path(row['source']);actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==row['source_sha256'],p
 summary['source_checks'].append({'path':str(p),'sha256':actual,'unchanged':True})
assert sum(v['replays'] for group in ['matrix','mode_ablations'] for v in summary[group].values())==84
for v in summary['matrix'].values():assert v['sept30_exact']==v['sept30_n']==8
windows=json.load(open(O/'window_results.json'));assert all(v['n']==62 for v in windows.values())
summary['window_diagnostic']={k:{'MY_top':v['my_top'],'n':v['n']} for k,v in windows.items()}
classes=json.load(open(O/'class_retention.json'));assert len(classes)==100
summary['reused_isolated_validation']={'n':sum(r['n'] for r in classes),'classes_improved':sum(r['current_minus_old']>0 for r in classes),'classes_worse':sum(r['current_minus_old']<0 for r in classes),'classes_equal':sum(r['current_minus_old']==0 for r in classes)}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('verified 84 replays, 4 x 62 window scores, 100-class retention, source hashes unchanged')
