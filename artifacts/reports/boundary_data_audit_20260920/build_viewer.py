from pathlib import Path
import json,os,html,subprocess
import numpy as np
import cv2
from PIL import Image,ImageDraw
OUT=Path('artifacts/reports/boundary_data_audit_20260920').resolve(); ROOT=Path.cwd(); (OUT/'media').mkdir(exist_ok=True)
a=json.load(open(OUT/'audit.json')); windows=json.load(open(OUT/'window_details.json'))
manifest=json.load(open('artifacts/reports/o5s5_citizen100_v17/combined_supervision.json'))
rows={r['source_item_id']:r for r in manifest['rows']}
selected=[]
for x in [r for r in a['source_clipping_examples'] if r['reason']=='source_crop_clips_annotation'][:2]:
 selected.append(dict(item=x['item'],video=x['video'],title='Source crop cuts a neighbouring sign: '+x['original']['Entry/variant gloss label'],start=0.,end=min(3.,max(e['end_seconds'] for e in rows[x['item']]['intervals'])),note='Original annotation extends beyond the source crop. The fragment is mapped to OTHER. This does not prove the target gloss itself is truncated.'))
for source in ['asllrp_other_ctc','o5s5']:
 candidates=sorted([x for x in windows if x['role']=='train' and x['source']==source and x['duration']==.27],key=lambda x:x['target_fraction'])
 for x in [candidates[0],candidates[-1]]:
  selected.append(dict(item=x['item'],video=x['video'],title=f"{source}: {x['label']} / target observed-frame fraction {x['target_fraction']:.0%}",start=max(0,x['start']-.45),end=x['end']+.45,note=f"Annotation {x['start']:.3f}–{x['end']:.3f}s; model input {x['window_start']:.3f}–{x['window_end']:.3f}s. Endpoint label {x['label']}.",window=x))
canvas=Image.new('RGB',(1200,250*len(selected)),(245,245,245)); draw=ImageDraw.Draw(canvas); cards=[]
for n,x in enumerate(selected):
 clip=OUT/'media'/f'example_{n+1}.mp4'
 subprocess.run(['ffmpeg','-v','error','-y','-ss',str(x['start']),'-i',str(ROOT/x['video']),'-t',str(x['end']-x['start']),'-an','-vf','scale=640:-2','-c:v','libx264','-crf','24',str(clip)],check=True)
 cap=cv2.VideoCapture(str(ROOT/x['video']))
 draw.text((8,n*250+4),x['title'],fill='black')
 for j,t in enumerate(np.linspace(x['start'],x['end'],6,endpoint=False)):
  cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000); ok,f=cap.read()
  if not ok:continue
  im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)); im.thumbnail((195,180));canvas.paste(im,(j*200,n*250+30))
  draw.text((j*200+4,n*250+214),f'{t:.3f}s',fill='black')
 cap.release()
 es=[e for e in rows[x['item']]['intervals'] if e['start_seconds']<x['end'] and e['end_seconds']>x['start']]
 table=''.join(f"<tr><td>{e['start_seconds']:.3f}</td><td>{e['end_seconds']:.3f}</td><td>{html.escape(e['label'])}</td><td>{html.escape(e.get('id_gloss',''))}</td></tr>" for e in es)
 win=''
 if 'window' in x:
  w=x['window']; path=OUT/'media'/f'model_window_{n+1}.mp4'
  subprocess.run(['ffmpeg','-v','error','-y','-ss',str(w['window_start']),'-i',str(ROOT/x['video']),'-t',str(w['duration']),'-an','-vf','scale=640:-2','-c:v','libx264','-crf','24',str(path)],check=True)
  win=f'<p>Actual time interval used as model input (original RGB; model sees sampled landmarks):</p><video controls loop src="media/{path.name}"></video>'
 cards.append(f'<article><h2>{html.escape(x["title"])}</h2><p>{html.escape(x["note"])}</p><video controls loop src="media/{clip.name}"></video>{win}<p><a href="{html.escape(os.path.relpath(ROOT/x["video"],OUT))}">Full original source clip</a> · {html.escape(x["item"])}</p><table><tr><th>Start (s)</th><th>End (s)</th><th>Mapped label</th><th>Original gloss if available</th></tr>{table}</table></article>')
canvas.save(OUT/'contact_sheet.jpg')
(OUT/'examples.json').write_text(json.dumps(selected,indent=2)+'\n')
(OUT/'videos.html').write_text('<!doctype html><meta charset="utf-8"><title>Boundary and target audit</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#fafafa}article{background:white;padding:20px;margin:22px 0;border:1px solid #ccc}video{width:46%;max-height:360px}td,th{padding:5px 15px;text-align:left}h2{font-size:20px}</style><h1>Source clips, annotations, and model inputs</h1><p>Six deliberately selected diagnostic examples, not a representative accuracy sample. Times refer to original local source videos. Annotation truth is inherited from the dataset, not independently certified by an ASL annotator. All excerpts play at original speed.</p>'+''.join(cards))
print('Created six annotated examples, contact sheet, and videos.html')
