"""Stage supplied clips; keep any unresolved class on its incumbent preview."""
from pathlib import Path
import json,hashlib,shutil
import cv2
O=Path(__file__).resolve().parent;R=O.parents[2]
OLD=R/'ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples'
rows=json.loads((O/'prepared.json').read_text())
pred={r['id']:r for r in json.loads((O/'unnamed_predictions.json').read_text())['records']}
original=json.loads((OLD/'catalog.json').read_text());known=set(original['examples'])
confirmed=json.loads((O/'confirmed_labels.json').read_text()) if (O/'confirmed_labels.json').exists() else {}
selected={};unresolved=[]
for row in rows:
 label=row['label'];method=row['label_source']
 if label is None:
  top=pred[row['id']]['primary_top5'];choice=top[0]
  if row['source_relative'] in confirmed:
   label=confirmed[row['source_relative']];method='user_confirmed'
  elif choice['probability']>=.6 and choice['label'] not in {r['label'] for r in rows if r['label']}:
   label=choice['label'];method='model_top1'
  else:
   unresolved.append({'source':row['source_relative'],'id':row['id'],'top5':top});continue
 assert label in known and label not in selected,(label,row['source_relative'])
 selected[label]={**row,'assigned_label':label,'label_source':method}
assert len(selected)+len(unresolved)==100
stage=O/'stage';stage.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
examples={}
for label,prior in original['examples'].items():
 video=stage/(label+'.mp4');poster=stage/(label+'.jpg')
 if label not in selected:
  for name in [video.name,poster.name]:shutil.copyfile(OLD/name,stage/name)
  examples[label]=prior;continue
 row=selected[label]
 shutil.copyfile(row['path'],video)
 cap=cv2.VideoCapture(str(video));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
 cap.set(cv2.CAP_PROP_POS_FRAMES,frames//2);ok,image=cap.read();cap.release();assert ok
 assert cv2.imwrite(str(poster),image,[cv2.IMWRITE_JPEG_QUALITY,88])
 metadata={k:prior[k] for k in ['canonical_label','class_index','category','citizen_raw_gloss','citizen_asl_lex_code']}
 metadata.update(source='owner_supplied_preview',source_video=row['source'],source_sha256=row['source_sha256'],
  split='preview_only',training_eligible=False,seconds=round(frames/fps,4),notes=[],
  preview_label_source=row['label_source'],assets={'loop':video.name,'poster':poster.name,'loop_sha256':sha(video),
   'poster_sha256':sha(poster),'frames':frames,'fps':round(fps,3),'width':image.shape[1],'height':image.shape[0],'codec':'h264'})
 if row['id'] in pred:metadata['preview_model_top5']=pred[row['id']]['primary_top5']
 examples[label]=metadata
cat={'format':original['format'],'version':original['version'],'manifest':original['manifest'],
 'selection':'Owner-supplied reference videos; named clips retain supplied labels, unnamed clips use model top1 or explicit user confirmation.',
 'source_split':'preview_only','training_eligible':False,'val_accessed':False,'test_accessed':False,
 'preview_source_root':'/Users/frnzlo/Downloads/signs','owner_preview_count':len(selected),
 'recognition_check':{'scope':'Label assistance for unnamed videos only; not a recognition-accuracy benchmark.',
    'recognizer':'SpanRecognizerV17ConfidenceDualB8FP16 retained primary','platform':'macOS native Swift replay',
    'unnamed_clips':34,'physical_phone_verified':False},'examples':examples}
assert set(examples)==known and len(examples)==100
(stage/'catalog.json').write_text(json.dumps(cat,indent=1)+'\n')
(O/'selection.json').write_text(json.dumps({'selected':selected,'unresolved':unresolved,'incumbent_fallbacks':sorted(known-set(selected))},indent=2)+'\n')
print('Staged',len(selected),'new previews; unresolved:',[(r['source'],r['top5'][0]['label']) for r in unresolved],flush=True)
