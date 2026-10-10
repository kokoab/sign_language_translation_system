"""Matched letter head for the existing clean-split recognizer; no recognizer retraining."""
from pathlib import Path
import json,sys,subprocess,hashlib,os
R=Path('/Volumes/secret/SLT/SLT');O=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
REC=R/'artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe/august/span_recognizer/best_model.pth'
RECIPE=R/'active/v17/phrase_segment_recipe_manifest_20261007.json'
ISO=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
HEAD=O/'letter_head/model.pth';PKG=O/'SpanRecognizerV17CleanRecoveryLettersB8FP16.mlpackage'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def train():
 import scripts.train_letter_head_v17 as m
 recipe=json.load(open(RECIPE));assert recipe['recipe_training_ready']
 allowed={e['video_sha256'] for e in recipe['entries'] if e['split']=='train'}
 original=m.lab.rows_for
 rows=original('local_train');admitted=[r for r in rows if r['video_sha256'] in allowed]
 assert len(admitted)==179
 held={r['video_sha256'] for name in ['tune','test'] for r in original(name)}
 assert not allowed & held
 # Positive A-Z examples keep their established split; NONE phrase negatives are
 # limited to approved local training videos. No ASLLRP/OTHER default admission.
 def scoped(name):
  if name=='local_train':return admitted
  if name=='asllrp_train':return []
  assert name=='tune',name
  return original(name)
 m.lab.rows_for=scoped;m.ISO=ISO
 inputs=[REC,RECIPE,m.CORRECTIONS]
 inputs += sorted(m.LETTER_ROOT.glob('train/*/*.npz'))+sorted(m.LETTER_ROOT.glob('validation/*/*.npz'))
 inputs += [ISO/f'{s}_{split}.npz' for s in ['citizen','semlex','local'] for split in ['train','val']]
 for row in admitted:
  inputs += [m.lab.CACHE/'train_labels'/(m.lab.key(row['source_item_id'])+'.json'),m.lab.memo_path(row)]
 pins=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs if p.exists()]
 (O/'letter_recipe.json').write_text(json.dumps({'scope':'letter-head-only; frozen existing clean recognizer','phrase_recipe_sha256':sha(RECIPE),'recognizer_sha256':sha(REC),'approved_phrase_train_videos':sorted(allowed),'train_videos':len(admitted),'validation':'established A-Z validation, tune NONE spans, isolated validation only','corrections':str(m.CORRECTIONS),'isolated_cache':str(ISO),'test_accessed':False,'pins':pins},indent=2)+'\n')
 sys.argv=['train_letter_head','--recognizer',str(REC),'--output',str(HEAD),'--device','mps'];m.main()
if sys.argv[1:]==['letter']:
 train();raise SystemExit()
try:
 py=str(R/'venv/bin/python')
 with (O/'letter_head.log').open('w') as f:subprocess.run([py,__file__,'letter'],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (O/'export.log').open('w') as f:subprocess.run([py,'active/v17/export_span_recognizer_batched_coreml_v17.py',str(REC),str(PKG),'--precision','float16','--letter-head',str(HEAD),'--fixed-batch','8'],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 sys.path.insert(0,str(R/'artifacts/reports/canonical_recognition_comparison_v17_20261007'))
 import check_recognizer_exports as c
 import torch
 check=c.check(REC,PKG);head=torch.load(HEAD,map_location='cpu',weights_only=False)
 (O/'candidate_result.json').write_text(json.dumps({'status':'complete','parity':check,'letter_threshold':head['threshold'],'letter_selected':head['selected'],'recognizer_unchanged':sha(REC)==json.load(open(O/'letter_recipe.json'))['recognizer_sha256']},indent=2)+'\n')
 print('CANDIDATE_COMPLETE',flush=True)
except Exception as e:
 (O/'candidate_failure.json').write_text(json.dumps({'status':'failed','error':repr(e)}));raise
finally:
 subprocess.run(['osascript','-e','display notification "Clean lineage recovery build finished; inspect results." with title "SLT experiment"'],check=False)
