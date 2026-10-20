"""Publish checked preview assets to both mobile bundles, preserving prior bytes."""
from pathlib import Path
import hashlib,json,shutil,os
O=Path(__file__).resolve().parent;R=O.parents[2]
stage=O/'stage';cat=json.loads((stage/'catalog.json').read_text())
selection=json.loads((O/'selection.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
for row in json.loads((O/'inventory.json').read_text())['clips']:
 assert sha(Path(row['source']))==row['source_sha256']
for row in cat['examples'].values():
 for kind in ['loop','poster']:assert sha(stage/row['assets'][kind])==row['assets'][kind+'_sha256']
changed={}
for key,dst in {'flutter':R/'ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples',
                'native_ios':R/'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/gloss_examples'}.items():
 changed[key]=[]
 for source in sorted(stage.iterdir()):
  target=dst/source.name
  if target.exists() and sha(target)==sha(source):continue
  backup=O/'before'/key/source.name
  backup.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() and not backup.exists():shutil.copyfile(target,backup)
  temporary=target.with_name(target.name+'.preview_update')
  shutil.copyfile(source,temporary);os.replace(temporary,target)
  assert sha(target)==sha(source)
  changed[key].append(source.name)
 for label in selection['incumbent_fallbacks']:
  for suffix in ['mp4','jpg']:assert sha(dst/f'{label}.{suffix}')==sha(stage/f'{label}.{suffix}')
receipt={'new_previews':len(selection['selected']),'pending':selection['unresolved'],
 'incumbent_fallbacks':selection['incumbent_fallbacks'],'changed':changed,
 'original_source_files_unchanged':True,'training_admission':False,'installed_on_phone':False}
(O/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Published',len(selection['selected']),'owner previews to Flutter/Android and native iOS.',flush=True)
print('Pending labels:',selection['incumbent_fallbacks'],flush=True)
