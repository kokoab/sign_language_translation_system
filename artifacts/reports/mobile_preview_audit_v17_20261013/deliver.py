"""Publish the checked preview assets to both app bundles; retain exact prior bytes."""
import json,hashlib,collections
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];stage=O/'delivery_preview';cat=json.loads((stage/'catalog.json').read_text())
checks={}
for name in ['delivered','latest_source']:
 records=json.loads((O/(name+'_replay.json')).read_text())['records'];assert len(records)==100 and {r['id'] for r in records}==set(cat['examples']);bad=[r for r in records if r['hypothesis']!=r['reference']];assert not bad,[(r['id'],r['hypothesis']) for r in bad];checks[name]=100
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path,digest in json.loads((O/'source_model_hashes.json').read_text()).items():assert sha(R/path)==digest,path
admitted=json.loads((R/cat['local_training_manifest']).read_text());allowed={(r['canonical_label'],r['raw_sha256']) for r in admitted['videos'] if r.get('training_eligible',False) and r.get('local_split')=='train'}
manifest={r['canonical_label']:r for r in json.loads((R/cat['manifest']).read_text())['classes']}
for label,row in cat['examples'].items():
 assert sha(R/row['source_video'])==row['source_sha256'];assert row['split']=='train'
 for key in ['class_index','citizen_raw_gloss','citizen_asl_lex_code']:assert row[key]==manifest[label][key]
 if row['source']=='local':assert (label,row['source_sha256']) in allowed
 else:assert '/raw/train/' in row['source_video']
 row['assets']['loop_sha256']=sha(stage/row['assets']['loop']);row['assets']['poster_sha256']=sha(stage/row['assets']['poster'])
cat['selection']='eligible train sources with native preview and source replay verification; reference selection only, not a generalization metric'
cat['recognition_check']={'report':'artifacts/reports/mobile_preview_audit_v17_20261013','platform':'macOS native Swift replay','compute_units':'all','recognizer':'SpanRecognizerV17ConfidenceDualB8FP16','image_encoder':'MobileCLIP2S0ImageEncoderV17FP16','exact_previews':100,'exact_sources':100,'total':100,'physical_phone_verified':False,'end_of_file_flush':True,'finish_gesture_enabled':False}
(stage/'catalog.json').write_text(json.dumps(cat,indent=1)+'\n')
destinations={'flutter':R/'ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples','native':R/'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/gloss_examples','desktop':R/'artifacts/app_assets/gloss_examples'};changed={}
for label,dst in destinations.items():
 dst.mkdir(parents=True,exist_ok=True);changed[label]=[]
 for src in stage.iterdir():
  if src.suffix not in ['.json','.jpg','.mp4']:continue
  target=dst/src.name;data=src.read_bytes()
  if target.exists() and target.read_bytes()==data:continue
  if target.exists():
   backup=O/'before'/label/src.name;backup.parent.mkdir(parents=True,exist_ok=True)
   if not backup.exists():backup.write_bytes(target.read_bytes())
  target.write_bytes(data);changed[label].append(src.name)
 for row in cat['examples'].values():
  for kind in ['poster','loop']:assert sha(dst/row['assets'][kind])==row['assets'][kind+'_sha256']
 assert (dst/'catalog.json').read_bytes()==(stage/'catalog.json').read_bytes()
(O/'delivery.json').write_text(json.dumps({'checks':checks,'source_counts':dict(collections.Counter(r['source'] for r in cat['examples'].values())),'changed_files':changed,'phone_installed':False,'source_model_hashes':'source_model_hashes.json'},indent=2)+'\n');print('delivered',checks,{k:len(v) for k,v in changed.items()})
