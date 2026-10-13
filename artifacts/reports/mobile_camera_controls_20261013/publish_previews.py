from pathlib import Path
import hashlib,json,shutil
O=Path(__file__).resolve().parent;R=O.parents[2]
quality=json.loads((O/'media_quality.json').read_text());assert len(quality['clips'])==100 and quality['minimum']>.99
assert (O/'h264_native_decode_final.log').read_text().strip()=='Checked 100, failures 0: []'
stage=O/'h264_previews';original=O/'before/media'
old=json.loads((original/'catalog.json').read_text());new=json.loads((stage/'catalog.json').read_text())
for label,row in new['examples'].items():
 assert {k:v for k,v in row.items() if k!='assets'}=={k:v for k,v in old['examples'][label].items() if k!='assets'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dests=[R/'ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples',R/'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/gloss_examples',R/'artifacts/app_assets/gloss_examples']
files=sorted(stage.glob('*.mp4'))+[stage/'catalog.json']
for dst in dests:
 for src in files:assert sha(dst/src.name)==sha(original/src.name),(dst,src.name,'changed since inspection')
for dst in dests:
 for src in files:shutil.copyfile(src,dst/src.name)
 for row in new['examples'].values():
  for kind in ['loop','poster']:assert sha(dst/row['assets'][kind])==row['assets'][kind+'_sha256']
(O/'media_publication.json').write_text(json.dumps({'count':100,'destinations':[str(p.relative_to(R)) for p in dests],'files_per_destination':101,'source_selection_unchanged':True,'originals':str(original.relative_to(R)),'minimum_ssim':quality['minimum'],'phone_verified':False},indent=2)+'\n')
print('Published 100 H264 references to Flutter/native/desktop; catalog hashes verified')
