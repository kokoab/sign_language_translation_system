from pathlib import Path
import json,hashlib,shutil
O=Path(__file__).resolve().parent; R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios')
B=O/'phone_backup';B.mkdir(exist_ok=True)
paths=['Runner/LiveReel/LiveReelDecoder.swift','Runner/LiveReel/LiveReelCore.swift','Runner/LiveReel/LiveReelEngine.swift','Runner/LiveReel/LiveReelModels.swift','Runner/LiveReel/LiveStage3.swift','RunnerTests/RunnerTests.swift','Runner/AppDelegate.swift']
hashes={}
for rel in paths:
 p=R/rel
 if not p.exists():continue
 q=B/p.name
 assert not q.exists(),q
 shutil.copy2(p,q);hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
(O/'phone_source_before.json').write_text(json.dumps(hashes,indent=2)+'\n')
assert (R/'Runner/LiveReel/LiveReelDecoder.swift').read_bytes()==(O.parent/'temporal_repair_training_v17_20261011/eval_control/LiveReelDecoder.swift').read_bytes()
helper=(O/'LiveMyLikeMotion.swift').read_text().replace('static let mode =','static var mode =')
(R/'Runner/LiveReel/LiveReelDecoder.swift').write_text((O/'native/LiveReelDecoder.swift').read_text()+'\n'+helper)
s=(O.parent/'my_have_phone_v17_20261011/SharedAppTest.swift').read_text()
s=s.replace('func testMYHaveSharedAppPacedReplay() throws {','func orderedMotionPacedReplay(_ arm: String, _ pass: Int) throws {\n    LiveMyLikeMotion.mode=arm; LiveMyLikeMotion.events=[]; LiveMyLikeMotion.logged=[]')
s=s.replace('"words": words.map', '"arm":arm,"pass":pass,"motion_events":LiveMyLikeMotion.events,"words": words.map')
s=s.replace('"my_have_shared_app_paced.json"','"ordered_my_like_\\(arm)_\\(pass).json"')
s += '\nextension RunnerTests {\n func testOrderedMyLikeSharedApp() throws {\n  for (pass,arm) in ["off","on","on","off"].enumerated() { try orderedMotionPacedReplay(arm,pass) }\n }\n}\n'
(O/'PhoneTest.swift').write_text(s)
(R/'RunnerTests/RunnerTests.swift').write_text((B/'RunnerTests.swift').read_text()+'\n'+s)
print('Prepared temporary phone test. Normal Core, Engine, Models and AppDelegate unchanged; shared resident English model, latest-frame gate and overlay exercised in ABBA order.')
