from pathlib import Path
import json,hashlib,subprocess
R=Path('/Volumes/secret/SLT/SLT');O=Path(__file__).resolve().parent
S=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel')
H=R/'artifacts/reports/my_have_phone_v17_20261011'
manifest=[]
for variant in ['historical','current']:
 d=O/variant;d.mkdir(exist_ok=True)
 sources={name:S/name for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift']}
 if variant=='historical':
  sources['LiveReelEngine.swift']=R/'artifacts/reports/similar_sign_local_audit_v17_20261008/app_source_backup/LiveReelEngine.swift'
  sources['LiveReelDecoder.swift']=R/'artifacts/reports/citizen_variant_local_filter_v17_20261008/letter_gate_replay/app_source_backup/LiveReelDecoder.swift'
 for name,p in sources.items():
  text=p.read_text()
  if name=='LiveReelCore.swift':
   needle='private(set) var processed = 0'; assert needle in text
   text=text.replace(needle,needle+'\n    func setDiagnosticPhase(_ phase: Int) { processed = phase }')
  if name=='LiveReelEngine.swift' and variant=='historical':
   text=text.replace('left: observation.left, right: observation.right, seconds: seconds)', 'left: observation.left, right: observation.right, body: observation.body, seconds: seconds)')
  if name=='LiveReelModels.swift':
   text=text.replace('name: String = "SpanRecognizerV17FragmentRejectLettersB8FP16"','name: String = ProcessInfo.processInfo.environment["MODEL"] ?? "SpanRecognizerV17FragmentRejectLettersB8FP16"')
  (d/name).write_text(text)
  manifest.append({'variant':variant,'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'copy':str(d/name),'copy_sha256':hashlib.sha256(text.encode()).hexdigest()})
 main=(H/'main.swift').read_text().replace('if seconds > 16 { break }','if seconds > (item["limit"] as? Double ?? 10000) { break }')
 (d/'main.swift').write_text(main)
 cmd=['xcrun','swiftc','-O','-o',str(O/(variant+'_probe'))]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']]
 with (O/(variant+'_build.log')).open('w') as f:
  r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(variant,'build',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
(O/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
clips=[{'id':'sept30_my_friend','path':str(R/'data/local/phone_session_pull_v17_20261009b/live_reel_videos/20260930_152243_04.mp4'),'reference':['HELLO','MY','FRIEND','I','FEEL','EXCITED']},
 {'id':'oct11_my_name','path':str(H/'device_inputs/take03.mp4'),'limit':16}]
cases=[dict(c,id=c['id']+f'_phase{p}',phase=p) for c in clips for p in range(8)]
cases += [{'id':'have_take11','phase':0,'path':str(R/'data/local/phone_session_pull_v17_20261009b/live_reel_videos/20261009_135024_11.mp4')}]
assert all(Path(c['path']).is_file() for c in cases)
(O/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
