from pathlib import Path
import hashlib,json,shutil,difflib
D=Path(__file__).resolve().parent;O=D.parent;G=O/'gap175';R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios')
before=json.loads((O/'phone_source_before.json').read_text())
for rel,h in before.items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==h,rel
(D/'sources_before.json').write_text(json.dumps(before,indent=2)+'\n')
shutil.copy2(R/'Runner/LiveReel/LiveReelDecoder.swift',D/'previous_LiveReelDecoder.swift')
s=(G/'native/LiveReelDecoder.swift').read_text()
start=s.index('                    let eventKey=');end=s.index('\n                }\n            }\n            if changed',start)
s=s[:start]+s[end:]
start=s.index('            if let originalGloss = g.gloss {');end=s.index('                let sameGap =',start)
s=s[:start]+'            if let gloss = g.gloss {\n'+s[end:]
s=s.replace('LiveMyLikeMotion.mode == "on", ','').replace('!(LiveMyLikeMotion.mode != "off" && gloss == "LIKE")','gloss != "LIKE"').replace('LiveMyLikeMotion.mode != "off" && ','').replace('let (decision,reason)=LiveMyLikeMotion.assess','let (decision,_)=LiveMyLikeMotion.assess')
s=s.replace('        if mode != .letters, let my=', '        // Confirm ambiguous LIKE as MY only with observed flat contact and no pinch.\n        // Unknown motion preserves the model; the original score cache stays unchanged.\n        if mode != .letters, let my=')
h=(G/'LiveMyLikeMotion.swift').read_text()
h=h.replace('/// Experiment only: original timestamps, same-hand ordered closure; unknown never means MY.','/// Original timestamps and same-hand ordered motion; unknown never means MY.')
h='\n'.join(line for line in h.splitlines() if not any(x in line for x in ['static let mode =','static var logged:','static var events:']))+'\n'
assert all(x not in s+h for x in ['LiveMyLikeMotion.mode','LiveMyLikeMotion.events','LiveMyLikeMotion.logged','motionConfirmedMy'])
assert h.count('0.175001')==2
s+='\n'+h
(D/'LiveReelDecoder.swift').write_text(s)
(D/'deployed.patch').write_text(''.join(difflib.unified_diff((D/'previous_LiveReelDecoder.swift').read_text().splitlines(True),s.splitlines(True),fromfile='a/Runner/LiveReel/LiveReelDecoder.swift',tofile='b/Runner/LiveReel/LiveReelDecoder.swift')))
# Native parity uses unchanged report-only extraction helpers; these are never deployed.
N=D/'native';N.mkdir(exist_ok=True)
for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelEngine.swift']:shutil.copy2(G/'native'/name,N/name)
shutil.copy2(D/'LiveReelDecoder.swift',N/'LiveReelDecoder.swift')
m=(G/'native/main.swift').read_text().replace('        LiveMyLikeMotion.events=[]; LiveMyLikeMotion.logged=[]\n','').replace('"motion_events": LiveMyLikeMotion.events, ','')
(N/'main.swift').write_text(m)
# Cover MY→NAME, MY→FRIEND, HAVE, and the two genuine LIKE clips with transient MY cues.
phase=json.loads((O.parent/'temporal_repair_training_v17_20261011/evaluation_cases.json').read_text());genuine=json.loads((O.parent/'temporal_repair_training_v17_20261011/genuine_control_cases.json').read_text())
ids={'sept30_offset0.00000000_phase0','oct11_offset0.00000000_phase1','have_take11_offset0.00000000_phase0','LIKE_5db52de4','LIKE_msasl_RMUydgQr9TE_5'}
cases=[c for c in phase+genuine if c['id'] in ids];assert len(cases)==5
(D/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
print('Prepared production decoder: tested 175ms policy, no diagnostic state or switches. Five focused native parity cases.')
