from pathlib import Path
import json,hashlib,shutil,subprocess
D=Path(__file__).resolve().parent;O=D.parent;R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((D/'parity.json').read_text())['all_pass']
before=json.loads((D/'sources_before.json').read_text());assert all(sha(R/p)==h for p,h in before.items())
name='SpanRecognizerV17ConfidenceDualB8FP16.mlpackage';src=O/name;dst=R/'Runner/LiveReel/Models'/name
if dst.exists():
 assert {str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file() and not p.name.startswith('._')}=={str(p.relative_to(dst)):sha(p) for p in dst.rglob('*') if p.is_file() and not p.name.startswith('._')}
else:shutil.copytree(src,dst,ignore=shutil.ignore_patterns('._*'))
for n in ['LiveReelDecoder.swift','LiveReelModels.swift']:shutil.copy2(D/n,R/'Runner/LiveReel'/n)
p=R/'Runner.xcodeproj/project.pbxproj';t=p.read_text();old='SpanRecognizerV17FragmentRejectLettersB8FP16.mlpackage';lines=t.splitlines(True);added=0
for i in range(len(lines)-1,-1,-1):
 if old in lines[i]:
  assert '8AEEAAD05DB84F8DE77E8621' in lines[i] or '6FCD9178D8E86FB41C671555' in lines[i]
  clone=lines[i].replace(old,name).replace('8AEEAAD05DB84F8DE77E8621','AA1210202600000000000001').replace('6FCD9178D8E86FB41C671555','AA1210202600000000000002')
  lines.insert(i+1,clone);added+=1
assert added==4;p.write_text(''.join(lines))
(D/'sources_deployed.json').write_text(json.dumps({p:sha(R/p) for p in before},indent=2)+'\n')
(D/'model_source_hashes.json').write_text(json.dumps({str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file() and not p.name.startswith('._')},indent=2)+'\n')
cmd=['xcodebuild','-workspace','Runner.xcworkspace','-scheme','Runner','-configuration','Release','-destination','id=00008110-00111D1A0130A01E','-derivedDataPath','/Users/frnzlo/slt_my_transition_release_build','DEVELOPMENT_TEAM=2BJ6GHCUJ2','CODE_SIGN_IDENTITY=Apple Development','build']
(D/'release_build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (D/'release_build.log').open('w') as f:p=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
(D/'release_build_status.json').write_text(json.dumps({'returncode':p.returncode},indent=2)+'\n')
print('Release build returncode',p.returncode)
if p.returncode:print((D/'release_build.log').read_text()[-4000:])
raise SystemExit(p.returncode)
