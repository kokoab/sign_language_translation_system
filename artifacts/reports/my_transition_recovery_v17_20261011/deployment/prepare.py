from pathlib import Path
import json,hashlib,shutil,difflib
D=Path(__file__).resolve().parent;O=D.parent;N=O/'merge_native';R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((O/'candidate_source_manifest.json').read_text())
assert all(sha(O/p)==h for p,h in manifest['pins'].items())
assert json.loads((O/'final_verification.json').read_text())['all_pass']
before=json.loads((O/'phone_source_before.json').read_text())
assert all(sha(R/p)==h for p,h in before.items())
paths=list(before)+['Runner.xcodeproj/project.pbxproj']
for p in paths:
 dest=D/'before'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/p,dest)
(D/'sources_before.json').write_text(json.dumps({p:sha(R/p) for p in paths},indent=2)+'\n')
s=(N/'LiveReelDecoder.swift').read_text()
# Remove audit-only blocks without changing recognition logic.
for marker in ['        if Experiment.audit {','                if Experiment.audit &&','                if Experiment.audit {']:
 while marker in s:
  start=s.index(marker);brace=s.index('{',start);depth=1;i=brace+1
  while depth:
   depth+=(s[i]=='{')-(s[i]=='}');i+=1
  s=s[:start]+s[i:]
s='\n'.join(l for l in s.splitlines() if 'Experiment.events.append' not in l)+'\n'
start=s.index('                    let eventKey=');end=s.index('\n                }\n            }\n            if changed',start);s=s[:start]+s[end:]
start=s.index('            if let originalGloss = g.gloss {');end=s.index('                let sameGap =',start)
s=s[:start]+'            if let gloss = g.gloss {\n                if gloss=="MY" {lastMyEnd=max(lastMyEnd,seconds(end))}\n'+s[end:]
s=s.replace('LiveMyLikeMotion.mode == "on", ','').replace('!(LiveMyLikeMotion.mode != "off" && gloss == "LIKE")','gloss != "LIKE"').replace('LiveMyLikeMotion.mode != "off" && ','').replace('let (decision,reason)=LiveMyLikeMotion.assess','let (decision,_)=LiveMyLikeMotion.assess')
h=(N/'LiveMyLikeMotion.swift').read_text().replace('/// Experiment only: original timestamps, same-hand ordered closure; unknown never means MY.','/// Original timestamps and same-hand ordered closure; unknown never means MY.')
h='\n'.join(l for l in h.splitlines() if not any(x in l for x in ['static let mode =','static var logged:','static var events:']))+'\n'
e=(N/'Experiment.swift').read_text().replace('// Report-local ablation controls. No production source imports this file.','// Frozen settings of the 2026-10-12 phone-verified transition repair.')
e=e.replace('static var support=0.7','static var support=0.5').replace('static var agreementFloor=2.0','static var agreementFloor=0.9').replace('static var contactRows=3, contactSeconds=0.08, followHorizon=0.5','static var contactRows=2, contactSeconds=0.03, followHorizon=0.35').replace('static var maxDuration=0.35, floor=0.9','static var maxDuration=0.35, floor=0.5')
e='\n'.join(l for l in e.splitlines() if not any(x in l for x in ['static var audit','static var events','static var mix:','static var rank=','static var strong=','static var policy=']))+'\n'
e=e.replace('        if !flat {return true}\n        let rows=', '        return true\n        /* let rows=')
e=e.replace('return rr.count>=2 && rr.last!.t>=end-0.100001 && rr.last!.t-rr.first!.t>=0.05 }','return rr.count>=2 && rr.last!.t>=end-0.100001 && rr.last!.t-rr.first!.t>=0.05 } */')
# flat=false in the selected candidate; remove unreachable ablation branch.
a=e.index('        /* let rows=');b=e.index(' */',a)+3;e=e[:a]+e[b:]
e=e.replace('static var ','static let ').replace('enum Experiment','enum LiveMYTransitionRepair')
s=(s+'\n'+h+'\n'+e).replace('Experiment.','LiveMYTransitionRepair.')
m=(N/'LiveReelModels.swift').read_text().replace('ProcessInfo.processInfo.environment["MODEL"] ?? "SpanRecognizerV17CleanRecoveryLettersB8FP16"','"SpanRecognizerV17ConfidenceDualB8FP16"').replace('Float(ProcessInfo.processInfo.environment["LETTER_THRESHOLD"] ?? "0.5") ?? 0.5','0.5').replace('Experiment.','LiveMYTransitionRepair.')
m=m.replace('    var lastOriginal: [[Float]] = []\n','').replace('lastAdapted=[];lastOriginal=[]','lastAdapted=[]').replace('                    lastOriginal.append(base)\n','')
assert all(x not in s+m for x in ['Experiment','LiveMyLikeMotion.mode','LiveMyLikeMotion.events','LiveMyLikeMotion.logged','processInfo.environment["MODEL"]'])
s='\n'.join(line.rstrip() for line in s.splitlines())+'\n'
for name,text in [('LiveReelDecoder.swift',s),('LiveReelModels.swift',m)]:
 (D/name).write_text(text)
 (D/(name+'.patch')).write_text(''.join(difflib.unified_diff((R/'Runner/LiveReel'/name).read_text().splitlines(True),text.splitlines(True),fromfile='a/'+name,tofile='b/'+name)))
# Parity harness uses identical production model/decoder with existing extraction helpers.
P=D/'native';P.mkdir(exist_ok=True)
for name in ['LiveReelCore.swift','LiveReelEngine.swift']:shutil.copy2(N/name,P/name)
for name in ['LiveReelDecoder.swift','LiveReelModels.swift']:shutil.copy2(D/name,P/name)
main=(N/'main.swift').read_text();main='\n'.join(l for l in main.splitlines() if not l.strip().startswith(('Experiment.','LiveMyLikeMotion.events=')))+'\n'
main=main.replace('"score_events": Experiment.events, "motion_events": LiveMyLikeMotion.events, ','')
(P/'main.swift').write_text(main)
# Cover two difficult current schedules plus old MY, HAVE, and genuine LIKE.
sets=[('joint_all_cases.json','merge_all.json',[0,5]),('joint_phase_cases.json','merge_phase.json',[0,24,48]),('joint_genuine_cases.json','merge_genuine.json',[22,23])]
cases=[];expected=[]
for inputs,output,indices in sets:
 cs=json.loads((O/inputs).read_text());rr={x['id']:x for x in json.loads((O/output).read_text())['records']}
 for i in indices:
  c=cs[i];assert c['id'] in rr;cases.append(c);expected.append(rr[c['id']])
(D/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');(D/'expected.json').write_text(json.dumps({'records':expected})+'\n')
print('Prepared production sources and',len(cases),'parity cases. Production not modified yet.')
