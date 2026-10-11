"""Build a diagnostic-only A/B phone app, restore every source in finally; does not install."""
from pathlib import Path
import json,hashlib,subprocess
D=Path(__file__).resolve().parent;R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios');B=D/'phone_build_backup';B.mkdir(exist_ok=True)
paths=['Runner/LiveReel/LiveReelDecoder.swift','Runner/LiveReel/LiveReelModels.swift','RunnerTests/RunnerTests.swift']
before={p:(R/p).read_bytes() for p in paths};sha=lambda b:hashlib.sha256(b).hexdigest()
for p,b in before.items():
 q=B/Path(p).name
 if q.exists():assert q.read_bytes()==b, "Current source differs from pinned backup: "+p
 else:q.write_bytes(b)
(D/'phone_source_before.json').write_text(json.dumps({p:sha(b) for p,b in before.items()},indent=2))
try:
 native=D/'context_native'
 helper=(native/'Experiment.swift').read_text().replace('static var audit = true','static var audit = false').replace('    static var pre =','    static var modelName="SpanRecognizerV17FragmentRejectLettersB8FP16"\n    static var pre =').replace('static var maxDuration=0.35, floor=0.9','static var maxDuration=0.35, floor=0.5').replace('static var support=0.7','static var support=0.5')
 decoder=(native/'LiveReelDecoder.swift').read_text().replace('                Experiment.events.append(["confirmation":true,','                if Experiment.audit {Experiment.events.append(["confirmation":true,').replace('"available":seconds(T)])\n            }','"available":seconds(T)])}\n            }')
 (R/paths[0]).write_text(decoder+'\n'+(native/'LiveMyLikeMotion.swift').read_text()+'\n'+helper)
 models=(native/'LiveReelModels.swift').read_text().replace('ProcessInfo.processInfo.environment["MODEL"] ?? "SpanRecognizerV17CleanRecoveryLettersB8FP16"','Experiment.modelName')
 needle='        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {'
 models=models.replace(needle,'''        let documents=FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0]
        let diagnostic=documents.appendingPathComponent("my_transition_20261012/"+name+".mlpackage")
        if FileManager.default.fileExists(atPath:diagnostic.path) {
            return try MLModel(contentsOf:MLModel.compileModel(at:diagnostic),configuration:configuration)
        }
'''+needle)
 (R/paths[1]).write_text(models)
 test=(D.parent/'my_have_phone_v17_20261011/SharedAppTest.swift').read_text().replace('func testMYHaveSharedAppPacedReplay() throws {','func myTransitionPacedReplay(_ arm:String,_ pass:Int) throws {').replace('my_have_phone_20261011/take03.mp4','live_reel_videos/20261011_220739_03.mp4')
 needle='    let engine = try XCTUnwrap(LiveReelShared.engine)'
 test=test.replace(needle,'''    try LiveReelShared.queue.sync {
      LiveReelShared.engine=nil
      Experiment.modelName=arm=="candidate" ? "SpanRecognizerV17ConfidenceDualB8FP16" : "SpanRecognizerV17FragmentRejectLettersB8FP16"
      Experiment.floor=0.5;Experiment.maxDuration=0.35;Experiment.support=0.5;Experiment.nextGap=0.4
      Experiment.agreementFloor=0.9;Experiment.onsetFloor=0.4;Experiment.touchLimit=0.06;Experiment.maxContact=0.35
      Experiment.successorFloor=0.85;Experiment.successorGap=0.6
      Experiment.events=[];LiveMyLikeMotion.events=[];LiveMyLikeMotion.logged=[]
      let replacement=try LiveReelEngine(recognizerUnits:.all,encoderUnits:.all)
      try replacement.warm();LiveReelShared.engine=replacement
    }
'''+needle)
 test=test.replace('["words": words.map','["arm":arm,"pass":pass,"words": words.map').replace('"my_have_shared_app_paced.json"','"my_transition_\\(arm)_\\(pass).json"')
 test+='''\nextension RunnerTests {
 func testMyTransitionRecoveryComparison() throws {
   for (pass,arm) in ["baseline","candidate","candidate","baseline"].enumerated() {try myTransitionPacedReplay(arm,pass)}
 }
}\n'''
 (D/'PhoneComparisonTest.swift').write_text(test);(R/paths[2]).write_bytes(before[paths[2]]+('\n'+test).encode())
 patched={p:sha((R/p).read_bytes()) for p in paths};(D/'phone_source_diagnostic.json').write_text(json.dumps(patched,indent=2))
 for p in paths:(D/('phone_diagnostic_'+Path(p).name)).write_bytes((R/p).read_bytes())
 command=['xcodebuild','-workspace','Runner.xcworkspace','-scheme','Runner','-configuration','Release','-destination','id=00008110-00111D1A0130A01E','-derivedDataPath','/Users/frnzlo/slt_my_transition_recovery_build','-only-testing:RunnerTests/RunnerTests/testMyTransitionRecoveryComparison','DEVELOPMENT_TEAM=2BJ6GHCUJ2','CODE_SIGN_IDENTITY=Apple Development','ENABLE_TESTABILITY=YES','build-for-testing']
 (D/'phone_build_command.json').write_text(json.dumps(command,indent=2))
 with (D/'phone_build.log').open('w') as f:completed=subprocess.run(command,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 (D/'phone_build_status.json').write_text(json.dumps({'returncode':completed.returncode,'source_sha256':patched},indent=2))
 if completed.returncode:raise SystemExit(completed.returncode)
finally:
 for p,b in before.items():(R/p).write_bytes(b)
 restored={p:sha((R/p).read_bytes()) for p in paths};assert restored=={p:sha(b) for p,b in before.items()}
 (D/'phone_source_restored.json').write_text(json.dumps(restored,indent=2))
 print('Phone sources restored exactly.',flush=True)
