"""Matched iPhone 13 timing of the August and 96.83-chain interval recognizers (FP16 and FP32).

Same procedure as artifacts/reports/phone_precision_v17_20261007 (226 recorded-input example
frames, 20 warm-up frames, two passes in forward/reverse order, hand encoder and word boundary
FP16, all compute units). Only the recognizer package changes between configurations.
App sources are backed up with hashes, patched temporarily, and always restored byte-exactly;
the production app is then rebuilt and reinstalled. No protected evaluation recordings are used.
"""
import hashlib,json,re,shutil,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
SLT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'phone_recognizer'
APP=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app');IOS=APP/'ios'
DEVICE='00008110-00111D1A0130A01E';DERIVED=APP/'build/chain9683_recognizer_20261007'
FILES=['ios/Runner/LiveReel/LiveReelEngine.swift','ios/RunnerTests/RunnerTests.swift','ios/Runner.xcodeproj/project.pbxproj']
EXPECTED={'ios/Runner/LiveReel/LiveReelEngine.swift':'15fafa76c9057edefb1af00334441bfa32102f7a8099d49f559e4a7b89911b26',
          'ios/RunnerTests/RunnerTests.swift':'2dbe32ceeb68611b272f6aa3d337540cbb7e81fbc014dba43d4b55e6b42a41db',
          'ios/Runner.xcodeproj/project.pbxproj':'aa4a130bb348efb73d1231142776f73f1b0f29675b493bb9e47f4096ba2ba4a2'}
PACKAGES=[SLT/'artifacts/coreml/SpanRecognizerV17LocalALettersB8FP32.mlpackage',
          HERE/'downstream_recipe/coreml/SpanRecognizerV17Chain9683LettersB8FP16.mlpackage',
          HERE/'downstream_recipe/coreml/SpanRecognizerV17Chain9683LettersB8FP32.mlpackage']
CONFIGS=[('august_recognizer_fp16','SpanRecognizerV17LocalALettersB8FP16'),
         ('chain9683_recognizer_fp16','SpanRecognizerV17Chain9683LettersB8FP16'),
         ('august_recognizer_fp32','SpanRecognizerV17LocalALettersB8FP32'),
         ('chain9683_recognizer_fp32','SpanRecognizerV17Chain9683LettersB8FP32')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(name,cmd,cwd=IOS,check=True,timeout=3600):
    with (OUT/(name+'.log')).open('w') as f:
        return subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=check,timeout=timeout)
def status(**kw):(OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n')
def patch():
    engine=APP/FILES[0];s=engine.read_text()
    s2=s.replace('boundaryUnits: MLComputeUnits = .all) throws {','boundaryUnits: MLComputeUnits = .all,\n         recognizerName: String = "SpanRecognizerV17LocalALettersB8FP16") throws {',1)
    s2=s2.replace('LiveSpanRecognizer(labels: labels, units: recognizerUnits)','LiveSpanRecognizer(name: recognizerName, labels: labels, units: recognizerUnits)',1)
    assert s2.count('recognizerName')==2,'engine patch did not apply'
    engine.write_text(s2)
    tests=APP/FILES[1];s=tests.read_text()
    start=s.index('    let context = CIContext()',s.index('func testStageProfile'));end=s.index('    let configurations:',start)
    rows=',\n'.join(f'      ("{n}", "{p}")' for n,p in CONFIGS)
    test=('\n  /// Matched recognizer timing; development example clips only, never protected tests.\n'
          '  func testChainRecognizerProfile() throws {\n'+s[start:end]+
          '    XCTAssertGreaterThan(frames.count, 100)\n'
          '    let previousEncoder = LiveHandEncoder.packageName\n    defer { LiveHandEncoder.packageName = previousEncoder }\n'
          '    LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"\n'
          '    let configs: [(String, String)] = [\n'+rows+'\n    ]\n'
          '''    for pass in 0..<2 {
      let order = pass == 0 ? Array(configs.indices) : Array(configs.indices.reversed())
      for index in order {
        try autoreleasepool {
          let (name, recognizer) = configs[index]
          let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
          let engine = try LiveReelEngine(recognizerUnits: .all, encoderUnits: .all,
                boundaryUnits: .all, recognizerName: recognizer)
          try engine.warm()
          for (i, frame) in frames.prefix(20).enumerated() {
            _ = try engine.process(frame, seconds: Double(i) * 0.05)
          }
          engine.resetStream()
          var totals: [Double] = []
          for (i, frame) in frames.enumerated() {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try engine.process(frame, seconds: Double(i) * 0.05)
            totals.append(1000 * (CFAbsoluteTimeGetCurrent() - start))
          }
          let sorted = totals.sorted()
          let row: [String: Any] = ["configuration": name, "pass": pass, "package": recognizer,
            "frames": totals.count, "median_ms": sorted[sorted.count / 2],
            "p90_ms": sorted[Int(Double(sorted.count) * 0.9)],
            "mean_ms": totals.reduce(0,+) / Double(totals.count),
            "thermal_before": thermalBefore,
            "thermal_after": ProcessInfo.processInfo.thermalState.rawValue,
            "low_power": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "samples_ms": totals]
          let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
          print("CHAIN_RESULT " + String(data: data, encoding: .utf8)!)
        }
      }
    }
  }
''')
    tests.write_text(s.replace('  func testStageProfile() throws {',test+'\n  func testStageProfile() throws {',1))
    (OUT/'RunnerTests_benchmark.swift').write_text(test)
    ruby=('require "xcodeproj"\np=Xcodeproj::Project.open(ARGV[0])\nt=p.targets.find{|x| x.name=="Runner"}\n'
          'ARGV[1..].each do |path|\n r=p.main_group.new_reference(path)\n r.source_tree="<absolute>"\n r.path=path\n'
          ' r.last_known_file_type="folder.mlpackage"\n t.source_build_phase.add_file_reference(r)\nend\np.save\n')
    (OUT/'add_packages.rb').write_text(ruby)
    log('add_packages',['ruby',str(OUT/'add_packages.rb'),str(IOS/'Runner.xcodeproj')]+[str(p) for p in PACKAGES])
def restore(backup):
    for rel in FILES:shutil.copy2(backup/Path(rel).name,APP/rel)
    restored={rel:sha(APP/rel) for rel in FILES}
    assert restored==EXPECTED,('restore mismatch',restored)
    return restored
def run():
    OUT.mkdir(exist_ok=False);backup=OUT/'source_backup';backup.mkdir()
    before={rel:sha(APP/rel) for rel in FILES}
    assert before==EXPECTED,('app sources are not in the recorded production state',before)
    for rel in FILES:shutil.copy2(APP/rel,backup/Path(rel).name)
    for p in PACKAGES:assert p.is_dir(),p
    (OUT/'inputs.json').write_text(json.dumps({'device':DEVICE,'configs':CONFIGS,'source_hashes_before':before,
        'packages':{p.name:str(p) for p in PACKAGES},'protected_evaluation_used':False},indent=2)+'\n')
    result='not run'
    try:
        status(state='running',stage='patch');patch()
        status(state='running',stage='device_test')
        common=['xcodebuild','-workspace','Runner.xcworkspace','-scheme','Runner','-configuration','Release',
                '-destination',f'id={DEVICE}','-destination-timeout','60','-derivedDataPath',str(DERIVED),
                '-allowProvisioningUpdates','DEVELOPMENT_TEAM=2BJ6GHCUJ2','ENABLE_TESTABILITY=YES']
        test=log('device_test',common+['-only-testing:RunnerTests/RunnerTests/testChainRecognizerProfile','test'],check=False,timeout=7200)
        rows=[json.loads(m) for m in re.findall(r'CHAIN_RESULT (\{.*\})',(OUT/'device_test.log').read_text())]
        (OUT/'device_samples.json').write_text(json.dumps(rows,indent=2)+'\n')
        result=f'exit {test.returncode}, {len(rows)} result rows'
        if test.returncode!=0 or len(rows)!=2*len(CONFIGS):raise RuntimeError('device test incomplete: '+result)
    except Exception:
        status(state='failed_restoring',error=traceback.format_exc())
        raise
    finally:
        restored=restore(backup)
        build=log('restore_build',common_build:=['xcodebuild','-workspace','Runner.xcworkspace','-scheme','Runner','-configuration','Release',
                  '-destination',f'id={DEVICE}','-derivedDataPath',str(DERIVED),'-allowProvisioningUpdates','DEVELOPMENT_TEAM=2BJ6GHCUJ2','build'],check=False)
        apps=sorted(DERIVED.glob('Build/Products/Release-iphoneos/Runner.app'))
        install=log('restore_install',['xcrun','devicectl','device','install','app','--device',DEVICE,str(apps[0])],check=False) if apps else None
        final=dict(result=result,sources_restored_exactly=restored==EXPECTED,restore_build_exit=build.returncode,
                   restore_install_exit=None if install is None else install.returncode)
        (OUT/'restore.json').write_text(json.dumps(final,indent=2)+'\n')
        state='complete' if 'result rows' in result and result.startswith('exit 0') else 'failed'
        status(state=state,**final)
        subprocess.run(['osascript','-e','display notification "iPhone recognizer benchmark finished or stopped." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
