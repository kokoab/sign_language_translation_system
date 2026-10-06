from pathlib import Path
import shutil,hashlib,json
root=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app')
out=Path('/Volumes/secret/SLT/SLT/artifacts/reports/phone_precision_v17_20261007')
files=['ios/Runner/LiveReel/LiveReelEngine.swift','ios/RunnerTests/RunnerTests.swift','ios/Runner.xcodeproj/project.pbxproj']
backup=out/'source_backup'; backup.mkdir(exist_ok=True)
records={}
for rel in files:
 p=root/rel; q=backup/p.name
 if q.exists(): raise RuntimeError('backup already exists')
 shutil.copy2(p,q); records[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'source_hashes_before.json').write_text(json.dumps(records,indent=2))
p=root/files[0];s=p.read_text()
s=s.replace('boundaryUnits: MLComputeUnits = .all) throws {','boundaryUnits: MLComputeUnits = .all,\n         recognizerName: String = "SpanRecognizerV17LocalALettersB8FP16",\n         wordBoundaryName: String = "AVBoundaryStudentV17L6FP16") throws {',1)
s=s.replace('LiveSpanRecognizer(labels: labels, units: recognizerUnits)','LiveSpanRecognizer(name: recognizerName, labels: labels, units: recognizerUnits)',1)
s=s.replace('LiveBoundaryModel(name: "AVBoundaryStudentV17L6FP16", units: boundaryUnits)','LiveBoundaryModel(name: wordBoundaryName, units: boundaryUnits)',1)
p.write_text(s)
p=root/files[1];s=p.read_text()
start=s.index('    let context = CIContext()',s.index('func testStageProfile'))
end=s.index('    let configurations:',start)
generator=s[start:end]
test='''
  /// Matched precision experiment; development example clips only, never protected tests.
  func testMatchedPrecisionProfile() throws {
'''+generator+'''
    XCTAssertGreaterThan(frames.count, 100)
    let previousEncoder = LiveHandEncoder.packageName
    defer { LiveHandEncoder.packageName = previousEncoder }
    let configs: [(String, String, String, String, MLComputeUnits)] = [
      ("selected_fp16_all", "FP16", "FP16", "FP16", .all),
      ("recognizer_fp32_only", "FP16", "FP32", "FP16", .all),
      ("encoder_fp32_only", "FP32", "FP16", "FP16", .all),
      ("all_visual_fp32", "FP32", "FP32", "FP32", .all),
      ("selected_recognizer_cpu_gpu", "FP16", "FP16", "FP16", .cpuAndGPU),
    ]
    for pass in 0..<2 {
      let order = pass == 0 ? Array(configs.indices) : Array(configs.indices.reversed())
      for index in order {
        try autoreleasepool {
          let (name, enc, rec, bound, units) = configs[index]
          let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
          LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17" + enc
          let engine = try LiveReelEngine(recognizerUnits: units, encoderUnits: .all,
                boundaryUnits: .all, recognizerName: "SpanRecognizerV17LocalALettersB8" + rec,
                wordBoundaryName: "AVBoundaryStudentV17L6" + bound)
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
          let row: [String: Any] = ["configuration": name, "pass": pass,
            "frames": totals.count, "median_ms": sorted[sorted.count / 2],
            "p90_ms": sorted[Int(Double(sorted.count) * 0.9)],
            "mean_ms": totals.reduce(0,+) / Double(totals.count),
            "thermal_before": thermalBefore,
            "thermal_after": ProcessInfo.processInfo.thermalState.rawValue,
            "low_power": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "samples_ms": totals]
          let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
          print("PRECISION_RESULT " + String(data: data, encoding: .utf8)!)
        }
      }
    }
  }
'''
s=s.replace('  func testStageProfile() throws {',test+'\n  func testStageProfile() throws {',1)
p.write_text(s)
(out/'RunnerTests_benchmark.swift').write_text(test)
print('Temporary test and injectable model names prepared; production defaults unchanged')
