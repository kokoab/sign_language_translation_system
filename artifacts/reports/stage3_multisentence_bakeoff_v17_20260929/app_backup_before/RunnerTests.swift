import CoreML
import CoreImage
import AVFoundation
import Flutter
import UIKit
import XCTest
@testable import Runner

class RunnerTests: XCTestCase {

  func testStage3ModelCompositionWithoutPhraseOverrides() throws {
    let renderer = try LiveStage3(units: .cpuAndGPU)
    func normalized(_ value: String) -> String {
      value.lowercased().filter { $0.isLetter || $0.isNumber || $0.isWhitespace }
        .split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }
    let cases = [
      ("HELLO MY FRIEND HOW YOU", "Hello, my friend. How are you?"),
      ("HELLO GOOD MORNING HOW YOU FRIEND", "Hello, good morning. How are you, friend?"),
      ("PLEASE GIVE MY CHILD WATER", "Please give my child water."),
      ("I GO DOCTOR TOMORROW MORNING", "I am going to the doctor tomorrow morning."),
      ("HELLO HOW YOU", "Hello. How are you?")
    ]
    for (input, expected) in cases {
      let glosses = input.split(separator: " ").map(String.init)
      let result = renderer.rephrase(glosses, Array(repeating: 0.9, count: glosses.count))
      XCTAssertEqual(result.mode, "t5_efficient_tiny", input)
      XCTAssertEqual(normalized(result.sentence), normalized(expected), input)
      let words = glosses.enumerated().map { index, gloss in
        LiveWord(gloss: gloss, startFrame: index * 60, endFrame: index * 60 + 20,
                 commitFrame: index * 60 + 25, startSeconds: Double(index) * 3,
                 endSeconds: Double(index) * 3 + 1, commitSeconds: Double(index) * 3 + 1.25,
                 score: 0.9, early: false)
      }
      let full = renderer.renderUtterance(words)
      XCTAssertEqual(full.clauses, [glosses], input)
      XCTAssertEqual(full.sentence, result.sentence, input)
      print("STAGE3_MODEL_REPAIR \(input) -> \(result.sentence)")
    }
    let name = renderer.renderSentence(["MY", "NAME", "fs-GELO"], [0.9, 0.9, 0.9])
    XCTAssertEqual(normalized(name.sentence), "my name is gelo")
    XCTAssertEqual(name.mode, "t5_efficient_tiny+slots")
    let negative = renderer.rephrase(["I", "NO", "WANT", "WATER"], [0.9, 0.9, 0.9, 0.9])
    XCTAssertTrue(negative.sentence.lowercased().contains("not") || negative.sentence.lowercased().contains("don't"))
    XCTAssertTrue(negative.sentence.lowercased().contains("water"))
    XCTAssertEqual(negative.mode, "t5_efficient_tiny")
    print("STAGE3_MODEL_REPAIR_CHECKPOINT \(renderer.checkpointIdentifier)")
  }

  func testLatestFrameDropsBacklogAndInvalidatesStoppedWork() throws {
    let gate = LiveLatestFrameGate<Int>()
    XCTAssertFalse(gate.offer(0))
    gate.setEnabled(true)
    let firstGeneration = try XCTUnwrap(gate.activeGeneration)
    XCTAssertTrue(gate.offer(1))
    XCTAssertEqual(gate.take()?.0, 1)
    // A slow recognition job is in flight while capture receives newer frames.
    for frame in 2...100 { XCTAssertFalse(gate.offer(frame)) }
    XCTAssertTrue(gate.complete())
    XCTAssertEqual(gate.take()?.0, 100)
    XCTAssertFalse(gate.complete())
    XCTAssertTrue(gate.offer(101))
    gate.setEnabled(false)
    XCTAssertFalse(gate.isCurrent(firstGeneration))
    XCTAssertNil(gate.take())
    XCTAssertFalse(gate.offer(102))
    // Restart before the cancelled worker exits must still schedule only one worker.
    gate.setEnabled(true)
    XCTAssertFalse(gate.offer(999, generation: firstGeneration))
    XCTAssertFalse(gate.offer(103))
    XCTAssertTrue(gate.complete())
    XCTAssertEqual(gate.take()?.0, 103)
    XCTAssertFalse(gate.complete())
    XCTAssertTrue(gate.offer(104))
  }

  func testIndependentHandTrackingUnderRecognitionLoad() throws {
    let key = FlutterDartProject.lookupKey(forAsset: "assets/gloss_examples/HELLO.mp4")
    let path = try XCTUnwrap(Bundle.main.path(forResource: key, ofType: nil))
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: path)))
    generator.appliesPreferredTrackTransform = true
    let context = CIContext()
    var frames: [CVPixelBuffer] = []
    for i in 0..<16 {
      let image = try generator.copyCGImage(at: CMTime(seconds: Double(i) * 0.05, preferredTimescale: 600), actualTime: nil)
      var value: CVPixelBuffer?
      CVPixelBufferCreate(kCFAllocatorDefault, 1280, 720, kCVPixelFormatType_32BGRA,
                          [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &value)
      let buffer = try XCTUnwrap(value)
      let scale = max(1280.0 / Double(image.width), 720.0 / Double(image.height))
      let input = CIImage(cgImage: image).transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        .transformed(by: CGAffineTransform(translationX: (1280 - Double(image.width) * scale) / 2,
                                          y: (720 - Double(image.height) * scale) / 2))
      context.render(input, to: buffer)
      frames.append(buffer)
    }
    let engine = try LiveReelEngine()
    try engine.warm()
    let gate = LiveLatestFrameGate<(CVPixelBuffer, Double)>()
    let modelQueue = DispatchQueue(label: "test.recognition", qos: .userInitiated)
    let tracker = LiveVision(auxiliaryInterval: 4), scaler = LiveDetectionScaler()
    var modelFrames = 0, modelErrors = 0
    func runModel() {
      if let (input, _) = gate.take() {
        do { _ = try engine.process(input.0, seconds: input.1); modelFrames += 1 }
        catch { modelErrors += 1 }
      }
      if gate.complete() { modelQueue.async { runModel() } }
    }
    gate.setEnabled(true)
    var costs: [Double] = [], times: [Double] = []
    let begin = CFAbsoluteTimeGetCurrent()
    for i in 0..<120 {
      let wait = begin + Double(i) * 0.05 - CFAbsoluteTimeGetCurrent()
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }
      let frame = frames[i % frames.count]
      let seconds = CFAbsoluteTimeGetCurrent() - begin
      if gate.offer((frame, seconds)) { modelQueue.async { runModel() } }
      let start = CFAbsoluteTimeGetCurrent()
      let observation = try autoreleasepool {
        try tracker.observe(detection: scaler.resize(frame), width: 1280, height: 720, seconds: seconds)
      }
      XCTAssertEqual(observation.seconds, seconds)
      costs.append(1000 * (CFAbsoluteTimeGetCurrent() - start))
      times.append(CFAbsoluteTimeGetCurrent() - begin)
    }
    gate.setEnabled(false)
    modelQueue.sync {}
    XCTAssertEqual(modelErrors, 0)
    XCTAssertGreaterThan(modelFrames, 0)
    let fps = Double(times.count - 1) / (times.last! - times.first!)
    print("INDEPENDENT_TRACKING replay_frames=120; overlay_hz=\(fps); median_ms=\(costs.sorted()[60]); p95_ms=\(costs.sorted()[114]); recognition_frames=\(modelFrames)")
    // This is a six-second recorded-input concurrency check, not a live-camera/thermal claim.
  }

  /// Per-stage phone profile over stitched gloss clips at 1280x720 (sequential; no camera).
  /// Prints one PROFILE line per configuration; the choice of production units follows from it.
  func testStageProfile() throws {
    let context = CIContext()
    var frames: [CVPixelBuffer] = []
    for gloss in ["HELLO", "MY", "NAME", "HOW", "YOU", "PLEASE", "HELP", "THANKYOU"] {
      let key = FlutterDartProject.lookupKey(forAsset: "assets/gloss_examples/\(gloss).mp4")
      let path = try XCTUnwrap(Bundle.main.path(forResource: key, ofType: nil))
      let generator = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: path)))
      generator.appliesPreferredTrackTransform = true
      generator.requestedTimeToleranceBefore = .zero
      generator.requestedTimeToleranceAfter = .zero
      for i in 0..<30 {
        guard let image = try? generator.copyCGImage(at: CMTime(seconds: Double(i) * 0.05, preferredTimescale: 600),
                                                     actualTime: nil) else { break }
        var value: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 1280, 720, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &value)
        let buffer = try XCTUnwrap(value)
        let scale = max(1280.0 / Double(image.width), 720.0 / Double(image.height))
        let input = CIImage(cgImage: image).transformed(by: CGAffineTransform(scaleX: scale, y: scale))
          .transformed(by: CGAffineTransform(translationX: (1280 - Double(image.width) * scale) / 2,
                                            y: (720 - Double(image.height) * scale) / 2))
        context.render(input, to: buffer)
        frames.append(buffer)
      }
    }
    let configurations: [(String, String, MLComputeUnits, MLComputeUnits, Bool)] = [
      ("A_current", "MobileCLIP2S0ImageEncoderV17FP32", .all, .cpuAndGPU, false),
      ("B_fp16_encoder", "MobileCLIP2S0ImageEncoderV17FP16", .all, .cpuAndGPU, false),
      ("C_fp16_recognizer_all", "MobileCLIP2S0ImageEncoderV17FP16", .all, .all, false),
      ("D_encoder_ane", "MobileCLIP2S0ImageEncoderV17FP16", .cpuAndNeuralEngine, .all, false),
      ("E_C_plus_display_tracker", "MobileCLIP2S0ImageEncoderV17FP16", .all, .all, true),
    ]
    for (name, encoder, encoderUnits, recognizerUnits, display) in configurations {
      LiveHandEncoder.packageName = encoder
      let engine = try LiveReelEngine(recognizerUnits: recognizerUnits, encoderUnits: encoderUnits)
      try engine.warm()
      var stop = false
      let tracker = LiveVision(auxiliaryInterval: 4), scaler = LiveDetectionScaler()
      let displayQueue = DispatchQueue(label: "test.display", qos: .userInteractive)
      if display {
        displayQueue.async {
          var i = 0
          while !stop {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try? autoreleasepool { try tracker.observe(detection: scaler.resize(frames[i % frames.count]), width: 1280,
                                                           height: 720, seconds: Double(i) * 0.05) }
            i += 1
            let wait = 0.05 - (CFAbsoluteTimeGetCurrent() - start)
            if wait > 0 { Thread.sleep(forTimeInterval: wait) }
          }
        }
      }
      var totals: [Double] = [], visions: [Double] = [], cropsMs: [Double] = []
      engine.runtime.timing = [:]; engine.wordsOnly.timing = [:]
      let probe = LiveVision(), probeScaler = LiveDetectionScaler()
      for (i, frame) in frames.enumerated() {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try engine.process(frame, seconds: Double(i) * 0.05)
        totals.append(1000 * (CFAbsoluteTimeGetCurrent() - start))
        if i % 4 == 0 {   // Vision and crop cutting alone, on a separate instance
          let v = CFAbsoluteTimeGetCurrent()
          let o = try autoreleasepool { try probe.observe(detection: probeScaler.resize(frame), width: 1280, height: 720,
                                                         seconds: Double(i) * 0.05) }
          visions.append(1000 * (CFAbsoluteTimeGetCurrent() - v))
          CVPixelBufferLockBaseAddress(frame, .readOnly)
          let image = LiveBGRAImage(base: CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self),
                                    width: 1280, height: 720, bytesPerRow: CVPixelBufferGetBytesPerRow(frame))
          let c = CFAbsoluteTimeGetCurrent()
          _ = LiveHandCrops.crops(o, image: image)
          cropsMs.append(1000 * (CFAbsoluteTimeGetCurrent() - c))
          CVPixelBufferUnlockBaseAddress(frame, .readOnly)
        }
      }
      stop = true
      displayQueue.sync {}
      func med(_ v: [Double]) -> Double { v.isEmpty ? 0 : v.sorted()[v.count / 2] }
      func p90(_ v: [Double]) -> Double { v.isEmpty ? 0 : v.sorted()[Int(Double(v.count) * 0.9)] }
      let stages = engine.timing.mapValues { String(format: "%.1f", 1000 * $0 / Double(frames.count)) }
      print("PROFILE \(name) frames=\(frames.count) total_median_ms=\(String(format: "%.1f", med(totals))) total_p90_ms=\(String(format: "%.1f", p90(totals))) vision_median_ms=\(String(format: "%.1f", med(visions))) crop_cut_median_ms=\(String(format: "%.1f", med(cropsMs))) stage_mean_ms=\(stages)")
    }
    LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP32"
  }

  func testFreshOverlayPreservesModelFeatures() throws {
    let key = FlutterDartProject.lookupKey(forAsset: "assets/gloss_examples/HELLO.mp4")
    let path = try XCTUnwrap(Bundle.main.path(forResource: key, ofType: nil))
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: path)))
    generator.appliesPreferredTrackTransform = true
    let context = CIContext()
    let sparse = LiveVision(), dense = LiveVision()
    var oldMS: [Double] = [], newMS: [Double] = []
    for i in 0..<16 {
      let image = try generator.copyCGImage(at: CMTime(seconds: Double(i) * 0.05, preferredTimescale: 600), actualTime: nil)
      let scale = min(1.0, 640.0 / Double(max(image.width, image.height)))
      let width = Int(Double(image.width) * scale), height = Int(Double(image.height) * scale)
      var value: CVPixelBuffer?
      CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                          [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &value)
      let buffer = try XCTUnwrap(value)
      context.render(CIImage(cgImage: image).transformed(by: CGAffineTransform(scaleX: scale, y: scale)), to: buffer)
      var observations: [Bool: LiveObservation] = [:]
      for fresh in (i % 2 == 0 ? [false, true] : [true, false]) {
        let start = CFAbsoluteTimeGetCurrent()
        observations[fresh] = try (fresh ? dense : sparse).observe(detection: buffer, width: width, height: height,
                                                                   seconds: Double(i) * 0.05, displayAuxiliary: fresh)
        let ms = 1000 * (CFAbsoluteTimeGetCurrent() - start)
        if fresh { newMS.append(ms) } else { oldMS.append(ms) }
      }
      XCTAssertEqual(LiveFeatures.raw(observations[false]!), LiveFeatures.raw(observations[true]!),
                     "Display-only body/face refresh must not alter model input")
      XCTAssertNotNil(observations[true]!.displayBody)
      XCTAssertNotNil(observations[true]!.displayFace)
    }
    print("OVERLAY_MODEL_PARITY exact; sparse_median_ms=\(oldMS.sorted()[8]); fresh_median_ms=\(newMS.sorted()[8])")
  }

  func testLetterGeometryAndProvisionalSpellingConflict() {
    var raw = [[Float]](repeating: [Float](repeating: 0, count: 305), count: 16)
    for i in raw.indices {
      for j in 0..<21 { raw[i][j * 5 + 4] = 0.9 }
      for (joint, x, y) in [(5, Float(0.1), Float(0.2)), (9, 0.2, 0.2), (8, 0.3, 0.2), (12, 0.2, 0.1)] {
        raw[i][joint * 5] = x; raw[i][joint * 5 + 1] = y
      }
    }
    var logits = [Float](repeating: -10, count: 27); logits[16] = 10
    XCTAssertEqual(liveRefineLetterGeometry(logits, raw: raw)[6], 10)   // sideways index: Q -> G
    for i in raw.indices { raw[i][8 * 5] = 0.1; raw[i][8 * 5 + 1] = 0.5 }   // downward index
    logits[16] = -10; logits[6] = 10
    let result = liveRefineLetterGeometry(logits, raw: raw)
    XCTAssertEqual(result[16], 10)
    XCTAssertEqual(result.sorted(), logits.sorted())
    for i in raw.indices { raw[i] = [Float](repeating: 0, count: 305) }
    XCTAssertEqual(liveRefineLetterGeometry(logits, raw: raw), logits)
    func word(_ gloss: String, _ start: Double, _ end: Double) -> LiveWord {
      LiveWord(gloss: gloss, startFrame: Int(start * 20), endFrame: Int(end * 20), commitFrame: Int(end * 20) + 6,
               startSeconds: start, endSeconds: end, commitSeconds: end + 0.3, score: 0.9, early: false)
    }
    let buffer = LiveSpellingBuffer()
    _ = buffer.push(word("FS_C", 1, 1.2)); _ = buffer.push(word("FS_B", 1.3, 1.5))
    XCTAssertEqual(buffer.push(word("HELLO", 1, 1.7)).map(\.gloss), ["HELLO"])
    _ = buffer.push(word("FS_G", 2, 2.2)); _ = buffer.push(word("FS_E", 2.4, 2.6))
    XCTAssertEqual(buffer.push(word("TAKE", 2.4, 3.5)).map(\.gloss), ["fs-GE", "TAKE"])
    _ = buffer.push(word("FS_A", 4, 4.2)); _ = buffer.push(word("FS_N", 4.4, 4.6))
    XCTAssertTrue(buffer.tick(7, activeHands: true).isEmpty)
    XCTAssertEqual(buffer.tick(7, activeHands: false).map(\.gloss), ["fs-AN"])
  }

  /// LIVE keeps a visible fingerspelling control (the ATLAS top-bar "Letters" button) that reports
  /// its state; PRACTICE has none (letters are always on there).
  @MainActor func testLiveLettersControlPresent() throws {
    for (mode, expected) in [(LiveReelViewController.Mode.live, 1), (.practice(["HELLO"]), 0)] {
      let controller = LiveReelViewController(mode: mode)
      controller.loadViewIfNeeded()
      controller.view.frame = CGRect(x: 0, y: 0, width: 844, height: 390)
      controller.view.setNeedsLayout(); controller.view.layoutIfNeeded()
      var found: [UIButton] = []
      func visit(_ view: UIView) {
        if let button = view as? UIButton, button.accessibilityLabel == "Letters", !button.isHidden, button.alpha > 0 {
          found.append(button)
        }
        view.subviews.forEach(visit)
      }
      visit(controller.view)
      XCTAssertEqual(found.count, expected)
      if let button = found.first {
        let frame = button.convert(button.bounds, to: controller.view)
        XCTAssertTrue(controller.view.bounds.insetBy(dx: -1, dy: -1).contains(frame), "Offscreen Letters: \(frame)")
        XCTAssertTrue(["On", "Off"].contains(button.accessibilityValue ?? ""))
      }
      controller.viewWillDisappear(false)
    }
  }

  @MainActor func testLiveAndPracticeKeepLandscapeLayout() {
    for mode: LiveReelViewController.Mode in [.live, .practice(["HELLO"])] {
      let controller = LiveReelViewController(mode: mode)
      XCTAssertFalse(controller.supportedInterfaceOrientations.contains(.portrait))
      XCTAssertTrue(controller.supportedInterfaceOrientations.contains(.landscapeLeft))
      controller.loadViewIfNeeded()
      for size in [CGSize(width: 844, height: 390)] {
        controller.view.frame = CGRect(origin: .zero, size: size)
        controller.view.setNeedsLayout(); controller.view.layoutIfNeeded()
        controller.view.setNeedsLayout(); controller.view.layoutIfNeeded()
        func visit(_ view: UIView) {
          if let button = view as? UIButton, ["START", "FINISH", "RESET", "SKIP THIS SIGN"].contains(button.currentTitle ?? "") {
            let frame = button.convert(button.bounds, to: controller.view)
            XCTAssertTrue(controller.view.bounds.insetBy(dx: -1, dy: -1).contains(frame), "Offscreen control: \(frame)")
          }
          view.subviews.forEach(visit)
        }
        visit(controller.view)
      }
      controller.viewWillDisappear(false)
    }
  }

  func testLiveHandEncoderBatchMatchesSerial() throws {
    let encoder = try LiveHandEncoder()
    let crops = (0..<3).map { view in
      (0..<(256 * 256 * 3)).map { UInt8(truncatingIfNeeded: ($0 * (view + 3) + view * 73) ^ ($0 >> 7)) }
    }
    XCTAssertTrue(try encoder.embedBatch([]).isEmpty)
    for count in 1...3 {
      let inputs = Array(crops.prefix(count))
      let reference = try inputs.map { try encoder.embed(rgb: $0) }
      let actual = try encoder.embedBatch(inputs)
      XCTAssertEqual(actual, reference, "Batch must preserve every FP32 embedding exactly")
    }
    var serial: [Double] = [], batched: [Double] = []
    for round in 0..<12 {
      // Alternate execution order to reduce warm-up/thermal order bias.
      for batch in (round % 2 == 0 ? [false, true] : [true, false]) {
        let start = CFAbsoluteTimeGetCurrent()
        if batch { _ = try encoder.embedBatch(crops) }
        else { for crop in crops { _ = try encoder.embed(rgb: crop) } }
        let milliseconds = 1000 * (CFAbsoluteTimeGetCurrent() - start)
        if batch { batched.append(milliseconds) } else { serial.append(milliseconds) }
      }
    }
    print("LIVE_ENCODER_PARITY exact; serial_median_ms=\(serial.sorted()[6]); batch_median_ms=\(batched.sorted()[6])")
  }

  func testStage2ActivityAlignmentUsesCompleteMotionSpan() throws {
    func array(_ shape: [Int]) throws -> MLMultiArray {
      let value = try MLMultiArray(
        shape: shape.map(NSNumber.init(value:)), dataType: .float32
      )
      for index in 0..<value.count { value[index] = 0 }
      return value
    }
    let landmarks = try array([1, 8, 32, 61, 5])
    for frame in 20...45 {
      for node in 0..<42 {
        let offset = (frame * 61 + node) * 5
        landmarks[offset] = NSNumber(value: Float(frame - 20) * 0.02)
        landmarks[offset + 3] = 1
      }
    }
    let raw = try array([1, 8, 32, 61, 3])
    let sourceMask = try array([1, 8, 32])
    let valid = try array([1, 8, 16, 3])
    let boxes = try array([1, 8, 16, 3, 4])
    let mask = try array([1, 8])
    mask[0] = 1
    mask[1] = 1
    let input = Stage2PreparedInput(
      landmarks: landmarks,
      rawLandmarks: raw,
      sourceFrameMask: sourceMask,
      handValid: valid,
      handBoxes: boxes,
      windowMask: mask,
      crops: [],
      windows: 2
    )
    let aligned = try Stage2MobileModelV17.activityAligned(
      input, embeddings: array([1, 8, 16, 3, 512])
    )
    XCTAssertEqual(aligned.sourceFrameRange, 13..<54)
    XCTAssertEqual(aligned.sourceWindow, 1)
    XCTAssertEqual(aligned.landmarks.shape.map { $0.intValue }, [1, 8, 32, 61, 5])
    XCTAssertEqual(aligned.windowMask[0].intValue, 1)
    XCTAssertEqual(aligned.windowMask[1].intValue, 0)
  }

  /// Stage 3 candidate latency (scripts/bench_stage3_latency_export_v17.py). Reads random-weight
  /// packages from Documents/stage3_bench; skipped when absent. Times the first graph (encoder or
  /// prompt prefill) and every decoder step; not a translation test.
  func testStage3CandidateLatency() throws {
    guard #available(iOS 18.0, *) else { throw XCTSkip("stateful Core ML needs iOS 18") }
    let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("stage3_bench")
    guard let data = try? Data(contentsOf: root.appendingPathComponent("manifest.json")) else {
      throw XCTSkip("no stage3_bench manifest in Documents")
    }
    let manifest = try JSONSerialization.jsonObject(with: data) as! [String: [String: Any]]
    let only = (try? String(contentsOf: root.appendingPathComponent("only.txt"), encoding: .utf8))
      .map { Set($0.split(whereSeparator: \.isWhitespace).map(String.init)) }
    let units: [(String, MLComputeUnits)] = [("cpuAndGPU", .cpuAndGPU), ("all", .all)]
    var results: [[String: Any]] = []

    func array(_ description: MLFeatureDescription, fill: Int32 = 0) throws -> MLMultiArray {
      let constraint = try XCTUnwrap(description.multiArrayConstraint)
      let value = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
      switch constraint.dataType {
      case .int32: value.withUnsafeMutableBufferPointer(ofType: Int32.self) { p, _ in p.update(repeating: fill) }
      case .float16: value.withUnsafeMutableBufferPointer(ofType: Float16.self) { p, _ in p.update(repeating: Float16(fill)) }
      default: value.withUnsafeMutableBufferPointer(ofType: Float.self) { p, _ in p.update(repeating: Float(fill)) }
      }
      return value
    }
    func milliseconds(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
    func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
    func p90(_ values: [Double]) -> Double { values.sorted()[min(values.count - 1, Int(Double(values.count) * 0.9))] }

    // Benchmark packages are large; drop folders from earlier manifests and each one once timed.
    for name in (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
    where manifest[name] == nil && !name.hasSuffix(".json") && !name.hasSuffix(".txt") {
      try? FileManager.default.removeItem(at: root.appendingPathComponent(name))
    }
    for tag in manifest.keys.sorted() where only?.contains(tag) ?? true {
      let meta = manifest[tag]!
      let folder = root.appendingPathComponent(tag)
      guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("Step.mlpackage").path) else { continue }
      defer { try? FileManager.default.removeItem(at: folder) }
      let compileStart = DispatchTime.now().uptimeNanoseconds
      let firstURL = try MLModel.compileModel(at: folder.appendingPathComponent("First.mlpackage"))
      let stepURL = try MLModel.compileModel(at: folder.appendingPathComponent("Step.mlpackage"))
      let compileMs = milliseconds(compileStart)
      defer { try? FileManager.default.removeItem(at: firstURL); try? FileManager.default.removeItem(at: stepURL) }
      let tokenInput = meta["token_input"] as! String
      let passThrough = meta["pass_through"] as? [String: String] ?? [:]
      let offset = meta["position_offset"] as? Int ?? 0

      for (unitName, unit) in units {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = unit
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let first: MLModel, step: MLModel
        do {
          first = try MLModel(contentsOf: firstURL, configuration: configuration)
          step = try MLModel(contentsOf: stepURL, configuration: configuration)
        } catch {
          // A compute-unit setting that cannot load (e.g. an ANE compile failure) is a result.
          var row: [String: Any] = meta
          row.merge(["tag": tag, "units": unitName, "load_error": "\(error)"]) { _, new in new }
          results.append(row)
          print("STAGE3_BENCH_LOAD_ERROR \(tag) \(unitName) \(error)")
          continue
        }
        let loadMs = milliseconds(loadStart)
        var firstInputs: [String: MLFeatureValue] = [:]
        for (name, description) in first.modelDescription.inputDescriptionsByName {
          let fill: Int32 = name.contains("mask") ? 1 : 0
          let value = try array(description, fill: fill)
          if name == "last" { value[0] = NSNumber(value: 63) }
          firstInputs[name] = MLFeatureValue(multiArray: value)
        }
        let firstProvider = try MLDictionaryFeatureProvider(dictionary: firstInputs)
        let stepDescriptions = step.modelDescription.inputDescriptionsByName
        let stateful = !step.modelDescription.stateDescriptionsByName.isEmpty

        func generate(_ steps: Int) throws -> (first: Double, steps: [Double]) {
          let start = DispatchTime.now().uptimeNanoseconds
          let firstOut = try first.prediction(from: firstProvider)
          let firstMs = milliseconds(start)
          var inputs: [String: MLFeatureValue] = [:]
          for (name, description) in stepDescriptions {
            if let source = passThrough[name] {
              inputs[name] = firstOut.featureValue(for: source)
            } else {
              inputs[name] = MLFeatureValue(multiArray: try array(description, fill: name.contains("mask") ? 1 : 0))
            }
          }
          let tokens = inputs[tokenInput]!.multiArrayValue!
          let position = inputs["position"]!.multiArrayValue!
          let state = stateful ? step.makeState() : nil
          var previous: Int32 = 0
          var times: [Double] = []
          for index in 0..<steps {
            if tokenInput == "token" { tokens[0] = NSNumber(value: previous) } else { tokens[index] = NSNumber(value: previous) }
            position[0] = NSNumber(value: index + offset)
            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            let stepStart = DispatchTime.now().uptimeNanoseconds
            let out = try state.map { try step.prediction(from: provider, using: $0) } ?? (try step.prediction(from: provider))
            times.append(milliseconds(stepStart))
            previous = out.featureValue(for: "next_token")!.multiArrayValue![0].int32Value
          }
          return (firstMs, times)
        }

        for _ in 0..<2 { _ = try generate(8) }
        var firsts: [Double] = [], steps: [Double] = [], total16: [Double] = [], total48: [Double] = []
        for _ in 0..<8 {
          let run = try generate(16)
          firsts.append(run.first); steps += run.steps; total16.append(run.first + run.steps.reduce(0, +))
        }
        for _ in 0..<3 {
          let run = try generate(48)
          firsts.append(run.first); steps += run.steps; total48.append(run.first + run.steps.reduce(0, +))
        }
        var row: [String: Any] = meta
        row.merge(["tag": tag, "units": unitName, "compile_ms": compileMs, "load_ms": loadMs,
                   "first_ms_median": median(firsts), "step_ms_median": median(steps), "step_ms_p90": p90(steps),
                   "total_16_steps_ms_median": median(total16), "total_16_steps_ms_p90": p90(total16),
                   "total_48_steps_ms_median": median(total48)]) { _, new in new }
        results.append(row)
        let line = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        print("STAGE3_BENCH " + String(data: line, encoding: .utf8)!)
        let output = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
        try output.write(to: root.appendingPathComponent("results.json"))
      }
    }
    XCTAssertFalse(results.isEmpty)
  }

}
