
  /// Matched precision experiment; development example clips only, never protected tests.
  func testMatchedPrecisionProfile() throws {
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
