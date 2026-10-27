
// Post-defense phone metrics (2026-10-27). Appended to RunnerTests.swift in a build mirror only.
// Reads Documents/post_defense/plan.json, replays each listed recording at camera pace through the
// Live screen's own path (latest-frame gate, 20 fps deadline, overlay Vision on the capture thread,
// Finish gesture enabled, captions on the language queue), and writes per-clip results after every clip.
extension RunnerTests {
  func testPostDefenseReplay() throws {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let root = docs.appendingPathComponent("post_defense")
    let planURL = root.appendingPathComponent("plan.json")
    try XCTSkipIf(!FileManager.default.fileExists(atPath: planURL.path), "no plan on this device")
    let plan = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: planURL)) as? [String: Any])
    let arm = try XCTUnwrap(plan["arm"] as? String), run = try XCTUnwrap(plan["run"] as? String)
    let clips = try XCTUnwrap(plan["clips"] as? [[String: Any]])
    let everyFrame = (plan["mode"] as? String) == "every_frame"
    let outURL = root.appendingPathComponent("results/\(run).json")
    try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)

    func footprint() -> Double {
      var info = task_vm_info_data_t()
      var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
      let status = withUnsafeMutablePointer(to: &info) { p in
        p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
          task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
      }
      return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1048576 : -1
    }

    // Let the normal app startup finish, then replace its engine with this arm's models.
    LiveReelShared.warm()
    LiveReelShared.queue.sync {}
    let loadStarted = CACurrentMediaTime()
    var captionModels: LiveCaptionModels?
    var stage3: LiveStage3?
    try LiveReelShared.queue.sync {
      LiveReelShared.engine = nil
      if arm == "fp32" {
        LiveModelLocator.packageDirectory = root.appendingPathComponent("fp32_models")
        LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP32"
      } else {
        LiveModelLocator.packageDirectory = nil
        LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"
      }
      let built = try LiveReelEngine(recognizerUnits: .all, encoderUnits: .all)
      try built.warm()
      LiveReelShared.engine = built
      captionModels = try LiveCaptionModels(spec: LiveModelLocator.resource("live_captions_v17", "json"),
                                            weights: LiveModelLocator.resource("live_captions_v17", "bin"))
      stage3 = try LiveStage3(units: .cpuAndGPU)
      stage3?.warm()
    }
    defer { LiveModelLocator.packageDirectory = nil; LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16" }
    let loadMS = (CACurrentMediaTime() - loadStarted) * 1000
    let engine = try XCTUnwrap(LiveReelShared.engine)
    let models = try XCTUnwrap(captionModels), renderer = try XCTUnwrap(stage3)
    var records: [[String: Any]] = []
    var peakMB = footprint()

    for clip in clips {
      let id = try XCTUnwrap(clip["id"] as? String)
      let url = root.appendingPathComponent("videos/" + (try XCTUnwrap(clip["file"] as? String)))
      let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
      LiveReelShared.queue.sync {
        engine.lettersAlways = false
        engine.fingerspellSign = false
        engine.nameLetters = false
        engine.reset()
      }
      let captions = LiveCaptions(models)
      var captionMS: [Double] = []

      let asset = AVURLAsset(url: url)
      let sem = DispatchSemaphore(value: 0)
      var track: AVAssetTrack?
      Task { track = try? await asset.loadTracks(withMediaType: .video).first; sem.signal() }
      sem.wait()
      let reader = try AVAssetReader(asset: asset)
      let output = AVAssetReaderTrackOutput(track: try XCTUnwrap(track),
                                            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
      reader.add(output)
      reader.startReading()

      let gate = LiveLatestFrameGate<CMSampleBuffer>()
      let queue = LiveReelShared.queue
      let poseScaler = LiveDetectionScaler()
      let poseVision = LiveVision(auxiliaryInterval: 4)
      var poseDeadline = 0.0, deadline = 0.0
      var firstStamp: CMTime?
      var words: [LiveWord] = [], processed: [Double] = [], compute: [Double] = []
      var finishHeld = 0, offered = 0, width = 0, height = 0
      var gestureFinished = false
      let lock = NSLock()
      func processCaptured(_ sample: CMSampleBuffer) {
        guard let frame = CMSampleBufferGetImageBuffer(sample) else { return }
        let stamp = CMSampleBufferGetPresentationTimeStamp(sample)
        if firstStamp == nil { firstStamp = stamp }
        let seconds = CMTimeGetSeconds(CMTimeSubtract(stamp, firstStamp!))
        guard seconds + 1e-6 >= deadline else { return }
        deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
        guard let step = try? engine.process(frame, seconds: seconds, allowFinishGesture: true) else { return }
        lock.lock()
        words += step.committed; processed.append(seconds); compute.append(step.computeSeconds)
        if step.finishProgress != nil { finishHeld += 1 }
        if step.finishRequested { gestureFinished = true }
        if processed.count % 20 == 0 { peakMB = max(peakMB, footprint()) }
        lock.unlock()
        let committed = step.committed, held = step.finishProgress != nil
        LiveReelShared.languageQueue.async {          // the Live screen's caption() path
          let started = CACurrentMediaTime()
          for w in committed { captions.push(w.gloss, score: w.score, seconds: seconds) }
          if !held { _ = captions.tick(seconds) }
          if !committed.isEmpty { captionMS.append((CACurrentMediaTime() - started) * 1000) }
        }
      }
      func processNext() {
        if let (sample, generation) = gate.take(), gate.isCurrent(generation) { processCaptured(sample) }
        if gate.complete() { queue.async { processNext() } }
      }
      let start = Date()
      if everyFrame {
        // Same schedule as the Mac harness (LiveReelHarness runVideos): source-frame clock, 20 fps deadline,
        // every due frame analysed in order with no pacing and no skipping; Finish gesture off.
        var rate: Float = 30
        let rateSem = DispatchSemaphore(value: 0)
        Task { rate = (try? await track!.load(.nominalFrameRate)) ?? 30; rateSem.signal() }
        rateSem.wait()
        var index = 0
        while let sample = output.copyNextSampleBuffer(), let frame = CMSampleBufferGetImageBuffer(sample) {
          let seconds = Double(index) / Double(rate)
          index += 1; offered += 1
          width = CVPixelBufferGetWidth(frame); height = CVPixelBufferGetHeight(frame)
          if seconds + 1e-6 < deadline { continue }
          deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
          let step = try queue.sync { try engine.process(frame, seconds: seconds) }
          words += step.committed; processed.append(seconds); compute.append(step.computeSeconds)
          if processed.count % 20 == 0 { peakMB = max(peakMB, footprint()) }
          let committed = step.committed
          LiveReelShared.languageQueue.async {
            let started = CACurrentMediaTime()
            for w in committed { captions.push(w.gloss, score: w.score, seconds: seconds) }
            _ = captions.tick(seconds)
            if !committed.isEmpty { captionMS.append((CACurrentMediaTime() - started) * 1000) }
          }
        }
      }
      gate.setEnabled(!everyFrame)
      var first: Double?
      while !everyFrame, let sample = output.copyNextSampleBuffer() {
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
        if first == nil { first = pts }
        let wait = (pts - first!) - Date().timeIntervalSince(start)
        if wait > 0 { Thread.sleep(forTimeInterval: wait) }       // camera pace
        guard let generation = gate.activeGeneration, let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
        width = CVPixelBufferGetWidth(frame); height = CVPixelBufferGetHeight(frame)
        XCTAssertLessThanOrEqual(max(width, height), LiveReelEngine.maximumSide)
        offered += 1
        if gate.offer(sample, generation: generation) { queue.async { processNext() } }
        let seconds = pts - first!
        if seconds + 1e-6 >= poseDeadline {                          // overlay Vision on the capture thread
          poseDeadline = max(poseDeadline + 1 / LiveReelEngine.fps, seconds)
          _ = try? autoreleasepool { try poseVision.observe(detection: try poseScaler.resize(frame), width: CVPixelBufferGetWidth(frame),
                                                            height: CVPixelBufferGetHeight(frame), seconds: seconds) }
        }
      }
      queue.sync {}
      Thread.sleep(forTimeInterval: 1)
      queue.sync {}
      gate.setEnabled(false)
      // Finish: flush the open segment (end-of-recording flush, not a live commit latency).
      let tail = try queue.sync { try engine.finish() }
      lock.lock(); words += tail; lock.unlock()
      let tailSeconds = tail.last?.commitSeconds ?? processed.last ?? 0
      // Finish → English text, current iPhone path (live captions), timed on the language queue.
      var captionSentence = "", captionFinishMS = 0.0
      LiveReelShared.languageQueue.sync {
        let started = CACurrentMediaTime()
        for w in tail { captions.push(w.gloss, score: w.score, seconds: tailSeconds) }
        captionSentence = captions.isEmpty ? "" : captions.finish()
        captionFinishMS = (CACurrentMediaTime() - started) * 1000
      }
      // Finish → English text, manuscript path (incremental T5-efficient-tiny), replayed after the
      // stream so it does not load the frame timing: sentence locks during signing, Finish renders the tail.
      let translator = LiveIncrementalTranslator()
      var t5AppendMS: [Double] = []
      let flowing = Array(words.dropLast(tail.count))
      for w in flowing {
        let started = CACurrentMediaTime()
        _ = translator.append([w], using: renderer)
        t5AppendMS.append((CACurrentMediaTime() - started) * 1000)
      }
      let t5Started = CACurrentMediaTime()
      let t5 = words.isEmpty ? (sentence: "", clauses: [[String]](), locked: 0, tailWords: 0) : translator.finish(tail, using: renderer)
      let t5FinishMS = (CACurrentMediaTime() - t5Started) * 1000

      lock.lock()
      for w in tail { w.computeSeconds = -1 }
      records.append([
        "id": id, "reference": clip["reference"] ?? [], "hypothesis": words.map(\.gloss), "words": words.map(\.json),
        "tail_count": tail.count, "processed_seconds": processed, "compute_seconds": compute,
        "frames_offered": offered, "frames_processed": processed.count, "width": width, "height": height,
        "finish_held_frames": finishHeld, "gesture_finished": gestureFinished,
        "wall_seconds": Date().timeIntervalSince(start),
        "caption_update_ms": captionMS, "caption_sentence": captionSentence, "caption_finish_ms": captionFinishMS,
        "t5_sentence": t5.sentence, "t5_locked": t5.locked, "t5_append_ms": t5AppendMS, "t5_finish_ms": t5FinishMS,
        "thermal_before": thermalBefore, "thermal_after": ProcessInfo.processInfo.thermalState.rawValue,
      ])
      lock.unlock()
      let summary: [String: Any] = ["arm": arm, "run": run, "mode": everyFrame ? "every_frame" : "camera_paced", "load_ms": loadMS, "peak_sampled_mb": peakMB,
                                    "device_model": "iPhone", "system_version": ProcessInfo.processInfo.operatingSystemVersionString,
                                    "complete": records.count == clips.count, "records": records]
      try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]).write(to: outURL)
    }
  }
}
