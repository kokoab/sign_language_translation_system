extension RunnerTests {
  func orderedGap175PacedReplay(_ arm: String, _ pass: Int) throws {
    LiveMyLikeMotion.mode=arm; LiveMyLikeMotion.events=[]; LiveMyLikeMotion.logged=[]
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let url = docs.appendingPathComponent("my_have_phone_20261011/take03.mp4")
    try XCTSkipIf(!FileManager.default.fileExists(atPath: url.path), "take not on this device")
    LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"
    // Exercise the normal app startup, including the resident English model.
    let startup = CACurrentMediaTime()
    LiveReelShared.warm()
    LiveReelShared.queue.sync {}
    LiveReelShared.languageQueue.sync {}
    let startupMS = (CACurrentMediaTime() - startup) * 1000
    let engine = try XCTUnwrap(LiveReelShared.engine)
    XCTAssertNotNil(LiveReelShared.stage3)
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
    let readyMB = footprint(), thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
    var peakMB = readyMB

    engine.lettersAlways = false
    engine.reset()
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
    let queue = DispatchQueue(label: "simulated.live.model", qos: .userInitiated)
    let poseScaler = LiveDetectionScaler()
    let poseVision = LiveVision(auxiliaryInterval: 4)
    var poseDeadline = 0.0
    var firstStamp: CMTime?
    var deadline = 0.0
    var words: [LiveWord] = [], processed: [Double] = [], compute: [Double] = []
    var finishHeld = 0
    let lock = NSLock()
    func processCaptured(_ sample: CMSampleBuffer) {
      guard let frame = CMSampleBufferGetImageBuffer(sample) else { return }
      let stamp = CMSampleBufferGetPresentationTimeStamp(sample)
      if firstStamp == nil { firstStamp = stamp }
      let seconds = CMTimeGetSeconds(CMTimeSubtract(stamp, firstStamp!))
      guard seconds + 1e-6 >= deadline else { return }
      deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
      guard let step = try? engine.process(frame, seconds: seconds, allowFinishGesture: true) else { return }
      lock.lock(); words += step.committed; processed.append(seconds); compute.append(step.computeSeconds)
      if step.finishProgress != nil { finishHeld += 1 }
      if processed.count % 20 == 0 { peakMB = max(peakMB, footprint()) }
      lock.unlock()
    }
    func processNext() {
      if let (sample, generation) = gate.take(), gate.isCurrent(generation) { processCaptured(sample) }
      if gate.complete() { queue.async { processNext() } }
    }
    gate.setEnabled(true)
    let start = Date()
    var first: Double?
    while let sample = output.copyNextSampleBuffer() {
      let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
      if first == nil { first = pts }
      let wait = (pts - first!) - Date().timeIntervalSince(start)
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }       // camera pace
      guard let generation = gate.activeGeneration, let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
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
    let tail = try queue.sync { try engine.finish() }
    lock.lock(); words += tail; lock.unlock()
    let result: [String: Any] = ["arm":arm,"pass":pass,"motion_events":LiveMyLikeMotion.events,"words": words.map(\.json), "hypothesis": words.map(\.gloss), "processed_seconds": processed,
                                 "compute_seconds": compute, "finish_held": finishHeld,
                                 "wall_seconds": Date().timeIntervalSince(start), "startup_ms":startupMS, "ready_mb":readyMB,"peak_sampled_mb":peakMB,
                                 "thermal_before":thermalBefore,"thermal_after":ProcessInfo.processInfo.thermalState.rawValue,
                                 "stage3_ready":LiveReelShared.stage3 != nil]
    try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]).write(to: docs.appendingPathComponent("ordered_my_like_gap175_\(arm)_\(pass).json"))
  }

}

extension RunnerTests {
 func testOrderedMyLikeGap175SharedApp() throws {
  for (pass,arm) in ["off","on","on","off"].enumerated() { try orderedGap175PacedReplay(arm,pass) }
 }
}
