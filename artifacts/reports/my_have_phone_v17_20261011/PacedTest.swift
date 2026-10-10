extension RunnerTests {
  func testMYHavePacedReplay() throws {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let url = docs.appendingPathComponent("my_have_phone_20261011/take03.mp4")
    try XCTSkipIf(!FileManager.default.fileExists(atPath: url.path), "take not on this device")
    LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"
    let engine = try LiveReelEngine(recognizerUnits: .all, encoderUnits: .all)
    try engine.warm()
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
    let result: [String: Any] = ["words": words.map(\.json), "hypothesis": words.map(\.gloss), "processed_seconds": processed,
                                 "compute_seconds": compute, "finish_held": finishHeld,
                                 "wall_seconds": Date().timeIntervalSince(start)]
    try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]).write(to: docs.appendingPathComponent("my_have_paced.json"))
  }

}
