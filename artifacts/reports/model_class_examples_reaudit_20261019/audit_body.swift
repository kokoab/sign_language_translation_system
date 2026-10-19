// Direct full-clip primary-head audit. Expected labels never enter inference.
func ranked(_ values: [Float], _ labels: [String]) -> [[String: Any]] {
    let m = values.max()!
    let exps = values.map { exp(Double($0 - m)) }
    let sum = exps.reduce(0,+)
    return values.indices.sorted { values[$0] > values[$1] }.prefix(5).map {
        ["label": labels[$0], "probability": exps[$0] / sum, "logit": Double(values[$0])]
    }
}
func dump(_ x: [Float], _ path: URL) throws {
    try x.withUnsafeBufferPointer { p in try Data(buffer: p).write(to: path) }
}
let args = CommandLine.arguments
LiveModelLocator.packageDirectory = URL(fileURLWithPath: args[1])
LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"
do {
    let labels = try JSONSerialization.jsonObject(with: Data(contentsOf: LiveModelLocator.resource("live_reel_labels", "json"))) as! [String]
    let recognizer = try LiveSpanRecognizer(units: .all)
    let encoder = try LiveHandEncoder(units: .all)
    let items = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[2]))) as! [[String: Any]]
    let folder = URL(fileURLWithPath: args[4])
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    var records: [[String: Any]] = []
    for item in items {
        let id = item["id"] as! String
        let vision = LiveVision()
        let scaler = LiveDetectionScaler()
        let reader = try VideoFrames(URL(fileURLWithPath: item["path"] as! String))
        var observations: [LiveObservation] = []
        var hands: [LiveHandFrame] = []
        var index = 0, deadline = 0.0
        while let raw = reader.next() {
            let t = Double(index) / reader.fps
            index += 1
            if t + 1e-6 < deadline { continue }
            deadline = max(deadline + 1 / LiveReelEngine.fps, t)
            try autoreleasepool {
                let frame = limited(raw, LiveReelEngine.maximumSide)
                let detection = try scaler.resize(frame)
                let w = CVPixelBufferGetWidth(frame), h = CVPixelBufferGetHeight(frame)
                let o = try vision.observe(detection: detection, width: w, height: h, seconds: t)
                observations.append(o)
                if o.left == nil && o.right == nil { hands.append(LiveHandFrame()); return }
                CVPixelBufferLockBaseAddress(frame, .readOnly)
                defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
                let image = LiveBGRAImage(base: CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self), width: w, height: h, bytesPerRow: CVPixelBufferGetBytesPerRow(frame))
                hands.append(try encoder.frameHand(o, image: image))
            }
        }
        var record: [String: Any] = ["id": id, "path": item["path"]!, "decoded_frames": index, "sampled_frames": observations.count]
        if let input = LiveSpanRecognizer.inputs(observations, hands) {
            let logits = try recognizer.logits([input], repairEligible: [false])[0]
            record["primary_top5"] = ranked(Array(logits.prefix(100)), labels)
            record["adapted_top5"] = ranked(recognizer.lastAdapted[0], labels)
            for (name, x) in [("landmarks", input.landmarks), ("hand_embeddings", input.embeddings), ("hand_valid", input.valid), ("hand_boxes", input.boxes)] {
                try dump(x, folder.appendingPathComponent(id + "_" + name + ".f32"))
            }
        } else { record["error"] = "no valid native full-clip span" }
        records.append(record)
        try JSONSerialization.data(withJSONObject: ["records": records], options: [.sortedKeys]).write(to: URL(fileURLWithPath: args[3]))
        print(id, record["primary_top5"] ?? record["error"]!)
        fflush(stdout)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
