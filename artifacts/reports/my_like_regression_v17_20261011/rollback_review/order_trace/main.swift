import Foundation
import CoreML
func readJSON(_ path: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) }
func writeJSON(_ value: Any, _ path: String) throws { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: URL(fileURLWithPath:path)) }

func runTraceWords(_ path: String, _ out: String) throws {
    let trace = try readJSON(path) as! [String: Any]
    let frames = (trace["frames"] as! [[String: Any]]).filter { $0["start"] == nil && ($0["held"] as? Bool) != true }
    func points(_ row: [String: Any], _ key: String) -> [LivePoint] {
        (row[key] as! [[Double]]).map { LivePoint(x: Float($0[0]), y: Float($0[1]), c: Float($0[2])) }
    }
    func hand(_ row: [String: Any], _ key: String) -> LiveHand? {
        guard row[key] != nil else { return nil }
        var h = LiveHand(); h.points = points(row, key)
        h.chirality = row[key + "c"] as? String ?? "unknown"
        h.score = Float(row[key + "s"] as? Double ?? 0)
        return h
    }
    let observations = frames.map { r in
        LiveObservation(seconds: r["s"] as! Double, width: r["w"] as! Int, height: r["h"] as! Int,
                        left: hand(r, "l"), right: hand(r, "r"), body: points(r, "b"), face: points(r, "f"),
                        faceForFeatures: r["ff"] as! Bool)
    }
    let hands = frames.map { r -> LiveHandFrame in
        var h = LiveHandFrame()
        let data = Data(base64Encoded: r["e"] as! String)!
        h.embeddings = data.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Float.init) }
        h.valid = (r["v"] as! [Double]).map(Float.init)
        h.boxes = (r["x"] as! [Double]).map(Float.init)
        return h
    }
    let engine = try LiveReelEngine(recognizerUnits: .all)
    let base = engine.wordsGated
    let selected=ProcessInfo.processInfo.environment["MODEL"] ?? "SpanRecognizerV17FragmentRejectLettersB8FP16"
    let recognizer=try LiveSpanRecognizer(name:selected, labels:base.recognizer.labels,letterThreshold:0.5,units:.all)
    let config = LiveDecoderConfig.v3Words
    let runtime = LiveSegmentalRuntime(boundary: base.boundary.model, recognizer: recognizer,
                                      encoder: base.encoder, config: config, mode: .both, letterGate: base.letterGate)
    var words: [LiveWord] = []
    for (o,h) in zip(observations,hands) {
        words += try autoreleasepool { try runtime.observe(o, image: nil, hand: h) }
    }
    words += try runtime.finish()
    let visible = words.filter { !$0.gloss.hasPrefix("FS_") }
    print(visible.map(\.gloss)); fflush(stdout)
    try writeJSON(["frames": frames.count, "words": visible.map(\.json), "hypothesis": visible.map(\.gloss)], out)
}


let args=CommandLine.arguments
LiveModelLocator.packageDirectory=URL(fileURLWithPath:args[1])
try runTraceWords(args[2],args[3])
