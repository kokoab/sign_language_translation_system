// macOS harness for the live segmental Reel Swift port (same sources as the iPhone app).
//
//   harness fixture <resources-dir> <fixture.json> <out.json>   logic parity on Python observations
//   harness crops   <resources-dir> <fixture-dir> <fixture.json> <out.json>   crop/embedding parity
//   harness videos  <resources-dir> <list.json> <out.json>      end-to-end replay (own Vision + crops)
//   harness stage3  <resources-dir> <cases.json> <out.json>     Stage 3 rendering parity
//
// Build: see ios/LiveReelHarness/build.sh

import AVFoundation
import CoreImage
import CoreML
import Foundation
import ImageIO

func readJSON(_ path: String) throws -> Any {
    try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
}

func writeJSON(_ value: Any, _ path: String) throws {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: URL(fileURLWithPath: path))
}

func floats(_ base64: String) -> [Float] {
    let data = Data(base64Encoded: base64)!
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func hand(_ value: Any?) -> LiveHand? {
    guard let d = value as? [String: Any] else { return nil }
    let xy = d["xy"] as! [[Double]], c = d["c"] as! [Double]
    var h = LiveHand()
    for j in 0..<21 { h.points[j] = LivePoint(x: Float(xy[j][0]), y: Float(xy[j][1]), c: Float(c[j])) }
    h.chirality = d["chirality"] as! String
    h.score = Float(d["score"] as! Double)
    return h
}

func points(_ value: Any?, _ n: Int) -> [LivePoint] {
    let d = value as! [String: Any]
    let xy = d["xy"] as! [[Double]], c = d["c"] as! [Double]
    return (0..<n).map { LivePoint(x: Float(xy[$0][0]), y: Float(xy[$0][1]), c: Float(c[$0])) }
}

func observation(_ f: [String: Any]) -> LiveObservation {
    LiveObservation(seconds: f["seconds"] as! Double, width: f["width"] as! Int, height: f["height"] as! Int,
                    left: hand(f["left"]), right: hand(f["right"]), body: points(f["body"], 4),
                    face: points(f["face"], 15), faceForFeatures: f["face_for_features"] as! Bool)
}

func handFrame(_ f: [String: Any]) -> LiveHandFrame {
    let d = f["hand"] as! [String: Any]
    var h = LiveHandFrame()
    h.embeddings = floats(d["emb"] as! String)
    h.valid = (d["valid"] as! [Double]).map(Float.init)
    h.boxes = (d["boxes"] as! [[Double]]).flatMap { $0.map(Float.init) }
    return h
}

func maxAbs(_ a: [Float], _ b: [Float]) -> Double {
    zip(a, b).map { Double(abs($0 - $1)) }.max() ?? 0
}

func runFixture(_ path: String, _ out: String) throws {
    let fixture = try readJSON(path) as! [String: Any]
    let engine = try LiveReelEngine()
    let frames = fixture["frames"] as! [[String: Any]]
    let observations = frames.map(observation)
    let hands = frames.map(handFrame)
    var bioWords: [[Float]] = [], bioLetters: [[Float]] = []
    engine.runtime.words.onBIO = { row, flushed in if !flushed { bioWords.append(row) } }
    engine.runtime.letters.onBIO = { row, flushed in if !flushed { bioLetters.append(row) } }
    var words: [LiveWord] = []
    let started = Date()
    for (o, h) in zip(observations, hands) {
        words += try autoreleasepool { try engine.runtime.observeWithHand(o, hand: h) }
    }
    words += try engine.runtime.finish()
    let seconds = Date().timeIntervalSince(started)
    var final: [LiveWord] = []
    for w in words { final += engine.speller.push(w) }
    final += engine.speller.flush()
    // Sampled spans: inputs and logits from the same observation indices.
    var spanChecks: [[String: Any]] = []
    for span in fixture["spans"] as! [[String: Any]] {
        let ids = span["ids"] as! [Int]
        let x = LiveSpanRecognizer.inputs(ids.map { observations[$0] }, ids.map { hands[$0] })
        guard let reference = span["landmarks"] as? String else {
            spanChecks.append(["ids": ids.count, "python_none": true, "swift_none": x == nil])
            continue
        }
        guard let x else { spanChecks.append(["ids": ids.count, "python_none": false, "swift_none": true]); continue }
        let logits = try engine.runtime.words.recognizer.logits([x])[0]
        let pyLogits = (span["logits"] as! [Double]).map(Float.init)
        spanChecks.append([
            "ids": ids.count, "swift_none": false, "python_none": false,
            "landmarks_max_abs": maxAbs(x.landmarks, floats(reference)),
            "embeddings_max_abs": maxAbs(x.embeddings, floats(span["embeddings"] as! String)),
            "valid_max_abs": maxAbs(x.valid, (span["valid"] as! [[Double]]).flatMap { $0.map(Float.init) }),
            "boxes_max_abs": maxAbs(x.boxes, floats(span["boxes"] as! String)),
            "logits_max_abs": maxAbs(logits, pyLogits),
            "word_argmax_same": logits[0..<100].firstIndex(of: logits[0..<100].max()!)! == pyLogits[0..<100].firstIndex(of: pyLogits[0..<100].max()!)!,
        ])
    }
    func bioDiff(_ a: [[Float]], _ key: String) -> [String: Any] {
        let b = (fixture[key] as! [[Double]]).map { $0.map(Float.init) }
        let n = min(a.count, b.count)
        var worst = 0.0, argDiff = 0
        for i in 0..<n {
            worst = max(worst, maxAbs(Array(a[i][1...]), Array(b[i][1...])))
            let pa = a[i].firstIndex(of: a[i].max()!)!, pb = b[i].firstIndex(of: b[i].max()!)!
            argDiff += pa == pb ? 0 : 1
        }
        return ["swift_rows": a.count, "python_rows": b.count, "max_abs_logprob": worst, "argmax_differs": argDiff]
    }
    let pyWords = (fixture["words"] as! [[String: Any]]).map { $0["gloss"] as! String }
    try writeJSON([
        "id": fixture["id"]!, "frames": frames.count, "decode_seconds": seconds,
        "bio_words": bioDiff(bioWords, "bio_words"), "bio_letters": bioDiff(bioLetters, "bio_letters"),
        "spans": spanChecks, "swift_words": final.map(\.json), "python_words": pyWords,
        "words_equal": final.map(\.gloss) == pyWords,
        "timing": engine.runtime.timing,
    ], out)
}

func loadRGB(_ url: URL) throws -> (bytes: [UInt8], width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw LiveReelError.input("bad png \(url)") }
    let w = image.width, h = image.height
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let context = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) { rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2] }
    return (rgb, w, h)
}

func runCrops(_ dir: String, _ path: String, _ out: String) throws {
    let fixture = try readJSON(path) as! [String: Any]
    let encoder = try LiveHandEncoder()
    var results: [[String: Any]] = []
    for case_ in fixture["crops"] as! [[String: Any]] {
        let frame = try loadRGB(URL(fileURLWithPath: dir).appendingPathComponent(case_["frame"] as! String))
        var bgra = [UInt8](repeating: 255, count: frame.width * frame.height * 4)
        for i in 0..<(frame.width * frame.height) {
            bgra[i * 4] = frame.bytes[i * 3 + 2]; bgra[i * 4 + 1] = frame.bytes[i * 3 + 1]; bgra[i * 4 + 2] = frame.bytes[i * 3]
        }
        let boxes = (case_["boxes"] as! [[Double]]).map { $0.map(Float.init) }
        let crops = case_["crops"] as! [String]
        let embeddings = (case_["embeddings"] as! [String]).map(floats)
        try bgra.withUnsafeBufferPointer { p in
            let image = LiveBGRAImage(base: p.baseAddress!, width: frame.width, height: frame.height, bytesPerRow: frame.width * 4)
            for (k, box) in boxes.enumerated() {
                let mine = LiveHandCrops.cropSquare(image, box: box)
                let reference = try loadRGB(URL(fileURLWithPath: dir).appendingPathComponent(crops[k])).bytes
                let diffs = zip(mine, reference).map { abs(Int($0) - Int($1)) }
                let embedding = try encoder.embed(rgb: mine)
                let referenceEmbedding = try encoder.embed(rgb: reference)
                func cosine(_ a: [Float], _ b: [Float]) -> Double {
                    let dot = zip(a, b).map { Double($0 * $1) }.reduce(0, +)
                    let na = sqrt(a.map { Double($0 * $0) }.reduce(0, +)), nb = sqrt(b.map { Double($0 * $0) }.reduce(0, +))
                    return dot / (na * nb)
                }
                results.append([
                    "box": box, "pixel_max_abs": diffs.max()!, "pixel_mean_abs": Double(diffs.reduce(0, +)) / Double(diffs.count),
                    "cosine_swift_crop_vs_python_embedding": cosine(embedding, embeddings[k]),
                    "cosine_python_crop_swift_encoder_vs_python_embedding": cosine(referenceEmbedding, embeddings[k]),
                ])
            }
        }
    }
    try writeJSON(["crops": results], out)
}

/// Decoded upright frames at the source clock, as cv2.VideoCapture reads them.
final class VideoFrames {
    let reader: AVAssetReader
    let output: AVAssetReaderTrackOutput
    let fps: Double

    init(_ url: URL) throws {
        let asset = AVURLAsset(url: url)
        let semaphore = DispatchSemaphore(value: 0)
        var track: AVAssetTrack?
        var rate: Float = 30
        var transform = CGAffineTransform.identity
        Task {
            track = try? await asset.loadTracks(withMediaType: .video).first
            if let track {
                rate = (try? await track.load(.nominalFrameRate)) ?? 30
                transform = (try? await track.load(.preferredTransform)) ?? .identity
            }
            semaphore.signal()
        }
        semaphore.wait()
        guard let track else { throw LiveReelError.input("no video track") }
        if !transform.isIdentity { FileHandle.standardError.write("warning: non-identity transform in \(url.lastPathComponent)\n".data(using: .utf8)!) }
        reader = try AVAssetReader(asset: asset)
        output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()
        fps = Double(rate)
    }

    func next() -> CVPixelBuffer? {
        guard let sample = output.copyNextSampleBuffer() else { return nil }
        return CMSampleBufferGetImageBuffer(sample)
    }
}

func limited(_ buffer: CVPixelBuffer, _ side: Int) -> CVPixelBuffer {
    let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
    guard max(w, h) > side else { return buffer }
    let scale = Double(side) / Double(max(w, h))
    let tw = Int((Double(w) * scale).rounded()), th = Int((Double(h) * scale).rounded())
    var target: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, tw, th, kCVPixelFormatType_32BGRA, nil, &target)
    CIContext().render(CIImage(cvPixelBuffer: buffer).transformed(by: CGAffineTransform(scaleX: CGFloat(tw) / CGFloat(w), y: CGFloat(th) / CGFloat(h))),
                       to: target!)
    return target!
}

func runVideos(_ path: String, _ out: String, recognizerUnits: MLComputeUnits) throws {
    let list = try readJSON(path) as! [[String: Any]]
    let engine = try LiveReelEngine(recognizerUnits: recognizerUnits)
    var records: [[String: Any]] = []
    var compute: [Double] = []
    for item in list {
        engine.reset()
        let frames = try VideoFrames(URL(fileURLWithPath: item["path"] as! String))
        var index = 0
        var deadline = 0.0
        var words: [LiveWord] = []
        engine.runtime.timing = [:]
        while let frame = frames.next() {
            let seconds = Double(index) / frames.fps
            index += 1
            if seconds + 1e-6 < deadline { continue }
            deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
            let step = try engine.process(limited(frame, LiveReelEngine.maximumSide), seconds: seconds)
            compute.append(step.computeSeconds)
            words += step.committed
        }
        let tail = try engine.finish()
        for w in tail { w.computeSeconds = -1 }       // end-of-file flush, not a live latency
        words += tail
        records.append(["id": item["id"]!, "reference": item["reference"] ?? [], "hypothesis": words.map(\.gloss),
                        "words": words.map(\.json), "timing": engine.runtime.timing, "frames": index])
        print(item["id"]!, words.map(\.gloss))
        fflush(stdout)
    }
    let sorted = compute.sorted()
    try writeJSON(["records": records, "compute_ms": [
        "median": 1000 * sorted[sorted.count / 2], "p90": 1000 * sorted[Int(Double(sorted.count) * 0.9)],
        "max": 1000 * sorted.last!, "frames": sorted.count]], out)
}

func runStage3(_ path: String, _ out: String) throws {
    let cases = try readJSON(path) as! [[String: Any]]
    let stage3 = try LiveStage3()
    var rows: [[String: Any]] = []
    var same = 0
    let started = Date()
    for c in cases {
        let glosses = c["glosses"] as! [String], scores = c["scores"] as! [Double]
        let mine = stage3.renderSentence(glosses, scores)
        let equal = mine.sentence == c["sentence"] as! String
        same += equal ? 1 : 0
        rows.append(["glosses": glosses, "python": c["sentence"]!, "swift": mine.sentence, "mode": mine.mode, "equal": equal])
    }
    try writeJSON(["cases": rows, "equal": same, "total": cases.count,
                   "ms_per_case": 1000 * Date().timeIntervalSince(started) / Double(cases.count)], out)
}

/// Fist-letter refinement parity: Python refine() outputs vs LiveFistGeometry on the same inputs.
func runFist(_ path: String, _ out: String) throws {
    let cases = try readJSON(path) as! [[String: Any]]
    let fist = try LiveFistGeometry(url: LiveModelLocator.resource("fist_geometry_v17", "json"))
    var worst = 0.0, featureWorst = 0.0, argmaxSame = 0, noneAgree = 0
    var worstCase: [String: Any] = [:]
    for (index, c) in cases.enumerated() {
        let lm = floats(c["landmarks"] as! String)
        let logits = (c["logits"] as! [Double]).map(Float.init)
        let python = (c["refined"] as! [Double]).map(Float.init)
        let mine = fist.refine(logits, landmarks: lm)
        // compare as probabilities (the unrefined path returns raw logits in both)
        func probs(_ v: [Float]) -> [Double] { let m = Double(v.max()!); let e = v.map { exp(Double($0) - m) }; let s = e.reduce(0, +); return e.map { $0 / s } }
        let diff = zip(probs(mine), probs(python)).map { abs($0 - $1) }.max()!
        if diff > worst { worst = diff; worstCase = ["index": index, "swift": probs(mine), "python": probs(python)] }
        argmaxSame += mine.indices.max(by: { mine[$0] < mine[$1] }) == python.indices.max(by: { python[$0] < python[$1] }) ? 1 : 0
        let f = LiveFistGeometry.features(lm)
        if let pf = c["features"] as? [Double], let f { featureWorst = max(featureWorst, zip(f, pf).map { abs($0 - $1) }.max()!); noneAgree += 1 }
        else if c["features"] is NSNull && f == nil { noneAgree += 1 }
    }
    try writeJSON(["cases": cases.count, "max_prob_diff": worst, "max_feature_diff": featureWorst,
                   "argmax_same": argmaxSame, "feature_presence_agree": noneAgree, "worst_case": worstCase], out)
}

let arguments = CommandLine.arguments
guard arguments.count >= 5 else {
    print("usage: harness fixture|crops|videos|stage3 <resources-dir> ...")
    exit(2)
}
LiveModelLocator.packageDirectory = URL(fileURLWithPath: arguments[2])
if let encoder = ProcessInfo.processInfo.environment["LIVE_REEL_ENCODER"] { LiveHandEncoder.packageName = encoder }
do {
    switch arguments[1] {
    case "fixture": try runFixture(arguments[3], arguments[4])
    case "crops": try runCrops(arguments[3], arguments[4], arguments[5])
    case "videos":
        let units: MLComputeUnits = arguments.count > 5 && arguments[5] == "all" ? .all : .cpuAndGPU
        try runVideos(arguments[3], arguments[4], recognizerUnits: units)
    case "stage3": try runStage3(arguments[3], arguments[4])
    case "fist": try runFist(arguments[3], arguments[4])
    default: print("unknown mode"); exit(2)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
