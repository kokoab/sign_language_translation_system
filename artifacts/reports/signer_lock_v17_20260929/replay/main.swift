import AVFoundation
import Accelerate
import CoreImage
import Foundation
import Vision

// Signer-lock regression replay on macOS with the app's LiveVision (LiveReelCore.swift).
//   single <clips.json> <out.json>   lock off vs on on every frame; any difference is a regression
//   pair   <pairs.json> <out.json>   two clips side by side; which person each mode reads
let args = CommandLine.arguments
let fps = 20.0, detectionSide = 640
let ciContext = CIContext(options: [.useSoftwareRenderer: false])

func resize(_ frame: CVPixelBuffer) -> CVPixelBuffer {
    // Copy of LiveDetectionScaler.resize (LiveReelEngine.swift): longest side 640, high-quality vImage.
    let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
    let longest = max(width, height)
    guard longest > detectionSide else { return frame }
    let scale = Double(detectionSide) / Double(longest)
    let tw = max(1, Int((Double(width) * scale).rounded(.toNearestOrEven)))
    let th = max(1, Int((Double(height) * scale).rounded(.toNearestOrEven)))
    var target: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, tw, th, kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &target)
    CVPixelBufferLockBaseAddress(frame, .readOnly); CVPixelBufferLockBaseAddress(target!, [])
    var source = vImage_Buffer(data: CVPixelBufferGetBaseAddress(frame), height: vImagePixelCount(height),
                               width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(frame))
    var destination = vImage_Buffer(data: CVPixelBufferGetBaseAddress(target!), height: vImagePixelCount(th),
                                    width: vImagePixelCount(tw), rowBytes: CVPixelBufferGetBytesPerRow(target!))
    vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
    CVPixelBufferUnlockBaseAddress(target!, []); CVPixelBufferUnlockBaseAddress(frame, .readOnly)
    return target!
}

/// Upright BGRA frames at the app's 20 Hz schedule (process when seconds >= deadline).
func frames(_ path: String, _ body: (CVPixelBuffer, Double) throws -> Void) throws {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    guard let track = asset.tracks(withMediaType: .video).first else { return }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output); reader.startReading()
    let transform = track.preferredTransform
    var deadline = 0.0
    while let sample = output.copyNextSampleBuffer() {
        try autoreleasepool {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { return }
            let seconds = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
            guard seconds + 1e-6 >= deadline else { return }
            deadline = max(deadline + 1 / fps, seconds)
            if transform.isIdentity { try body(buffer, seconds); return }
            let image = CIImage(cvPixelBuffer: buffer).transformed(by: transform)
            let upright = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
            try body(render(upright), seconds)
        }
    }
}

func render(_ image: CIImage) -> CVPixelBuffer {
    var out: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, Int(image.extent.width), Int(image.extent.height), kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &out)
    ciContext.render(image, to: out!)
    return out!
}

func same(_ a: [LivePoint], _ b: [LivePoint]) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { $0.x == $1.x && $0.y == $1.y && $0.c == $1.c }
}
func same(_ a: LiveHand?, _ b: LiveHand?) -> Bool {
    switch (a, b) {
    case (nil, nil): return true
    case let (x?, y?): return same(x.points, y.points) && x.chirality == y.chirality
    default: return false
    }
}

let mode = args[1]
let items = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[2]))) as! [[String: Any]]
var rows: [[String: Any]] = []
var offSeconds = 0.0, onSeconds = 0.0, observed = 0

if mode == "single" {
    var totalFrames = 0, totalDiff = 0
    for item in items {
        let off = LiveVision(signerLock: false), on = LiveVision(signerLock: true)
        var n = 0, diffs: [[String: Any]] = []
        try frames(item["path"] as! String) { frame, seconds in
            let detection = resize(frame)
            let w = CVPixelBufferGetWidth(frame), h = CVPixelBufferGetHeight(frame)
            var t = Date()
            let a = try off.observe(detection: detection, width: w, height: h, seconds: seconds)
            offSeconds += Date().timeIntervalSince(t); t = Date()
            let b = try on.observe(detection: detection, width: w, height: h, seconds: seconds)
            onSeconds += Date().timeIntervalSince(t); observed += 1
            var fields: [String] = []
            if !same(a.left, b.left) { fields.append("left") }
            if !same(a.right, b.right) { fields.append("right") }
            if !same(a.body, b.body) { fields.append("body") }
            if !same(a.face, b.face) { fields.append("face") }
            if !fields.isEmpty { diffs.append(["frame": n, "seconds": seconds, "fields": fields]) }
            n += 1
        }
        totalFrames += n; totalDiff += diffs.count
        rows.append(["id": item["id"]!, "frames": n, "differing_frames": diffs.count, "diffs": Array(diffs.prefix(20))])
        print("\(item["id"]!) frames \(n) differing \(diffs.count)")
    }
    let out: [String: Any] = ["mode": mode, "clips": rows.count, "frames": totalFrames, "differing_frames": totalDiff,
                              "observe_ms_off": 1000 * offSeconds / Double(max(observed, 1)),
                              "observe_ms_on": 1000 * onSeconds / Double(max(observed, 1)), "rows": rows]
    try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: args[3]))
    print("SINGLE clips \(rows.count) frames \(totalFrames) differing \(totalDiff)  observe ms off \(out["observe_ms_off"]!) on \(out["observe_ms_on"]!)")
} else {
    // Signer (scale 1) and bystander (scale 0.8, as if farther away) side by side; both placements.
    var totals: [String: Int] = [:]
    for item in items {
        for signerLeft in [true, false] {
            let bystanderFrames = try { () -> [CIImage] in
                var list: [CIImage] = []
                try frames(item["bystander"] as! String) { f, _ in list.append(CIImage(cvPixelBuffer: f).copyForBystander()) }
                return list
            }()
            guard !bystanderFrames.isEmpty else { continue }
            let off = LiveVision(signerLock: false), on = LiveVision(signerLock: true)
            var n = 0
            var counts: [String: Int] = [:]
            var picks: [(String, String, Bool)] = []
            var lockedOnPlacedSigner: Bool?
            try frames(item["signer"] as! String) { frame, seconds in
                let signer = CIImage(cvPixelBuffer: frame)
                let other = bystanderFrames[min(n, bystanderFrames.count - 1)]
                let otherScaled = other.transformed(by: CGAffineTransform(scaleX: 0.8, y: 0.8))
                let width = signer.extent.width + otherScaled.extent.width
                let height = max(signer.extent.height, otherScaled.extent.height)
                let signerX = signerLeft ? 0 : otherScaled.extent.width
                let otherX = signerLeft ? signer.extent.width : 0
                let canvas = signer.transformed(by: CGAffineTransform(translationX: signerX, y: height - signer.extent.height))
                    .composited(over: otherScaled.transformed(by: CGAffineTransform(translationX: otherX, y: height - otherScaled.extent.height))
                    .composited(over: CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))))
                let composite = render(canvas)
                let lo = Float(signerX / width), hi = Float((signerX + signer.extent.width) / width)
                func isSigner(_ x: Float) -> Bool { x >= lo && x <= hi }
                let detection = resize(composite)
                let w = CVPixelBufferGetWidth(composite), h = CVPixelBufferGetHeight(composite)
                for (name, vision) in [("off", off), ("on", on)] {
                    let o = try vision.observe(detection: detection, width: w, height: h, seconds: seconds)
                    for hand in [o.left, o.right].compactMap({ $0 }) { picks.append((name, "hands", isSigner(LiveSigner.anchor(hand).x))) }
                    if let p = o.face.first(where: { $0.c > 0 }) {
                        picks.append((name, "faces", isSigner(p.x)))
                        if name == "on" && lockedOnPlacedSigner == nil { lockedOnPlacedSigner = isSigner(p.x) }
                    }
                    if let p = o.body.first(where: { $0.c > 0 }) { picks.append((name, "bodies", isSigner(p.x))) }
                }
                n += 1
            }
            // "Foreign" is judged against the person the lock acquired (the larger face), so both
            // modes are scored on staying with one person.
            let locked = lockedOnPlacedSigner ?? true
            for (name, kind, onPlaced) in picks {
                counts["\(name)_\(kind)", default: 0] += 1
                if onPlaced != locked { counts["\(name)_foreign_\(kind)", default: 0] += 1 }
            }
            counts["acquired_placed_signer"] = locked ? 1 : 0
            counts["frames"] = n
            for (k, v) in counts { totals[k, default: 0] += v }
            rows.append(["signer": item["signer"]!, "bystander": item["bystander"]!, "signer_left": signerLeft, "counts": counts])
            print("pair \(rows.count) signerLeft=\(signerLeft) \(counts.sorted { $0.key < $1.key })")
        }
    }
    let out: [String: Any] = ["mode": mode, "placements": rows.count, "totals": totals, "rows": rows]
    try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: args[3]))
    print("PAIR totals \(totals.sorted { $0.key < $1.key })")
}

extension CIImage {
    /// Bystander frames are held in memory; render once so the source buffers can be released.
    func copyForBystander() -> CIImage { CIImage(cvPixelBuffer: render(self)) }
}
