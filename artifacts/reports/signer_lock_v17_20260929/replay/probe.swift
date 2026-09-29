import AVFoundation
import Accelerate
import CoreImage
import Foundation
import Vision

// Vision's raw view of chosen frames: faces, bodies, and hands at maximumHandCount 2 and 4.
let args = CommandLine.arguments
let path = args[1]
let targets = Set(args[2].split(separator: ",").map { Int($0)! })
func describeHands(_ max: Int, _ image: CVPixelBuffer) throws -> String {
    let r = VNDetectHumanHandPoseRequest(); r.maximumHandCount = max
    try VNImageRequestHandler(cvPixelBuffer: image, orientation: .up).perform([r])
    return (r.results ?? []).map { o in
        let w = try? o.recognizedPoint(.wrist)
        return String(format: "[%@ conf %.2f wrist %.2f,%.2f]", o.chirality == .left ? "L" : o.chirality == .right ? "R" : "?", o.confidence, w?.location.x ?? -1, 1 - (w?.location.y ?? 2))
    }.joined(separator: " ")
}
var n = 0, deadline = 0.0
let asset = AVURLAsset(url: URL(fileURLWithPath: path))
let track = asset.tracks(withMediaType: .video).first!
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
reader.add(output); reader.startReading()
while let sample = output.copyNextSampleBuffer() {
    let seconds = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
    guard seconds + 1e-6 >= deadline else { continue }
    deadline = max(deadline + 1 / 20.0, seconds)
    defer { n += 1 }
    guard targets.contains(n), let image = CMSampleBufferGetImageBuffer(sample) else { continue }
    let f = VNDetectFaceLandmarksRequest(), b = VNDetectHumanBodyPoseRequest()
    try VNImageRequestHandler(cvPixelBuffer: image, orientation: .up).perform([f, b])
    let faces = (f.results ?? []).map { String(format: "[box %.2f,%.2f %.2fx%.2f conf %.2f]", $0.boundingBox.midX, 1 - $0.boundingBox.midY, $0.boundingBox.width, $0.boundingBox.height, $0.confidence) }
    print("frame \(n) t=\(String(format: "%.2f", seconds)) size \(CVPixelBufferGetWidth(image))x\(CVPixelBufferGetHeight(image))")
    print("  faces \(faces.joined(separator: " "))  bodies \((b.results ?? []).count)")
    print("  hands max2: \(try describeHands(2, image))")
    print("  hands max4: \(try describeHands(4, image))")
    let ci = CIImage(cvPixelBuffer: image)
    let ctx = CIContext()
    try ctx.writeJPEGRepresentation(of: ci, to: URL(fileURLWithPath: "frame_\(n).jpg"), colorSpace: CGColorSpaceCreateDeviceRGB())
}
