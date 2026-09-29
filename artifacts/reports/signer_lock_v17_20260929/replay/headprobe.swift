import AVFoundation
import Foundation
import Vision
let args = CommandLine.arguments
let targets = Set(args[2].split(separator: ",").map { Int($0)! })
let asset = AVURLAsset(url: URL(fileURLWithPath: args[1]))
let track = asset.tracks(withMediaType: .video).first!
let reader = try AVAssetReader(asset: asset)
let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
reader.add(out); reader.startReading()
var n = 0, deadline = 0.0
print("transform identity: \(track.preferredTransform.isIdentity)")
while let s = out.copyNextSampleBuffer() {
    let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(s))
    guard t + 1e-6 >= deadline else { continue }
    deadline = max(deadline + 0.05, t); defer { n += 1 }
    guard targets.contains(n), let img = CMSampleBufferGetImageBuffer(s) else { continue }
    let f = VNDetectFaceLandmarksRequest(), b = VNDetectHumanBodyPoseRequest()
    try VNImageRequestHandler(cvPixelBuffer: img, orientation: .up).perform([f, b])
    print("frame \(n) \(CVPixelBufferGetWidth(img))x\(CVPixelBufferGetHeight(img))")
    for face in f.results ?? [] { let r = face.boundingBox; print(String(format: "  face mid %.3f,%.3f  w %.3f h %.3f (vision y-up)", r.midX, r.midY, r.width, r.height)) }
    for body in b.results ?? [] {
        let p = try body.recognizedPoints(.all)
        for j in [VNHumanBodyPoseObservation.JointName.nose, .neck, .leftShoulder, .rightShoulder] {
            if let q = p[j] { print(String(format: "  %@ %.3f,%.3f conf %.2f", j.rawValue.rawValue, q.location.x, q.location.y, q.confidence)) }
        }
    }
}
