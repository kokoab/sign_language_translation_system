import AVFoundation
import CoreImage
import Foundation
import Vision
// Composite one pair (signer left, bystander right at 0.8) and print Vision's raw detections for some frames.
let args = CommandLine.arguments
func firstFrames(_ path: String, _ count: Int) -> [CIImage] {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let track = asset.tracks(withMediaType: .video).first!
    let reader = try! AVAssetReader(asset: asset)
    let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(out); reader.startReading()
    var list: [CIImage] = []
    let ctx = CIContext()
    while let s = out.copyNextSampleBuffer(), list.count < count {
        let img = CIImage(cvPixelBuffer: CMSampleBufferGetImageBuffer(s)!).transformed(by: track.preferredTransform)
        let up = img.transformed(by: CGAffineTransform(translationX: -img.extent.minX, y: -img.extent.minY))
        list.append(CIImage(cgImage: ctx.createCGImage(up, from: up.extent)!))
    }
    return list
}
let a = firstFrames(args[1], 40), b = firstFrames(args[2], 40)
let ctx = CIContext()
for i in stride(from: 0, to: min(a.count, b.count), by: 8) {
    let s = a[i], o = b[i].transformed(by: CGAffineTransform(scaleX: 0.8, y: 0.8))
    let W = s.extent.width + o.extent.width, H = max(s.extent.height, o.extent.height)
    let canvas = s.transformed(by: CGAffineTransform(translationX: 0, y: H - s.extent.height))
        .composited(over: o.transformed(by: CGAffineTransform(translationX: s.extent.width, y: H - o.extent.height)))
        .composited(over: CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: W, height: H)))
    let cg = ctx.createCGImage(canvas, from: canvas.extent)!
    let f = VNDetectFaceLandmarksRequest(), bo = VNDetectHumanBodyPoseRequest(), h = VNDetectHumanHandPoseRequest()
    h.maximumHandCount = 4
    try VNImageRequestHandler(cgImage: cg, orientation: .up).perform([f, bo, h])
    let split = Float(s.extent.width / W)
    print("frame \(i) canvas \(Int(W))x\(Int(H)) signer x<\(String(format: "%.2f", split))")
    print("  faces: " + (f.results ?? []).map { String(format: "(%.2f,%.2f w%.2f)", $0.boundingBox.midX, 1 - $0.boundingBox.midY, $0.boundingBox.width) }.joined(separator: " "))
    for body in bo.results ?? [] {
        let pts = (try? body.recognizedPoints(.all)) ?? [:]
        let nose = pts[.nose].flatMap { $0.confidence > 0.15 ? $0 : nil }, neck = pts[.neck].flatMap { $0.confidence > 0.15 ? $0 : nil }
        print("  body nose " + (nose.map { String(format: "%.2f,%.2f", $0.location.x, 1 - $0.location.y) } ?? "-") + " neck " + (neck.map { String(format: "%.2f,%.2f", $0.location.x, 1 - $0.location.y) } ?? "-"))
    }
    print("  hands: " + (h.results ?? []).map { o in let w = try? o.recognizedPoint(.wrist); return String(format: "(%.2f,%.2f)", w?.location.x ?? -1, 1 - (w?.location.y ?? 2)) }.joined(separator: " "))
    if i == 16 { try ctx.writeJPEGRepresentation(of: canvas, to: URL(fileURLWithPath: "composite_16.jpg"), colorSpace: CGColorSpaceCreateDeviceRGB()) }
}
