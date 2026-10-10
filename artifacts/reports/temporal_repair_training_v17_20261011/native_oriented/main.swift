import Foundation
import AVFoundation
import CoreImage
import CoreML
func readJSON(_ path:String)throws->Any{try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:path)))}
func writeJSON(_ value:Any,_ path:String)throws{try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]).write(to:URL(fileURLWithPath:path))}
/// Decoded upright frames at the source clock, as cv2.VideoCapture reads them.
final class VideoFrames {
    let reader: AVAssetReader
    let output: AVAssetReaderTrackOutput
    let fps: Double
    let rotate180: Bool
    let displayTransform: CGAffineTransform

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
        rotate180 = abs(transform.a+1)<1e-6 && abs(transform.d+1)<1e-6 && abs(transform.b)<1e-6 && abs(transform.c)<1e-6
        displayTransform=transform
        if !transform.isIdentity && !rotate180 { throw LiveReelError.input("Unsupported display transform") }
        reader = try AVAssetReader(asset: asset)
        output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()
        fps = Double(rate)
    }

    private var firstPTS: Double?
    /// Presentation time of the last frame returned by next(), relative to the first frame (variable frame rate).
    var lastPTS = 0.0
    func next() -> CVPixelBuffer? {
        guard let sample = output.copyNextSampleBuffer() else { return nil }
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
        if firstPTS == nil { firstPTS = pts }
        lastPTS = pts - firstPTS!
        guard let raw=CMSampleBufferGetImageBuffer(sample) else { return nil }
        let frame: CVPixelBuffer
        if rotate180 {
            let w=CVPixelBufferGetWidth(raw),h=CVPixelBufferGetHeight(raw)
            precondition(abs(displayTransform.tx-Double(w))<1e-3 && abs(displayTransform.ty-Double(h))<1e-3,"Unsupported rotated translation")
            var target: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault,w,h,kCVPixelFormatType_32BGRA,nil,&target)
            guard let out=target else { return nil }
            CVPixelBufferLockBaseAddress(raw,.readOnly);CVPixelBufferLockBaseAddress(out,[])
            let src=CVPixelBufferGetBaseAddress(raw)!.assumingMemoryBound(to:UInt32.self),dst=CVPixelBufferGetBaseAddress(out)!.assumingMemoryBound(to:UInt32.self)
            let si=CVPixelBufferGetBytesPerRow(raw)/4,di=CVPixelBufferGetBytesPerRow(out)/4
            for y in 0..<h { for x in 0..<w { dst[y*di+x]=src[(h-1-y)*si+(w-1-x)] } }
            CVPixelBufferUnlockBaseAddress(out,[]);CVPixelBufferUnlockBaseAddress(raw,.readOnly)
            frame=out
        } else { frame=raw }
        if lastPTS==0,let path=ProcessInfo.processInfo.environment["DUMP_FIRST_FRAME"] {
            CVPixelBufferLockBaseAddress(frame,.readOnly)
            let stride=CVPixelBufferGetBytesPerRow(frame),h=CVPixelBufferGetHeight(frame)
            try! Data(bytes:CVPixelBufferGetBaseAddress(frame)!,count:stride*h).write(to:URL(fileURLWithPath:path+".bin"))
            try! writeJSON(["width":CVPixelBufferGetWidth(frame),"height":h,"stride":stride],path+".json")
            CVPixelBufferUnlockBaseAddress(frame,.readOnly)
        }
        return frame
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


let a=CommandLine.arguments
LiveModelLocator.packageDirectory=URL(fileURLWithPath:a[1])
LiveHandEncoder.packageName="MobileCLIP2S0ImageEncoderV17FP16"
let encoder=try LiveHandEncoder(units:.all)
let cases=try readJSON(a[2]) as! [[String:Any]]
var rows:[[String:Any]]=[]
var arrays=[[Float]](repeating:[],count:4)
for item in cases {
 LiveVision.clearReplayDetections()
 for offset in [0.0,1.0/60,1.0/30] {
  for phase in 0..<8 {
   let vision=LiveVision();vision.setDiagnosticPhase(phase)
   let scaler=LiveDetectionScaler()
   let frames=try VideoFrames(URL(fileURLWithPath:item["path"] as! String))
   var deadline=offset
   var observations:[LiveObservation]=[],hands:[LiveHandFrame]=[]
   while let source=frames.next() {
    let seconds=frames.lastPTS
    if seconds > (item["limit"] as! Double) { break }
    if seconds+1e-6 < deadline {continue}
    deadline=max(deadline+0.05,seconds)
    try autoreleasepool {
     let sw=CVPixelBufferGetWidth(source),sh=CVPixelBufferGetHeight(source)
     let scale=min(1.0,1280.0/Double(max(sw,sh)))
     let w=Int((Double(sw)*scale).rounded()),h=Int((Double(sh)*scale).rounded())
     let needed=(item["events"] as! [[String:Any]]).contains { event in
      let core=event["core"] as! [Double]
      return seconds >= core[0]-0.2-1e-9 && seconds < core[1]+0.1-1e-9
     }
     let cached=vision.hasReplayDetection(width:w,height:h,seconds:seconds)
     let frame=(needed || !cached) ? limited(source,1280) : source
     let observation=try vision.observe(detection:cached ? frame : try scaler.resize(frame),width:w,height:h,seconds:seconds)
     if needed {
      CVPixelBufferLockBaseAddress(frame,.readOnly)
      defer {CVPixelBufferUnlockBaseAddress(frame,.readOnly)}
      let image=LiveBGRAImage(base:CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to:UInt8.self),width:w,height:h,bytesPerRow:CVPixelBufferGetBytesPerRow(frame))
      observations.append(observation);hands.append(try encoder.frameHand(observation,image:image))
     }
    }
   }
   for event in item["events"] as! [[String:Any]] {
    let core=event["core"] as! [Double]
    // Expand context at one edge; never trim away the verified semantic core.
    for (view,extra) in [0.0,0.05,0.10].enumerated() {
     let lo=core[0]-0.1-extra,hi=core[1]+0.1
     let ids=observations.indices.filter {lo-1e-9<=observations[$0].seconds && observations[$0].seconds<hi-1e-9}
     var row:[String:Any]=["id":event["id"]!,"phase":phase,"capture_offset":offset,"view":view,
       "core":core,"requested_interval":[lo,hi],"selected_seconds":ids.map{observations[$0].seconds},
       "frames":ids.count,"valid":false]
     let selected=ids.map{observations[$0]}
     row["core_hand_frames"]=ids.filter{ observations[$0].seconds>=core[0]-1e-9 && observations[$0].seconds<core[1]-1e-9 && (observations[$0].left != nil || observations[$0].right != nil) }.count
     if let trimmed=LiveSpanLandmarks.build(selected) {
      let retained=Array(ids[trimmed.trimStart..<trimmed.trimEnd])
      row["retained_seconds"]=retained.map{observations[$0].seconds}
      let length=retained.count
      let sample=(0..<16).map{retained[Int((Double($0)*Double(length-1)/15).rounded(.toNearestOrEven))]}
      row["core_crop_positions"]=sample.filter{observations[$0].seconds>=core[0]-1e-9 && observations[$0].seconds<core[1]-1e-9 && hands[$0].valid.contains(where:{$0>0.5})}.count
     }
     if ids.count>=4,let x=LiveSpanRecognizer.inputs(selected,ids.map{hands[$0]}) {
      row["valid"]=true;row["array_index"]=arrays[0].count/(32*61*5)
      arrays[0]+=x.landmarks;arrays[1]+=x.embeddings;arrays[2]+=x.valid;arrays[3]+=x.boxes
     }
     rows.append(row)
    }
   }
  }
 }
 print(item["id"]!,"extracted");fflush(stdout)
}
for (i,x) in arrays.enumerated() {
 try x.withUnsafeBufferPointer {Data(buffer:$0)}.write(to:URL(fileURLWithPath:a[3]+".\(i).bin"))
}
try writeJSON(["rows":rows,"classification_run":false,"encoding":"float32_little_endian","shapes":[[32,61,5],[16,3,512],[16,3],[16,3,4]]],a[3]+".json")
