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

    private var firstPTS: Double?
    /// Presentation time of the last frame returned by next(), relative to the first frame (variable frame rate).
    var lastPTS = 0.0
    func next() -> CVPixelBuffer? {
        guard let sample = output.copyNextSampleBuffer() else { return nil }
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
        if firstPTS == nil { firstPTS = pts }
        lastPTS = pts - firstPTS!
        return CMSampleBufferGetImageBuffer(sample)
    }
}

let replayResizeContext=CIContext()
func limited(_ buffer: CVPixelBuffer, _ side: Int) -> CVPixelBuffer {
    let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
    guard max(w, h) > side else { return buffer }
    let scale = Double(side) / Double(max(w, h))
    let tw = Int((Double(w) * scale).rounded()), th = Int((Double(h) * scale).rounded())
    var target: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, tw, th, kCVPixelFormatType_32BGRA, nil, &target)
    replayResizeContext.render(CIImage(cvPixelBuffer: buffer).transformed(by: CGAffineTransform(scaleX: CGFloat(tw) / CGFloat(w), y: CGFloat(th) / CGFloat(h))),
                       to: target!)
    return target!
}

func runVideos(_ path: String, _ out: String, recognizerUnits: MLComputeUnits) throws {
    let list = try readJSON(path) as! [[String: Any]]
    let engine = try LiveReelEngine(recognizerUnits: recognizerUnits)
    // LETTERS=always replays the manual letters-on mode; default is words + FINGERSPELL/NAME spell mode.
    engine.lettersAlways = ProcessInfo.processInfo.environment["LETTERS"] == "always"
    // WORD_SINK=0: word mode uses the words-only decoder (no letter classes).
    LiveReelEngine.wordModeLetterSink = ProcessInfo.processInfo.environment["WORD_SINK"] != "0"
    var records: [[String: Any]] = []
    var compute: [Double] = []
    var previousPath=""
    for item in list {
        let path=item["path"] as! String
        if path != previousPath { LiveVision.clearReplayDetections();previousPath=path }
        engine.reset()
        Experiment.pre = item["pre"] as? Double ?? 0.1
        Experiment.post = item["post"] as? Double ?? 0.1
        Experiment.events = []
        Experiment.dumped=[];Experiment.dumpKeys=[]
        Experiment.annotations=item["annotations"] as? [[String:Any]] ?? []
        Experiment.dumpPath=item["dump_path"] as! String
        FileManager.default.createFile(atPath:Experiment.dumpPath,contents:Data())
        LiveMyLikeMotion.events=[]; LiveMyLikeMotion.logged=[]
        engine.vision.setDiagnosticPhase(item["phase"] as! Int)
        let frames = try VideoFrames(URL(fileURLWithPath: item["path"] as! String))
        var index = 0
        var deadline = item["capture_offset"] as? Double ?? 0.0
        var words: [LiveWord] = []
        var modes: [[String: Any]] = []
        var held: [Double] = []
        engine.runtime.timing = [:]
        // LIVE_TIMES=<json [seconds]> + LIVE_OFFSET=<s>: decode exactly the frames the phone processed live
        // (video time = live time + offset) and give the engine the live timestamps.
        let liveTimes: [Double]? = ProcessInfo.processInfo.environment["LIVE_TIMES"].flatMap {
            (try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: $0)))) as? [Double] }
        let liveOffset = Double(ProcessInfo.processInfo.environment["LIVE_OFFSET"] ?? "0") ?? 0
        var next = 0
        while let frame = frames.next() {
            var seconds = frames.lastPTS
            if seconds > (item["limit"] as? Double ?? 10000) { break }
            index += 1
            if liveTimes != nil { seconds = frames.lastPTS }
            if let liveTimes {
                guard next < liveTimes.count else { continue }
                if seconds + 1e-3 < liveTimes[next] + liveOffset - 0.5 / frames.fps { continue }
                seconds = liveTimes[next]; next += 1
            } else {
                if seconds + 1e-6 < deadline { continue }
                deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
            }
            // FINISH=1 replays with the live app's finish-gesture detector on (frames it holds are not decoded).
            let step = try engine.process(limited(frame, LiveReelEngine.maximumSide), seconds: seconds,
                                          allowFinishGesture: ProcessInfo.processInfo.environment["FINISH"] == "1")
            if step.finishProgress != nil { held.append(seconds) }
            compute.append(step.computeSeconds)
            words += step.committed
            if let event = step.spellEvent { modes.append(["seconds": seconds, "event": event]) }
        }
        let tail = try engine.finish()
        for w in tail { w.computeSeconds = -1 }       // end-of-file flush, not a live latency
        words += tail
        records.append(["id": item["id"]!, "reference": item["reference"] ?? [], "hypothesis": words.map(\.gloss),
                        "dumped":Experiment.dumped, "score_events": Experiment.events, "motion_events": LiveMyLikeMotion.events, "words": words.map(\.json), "timing": engine.runtime.timing, "frames": index, "spell_events": modes,
                        "finish_held_seconds": held])
        print(item["id"]!, "finish-gesture held frames:", held.count, held.prefix(40).map { String(format: "%.2f", $0) }.joined(separator: " "))
        print(item["id"]!, words.map(\.gloss), modes.map { $0["event"]! })
        fflush(stdout)
        try writeJSON(["records": records], out)
    }
    let sorted = compute.sorted()
    try writeJSON(["records": records, "compute_ms": [
        "median": 1000 * sorted[sorted.count / 2], "p90": 1000 * sorted[Int(Double(sorted.count) * 0.9)],
        "max": 1000 * sorted.last!, "frames": sorted.count]], out)
}

LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"
let a=CommandLine.arguments
LiveModelLocator.packageDirectory=URL(fileURLWithPath:a[1])
try runVideos(a[2],a[3],recognizerUnits:.all)
