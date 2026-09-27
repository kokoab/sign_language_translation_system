// One live segmental Reel: Apple Vision -> dual segmental decoder -> spelling buffer.
// Shared by the iPhone screen (LiveReelViewController) and the macOS replay harness.

import Accelerate
import CoreML
import CoreVideo
import Foundation

final class LiveReelEngine {
    static let fps = 20.0
    static let detectionSide = 640       // Vision input side (args.detection_image_side)
    static let maximumSide = 1280        // processed frame side (args.maximum_image_side)

    let vision = LiveVision()
    let runtime: LiveDualRuntime
    let speller = LiveSpellingBuffer()
    let labels: [String]
    private var finishGesture = LiveFinishGesture()
    private var detectionBuffer: CVPixelBuffer?

    struct Step {
        let observation: LiveObservation
        let committed: [LiveWord]       // after the spelling buffer
        let computeSeconds: Double
        let finishRequested: Bool
        let finishProgress: Double?
    }

    init(recognizerUnits: MLComputeUnits = .cpuAndGPU, encoderUnits: MLComputeUnits = .all,
         boundaryUnits: MLComputeUnits = .all) throws {
        let labelData = try Data(contentsOf: LiveModelLocator.resource("live_reel_labels", "json"))
        labels = try JSONDecoder().decode([String].self, from: labelData)
        guard labels.count == 100 else { throw LiveReelError.model("Expected 100 labels") }
        let recognizer = try LiveSpanRecognizer(labels: labels, units: recognizerUnits)
        let encoder = try LiveHandEncoder(units: encoderUnits)
        let wordBoundary = try LiveBoundaryModel(name: "AVBoundaryStudentV17L6FP16", units: boundaryUnits)
        let letterBoundary = try LiveBoundaryModel(name: "AVBoundaryStudentV17L6LettersFP16", units: boundaryUnits)
        runtime = LiveDualRuntime(
            words: LiveSegmentalRuntime(boundary: wordBoundary, recognizer: recognizer, encoder: encoder, mode: .both),
            letters: LiveSegmentalRuntime(boundary: letterBoundary, recognizer: recognizer, encoder: encoder, mode: .letters))
    }

    /// First Core ML calls plan the graphs (slow); do them before the camera starts.
    func warm() throws {
        let recognizer = runtime.words.recognizer
        let zero = LiveSpanInput(landmarks: [Float](repeating: 0, count: 32 * 61 * 5),
                                 embeddings: [Float](repeating: 0, count: 16 * 3 * 512),
                                 valid: [Float](repeating: 0, count: 48), boxes: [Float](repeating: 0, count: 192))
        _ = try recognizer.logits([zero])
        _ = try runtime.words.encoder.embed(rgb: [UInt8](repeating: 0, count: 256 * 256 * 3))
        let rows = [[Float]](repeating: [Float](repeating: 0, count: LiveFeatures.boundaryDimension), count: 64)
        let mask = [Bool](repeating: true, count: 64)
        _ = try runtime.words.boundary.model.logits(window: rows, mask: mask)
        _ = try runtime.letters.boundary.model.logits(window: rows, mask: mask)
    }

    func resetFinishGesture() { finishGesture = LiveFinishGesture() }

    func reset() {
        resetFinishGesture()
        runtime.reset()
        _ = speller.flush()
        vision.resetTracking()
    }

    /// Process one upright, unmirrored BGRA frame already limited to `maximumSide`.
    func process(_ frame: CVPixelBuffer, seconds: Double, allowFinishGesture: Bool = false) throws -> Step {
        // Core ML outputs are autoreleased IOSurfaces; drain them every frame.
        try autoreleasepool { try processFrame(frame, seconds: seconds, allowFinishGesture: allowFinishGesture) }
    }

    private func processFrame(_ frame: CVPixelBuffer, seconds: Double, allowFinishGesture: Bool) throws -> Step {
        let started = Date()
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
        let detection = try detectionImage(frame)
        let observation = try vision.observe(detection: detection, width: width, height: height, seconds: seconds)
        if !allowFinishGesture { resetFinishGesture() }
        let finishRequested = allowFinishGesture && finishGesture.update(
            left: observation.left, right: observation.right, seconds: seconds)
        if finishGesture.active {
            // Preserve the pending sign for Finish; never classify the control pose.
            return Step(observation: observation, committed: [], computeSeconds: Date().timeIntervalSince(started),
                        finishRequested: finishRequested, finishProgress: finishGesture.progress)
        }
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        let image = LiveBGRAImage(base: CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self),
                                  width: width, height: height, bytesPerRow: CVPixelBufferGetBytesPerRow(frame))
        let words = try runtime.observe(observation, image: image)
        var shown = speller.tick(seconds)
        for w in words { shown += speller.push(w) }
        let spent = Date().timeIntervalSince(started)
        for w in shown { w.computeSeconds = spent }
        return Step(observation: observation, committed: shown, computeSeconds: spent,
                    finishRequested: false, finishProgress: nil)
    }

    /// Flush at Finish/Stop: the open segment and any spelled run are committed.
    func finish() throws -> [LiveWord] {
        try autoreleasepool { try finishStream() }
    }

    private func finishStream() throws -> [LiveWord] {
        var shown: [LiveWord] = []
        for w in try runtime.finish() { shown += speller.push(w) }
        shown += speller.flush()
        runtime.reset()
        return shown
    }

    /// limit_image_side(frame, 640) with area-like downscaling; the frame itself when small enough.
    private func detectionImage(_ frame: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
        let longest = max(width, height)
        guard longest > Self.detectionSide else { return frame }
        let scale = Double(Self.detectionSide) / Double(longest)
        let tw = max(1, Int((Double(width) * scale).rounded(.toNearestOrEven)))
        let th = max(1, Int((Double(height) * scale).rounded(.toNearestOrEven)))
        if detectionBuffer == nil || CVPixelBufferGetWidth(detectionBuffer!) != tw || CVPixelBufferGetHeight(detectionBuffer!) != th {
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, tw, th, kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
            detectionBuffer = buffer
        }
        guard let target = detectionBuffer else { throw LiveReelError.input("Could not allocate the detection image") }
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        CVPixelBufferLockBaseAddress(target, [])
        defer {
            CVPixelBufferUnlockBaseAddress(target, [])
            CVPixelBufferUnlockBaseAddress(frame, .readOnly)
        }
        var source = vImage_Buffer(data: CVPixelBufferGetBaseAddress(frame), height: vImagePixelCount(height),
                                   width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(frame))
        var destination = vImage_Buffer(data: CVPixelBufferGetBaseAddress(target), height: vImagePixelCount(th),
                                        width: vImagePixelCount(tw), rowBytes: CVPixelBufferGetBytesPerRow(target))
        let error = vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
        guard error == kvImageNoError else { throw LiveReelError.input("Detection resize failed (\(error))") }
        return target
    }
}
