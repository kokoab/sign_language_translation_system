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
    /// Words only (no letter head / letter decoder): the desktop --no-fingerspelling mode.
    let wordsOnly: LiveSegmentalRuntime
    /// Word mode decodes with the letter classes as competitors (the dual runtime's word decoder) and
    /// hides the letters: they absorb transitions between signs that words-only turns into extra words.
    nonisolated(unsafe) static var wordModeLetterSink = true
    private var wordDecoder: LiveSegmentalRuntime { Self.wordModeLetterSink ? runtime.words : wordsOnly }
    /// Letters always on (words + letters, the manual override; Practice). Off: words by default, and
    /// the FINGERSPELL sign or NAME switches spelling on. Set on the engine queue; resets the stream.
    var lettersAlways = false {
        didSet { if oldValue != lettersAlways { reset() } }
    }
    /// The spell-mode switch; nil if its model is not bundled (then only NAME arms letters).
    let trigger = try? LiveFingerspellTrigger(url: LiveModelLocator.resource("fingerspell_trigger_v17", "json"))

    /// words: word decoder only. name: words + letters for a few seconds after NAME. spelling: letter
    /// decoder only, from FINGERSPELL until FINGERSPELL again or hands down. always: words + letters.
    enum SpellMode: String { case words, name, spelling, always }
    private(set) var spellMode: SpellMode = .words
    private var spellSince = 0.0                    // when the mode was entered
    private var lastHandSeconds = 0.0
    /// Recent frames with their encoded hands, replayed into words + letters when NAME arms them.
    private var backlog: [(LiveObservation, LiveHandFrame)] = []
    /// An early NAME waits until its segment closes (else the replay reads its tail as a second NAME).
    private var pendingName: LiveWord?
    /// Output starting before this is the switching sign itself: extended while the trigger stays up.
    private var suppressUntil = -Double.infinity
    private var signOpen = false
    private var lastWord: LiveWord?
    private var lastEmitted: LiveWord?
    static let pauseExitSeconds = 3.0, nameWindowSeconds = 4.0
    let speller = LiveSpellingBuffer()
    let labels: [String]
    private var finishGesture = LiveFinishGesture()
    private let detectionScaler = LiveDetectionScaler()

    struct Step {
        let observation: LiveObservation
        let committed: [LiveWord]       // after the spelling buffer
        let computeSeconds: Double
        let finishRequested: Bool
        let finishProgress: Double?
        var spellMode: SpellMode = .words
        /// "on:fingerspell", "on:name", "off:fingerspell", "off:pause", "off:name" when the mode changed.
        var spellEvent: String? = nil
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
        wordsOnly = LiveSegmentalRuntime(boundary: wordBoundary, recognizer: recognizer, encoder: encoder, mode: .words)
    }

    /// The still-open segment's best guess, from whichever decoder is active.
    var preview: LivePreview? {
        switch spellMode {
        case .always, .name: return runtime.preview
        case .spelling: return runtime.letters.preview
        case .words: return wordDecoder.preview.flatMap { $0.gloss.hasPrefix("FS_") ? nil : $0 }
        }
    }

    var timing: [String: Double] {
        spellMode == .words ? wordDecoder.timing : spellMode == .spelling ? runtime.letters.timing : runtime.timing
    }

    /// Restart recognition (after a rotation): the decoders only, not the spelling buffer.
    func resetStream() {
        runtime.reset()
        wordsOnly.reset()
        backlog = []
        trigger?.reset()
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
        wordsOnly.reset()
        _ = speller.flush()
        vision.resetTracking()
        resetSpellMode()
    }

    private func resetSpellMode() {
        spellMode = lettersAlways ? .always : .words
        spellSince = 0; lastHandSeconds = 0
        backlog = []; pendingName = nil; lastWord = nil; lastEmitted = nil
        suppressUntil = -.infinity; signOpen = false
        trigger?.reset()
    }

    /// Process one upright, unmirrored BGRA frame already limited to `maximumSide`.
    func process(_ frame: CVPixelBuffer, seconds: Double, allowFinishGesture: Bool = false,
                 onObservation: ((LiveObservation) -> Void)? = nil) throws -> Step {
        // Core ML outputs are autoreleased IOSurfaces; drain them every frame.
        try autoreleasepool { try processFrame(frame, seconds: seconds, allowFinishGesture: allowFinishGesture,
                                               onObservation: onObservation) }
    }

    private func processFrame(_ frame: CVPixelBuffer, seconds: Double, allowFinishGesture: Bool,
                              onObservation: ((LiveObservation) -> Void)?) throws -> Step {
        let started = Date()
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
        let detection = try detectionScaler.resize(frame)
        let observation = try vision.observe(detection: detection, width: width, height: height, seconds: seconds)
        // Display the measured pose before waiting for image encoding and recognition.
        // The decoder still receives this exact observation on its original clock.
        onObservation?(observation)
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
        let hands = observation.left != nil || observation.right != nil
        if hands { lastHandSeconds = seconds }
        var shown: [LiveWord] = []
        var event: String?
        if spellMode == .always {
            for w in try runtime.observe(observation, image: image) { shown += speller.push(w) }
        } else {
            let fired = trigger?.push(LiveFeatures.raw(observation), seconds: seconds)
            if let trigger, signOpen, fired == nil {
                if trigger.score > trigger.threshold { suppressUntil = seconds } else { signOpen = false; event = "sign_end" }
            }
            if let fired {
                signOpen = true; suppressUntil = seconds; pendingName = nil
                if spellMode == .spelling {
                    shown += try leaveSpelling(signStart: fired.start); event = "off:fingerspell"
                } else {
                    shown += try enterSpelling(signStart: fired.start, at: fired.end); event = "on:fingerspell"
                }
            }
            switch spellMode {
            case .words:
                let out = try wordDecoder.observe(observation, image: image).filter { !$0.gloss.hasPrefix("FS_") }
                if let hand = wordDecoder.hands.last {
                    backlog.append((observation, hand))
                    if backlog.count > 40 { backlog.removeFirst(backlog.count - 40) }
                }
                for w in out where w.startSeconds >= suppressUntil { shown += accept(w) }
                if let name = out.last(where: { $0.gloss == "NAME" && $0.startSeconds >= suppressUntil }) { pendingName = name }
                if let name = pendingName, !name.early || name.closed || seconds - name.commitSeconds > 1.5 {
                    pendingName = nil
                    shown += try enterName(after: name.endSeconds, at: seconds); event = "on:name"
                }
            case .name:
                // Leave only between signs: never while an early-shown word is still open.
                for w in try runtime.observe(observation, image: image) { shown += accept(w); lastWord = w }
            case .spelling:
                // Letters that began before the switch are the FINGERSPELL sign itself.
                for w in try runtime.letters.observe(observation, image: image)
                    where w.startSeconds >= max(spellSince, suppressUntil) {
                    shown += accept(w)
                }
            case .always:
                break
            }
        }
        shown += speller.tick(seconds, activeHands: hands)
        if spellMode == .spelling && seconds - lastHandSeconds >= Self.pauseExitSeconds {
            shown += try leaveSpelling(signStart: nil); event = "off:pause"
        } else if spellMode == .name && (shown.contains { $0.gloss.hasPrefix("fs-") }
                    || (seconds - spellSince > Self.nameWindowSeconds && speller.letters.isEmpty && runtime.preview == nil
                        && !(lastWord.map { $0.early && !$0.closed } ?? false))) {
            for w in try runtime.finish() { shown += accept(w) }
            shown += speller.flush()
            runtime.reset(); wordsOnly.reset(); backlog = []
            spellMode = .words; event = "off:name"
        }
        let spent = Date().timeIntervalSince(started)
        for w in shown { w.computeSeconds = spent }
        return Step(observation: observation, committed: shown, computeSeconds: spent,
                    finishRequested: false, finishProgress: nil, spellMode: spellMode, spellEvent: event)
    }

    /// A decoder switch restarts the word decoder, which then no longer knows the last word: keep the
    /// decoder's own rule (the same word again within duplicateGap, 1 s, is the same sign).
    private func accept(_ w: LiveWord) -> [LiveWord] {
        if !w.gloss.hasPrefix("FS_") {
            if let l = lastEmitted, l !== w, l.gloss == w.gloss, w.startSeconds >= l.startSeconds,
               w.startSeconds - l.endSeconds <= Double(runtime.words.config.duplicateGap) / Self.fps { return [] }
            lastEmitted = w
        }
        return speller.push(w)
    }

    /// FINGERSPELL seen in word (or NAME) mode: commit what came before the sign, then letters only.
    private func enterSpelling(signStart: Double, at seconds: Double) throws -> [LiveWord] {
        var shown: [LiveWord] = []
        let pending = try spellMode == .name ? runtime.finish()
            : wordDecoder.finish().filter { !$0.gloss.hasPrefix("FS_") }
        for w in pending where w.endSeconds <= signStart + 0.1 { shown += accept(w) }
        shown += speller.flush()
        runtime.reset(); wordsOnly.reset(); backlog = []
        spellMode = .spelling; spellSince = seconds
        return shown
    }

    /// FINGERSPELL again (signStart) or hands down (nil): commit the spelled word, back to words.
    private func leaveSpelling(signStart: Double?) throws -> [LiveWord] {
        let end = signStart ?? .infinity
        for w in try runtime.letters.finish() where w.startSeconds >= spellSince && w.endSeconds <= end + 0.1 {
            _ = accept(w)
        }
        if let signStart { speller.drop(from: signStart - 0.1) }
        let shown = speller.flush()
        runtime.reset(); wordsOnly.reset(); backlog = []
        spellMode = .words; spellSince = 0
        return shown
    }

    /// NAME committed: words + letters, replaying the frames since NAME so a name spelled straight
    /// after it is not cut.
    private func enterName(after nameEnd: Double, at seconds: Double) throws -> [LiveWord] {
        var shown: [LiveWord] = []
        for w in try wordDecoder.finish() where w.endSeconds <= nameEnd + 0.1 && !w.gloss.hasPrefix("FS_") {
            shown += accept(w)
        }
        wordsOnly.reset(); runtime.reset()
        let replay = backlog.filter { $0.0.seconds > nameEnd }
        backlog = []
        for (o, h) in replay { for w in try runtime.observeWithHand(o, hand: h) { shown += accept(w) } }
        spellMode = .name; spellSince = seconds; lastWord = nil
        return shown
    }

    /// Flush at Finish/Stop: the open segment and any spelled run are committed.
    func finish() throws -> [LiveWord] {
        try autoreleasepool { try finishStream() }
    }

    private func finishStream() throws -> [LiveWord] {
        var shown: [LiveWord] = []
        switch spellMode {
        case .always: for w in try runtime.finish() { shown += speller.push(w) }
        case .name: for w in try runtime.finish() { shown += accept(w) }
        case .spelling:
            for w in try runtime.letters.finish() where w.startSeconds >= max(spellSince, suppressUntil) { shown += accept(w) }
        case .words:
            for w in try wordDecoder.finish() where w.startSeconds >= suppressUntil && !w.gloss.hasPrefix("FS_") {
                shown += accept(w)
            }
        }
        shown += speller.flush()
        runtime.reset()
        wordsOnly.reset()
        resetSpellMode()
        return shown
    }

}

final class LiveDetectionScaler {
    private static let detectionSide = LiveReelEngine.detectionSide
    private var detectionBuffer: CVPixelBuffer?
    /// limit_image_side(frame, 640) with area-like downscaling; the frame itself when small enough.
    func resize(_ frame: CVPixelBuffer) throws -> CVPixelBuffer {
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
