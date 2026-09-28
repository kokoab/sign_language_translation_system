// Front camera + LiveReelEngine, shared by the LIVE and PRACTICE pages.
// The capture session runs only while one of those pages shows; frames are processed only while
// `processing` is on (LIVE: after START; PRACTICE: during a round).

import AVFoundation
import UIKit

final class ReelCamera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    struct Update {
        let committed: [LiveWord]
        let preview: LivePreview?
        let pending: String?
        let observation: LiveObservation
        let body: [LivePoint]?
        let face: [LivePoint]?
        let fps: Double
        let frameMilliseconds: Double
        let frames: Int
    }

    var onUpdate: ((Update) -> Void)?       // main thread
    var onReady: (() -> Void)?              // main thread
    var onError: ((String) -> Void)?        // main thread
    private(set) var ready = false          // main thread

    let session = AVCaptureSession()
    let previewLayer: AVCaptureVideoPreviewLayer
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "slt.reel.camera", qos: .userInitiated)
    private let languageQueue = DispatchQueue(label: "slt.reel.language", qos: .userInitiated)
    private var rotation: AVCaptureDevice.RotationCoordinator?
    private var observers: [NSKeyValueObservation] = []

    // cameraQueue state.
    private var engine: LiveReelEngine?
    private var stage3: LiveStage3?
    private var processing = false
    private var firstStamp: CMTime?
    private var deadline = 0.0
    private var computeHistory: [Double] = []
    private var frameTimes: [Double] = []
    private var processed = 0
    private var utterance: [LiveWord] = []
    private var visibleBody: [LivePoint]?
    private var visibleFace: [LivePoint]?

    override init() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        super.init()
        previewLayer.videoGravity = .resizeAspect      // the whole frame the model sees
        configure()
    }

    // MARK: models

    func load() {
        queue.async {
            do {
                let engine = try LiveReelEngine()
                try engine.warm()
                self.engine = engine
                DispatchQueue.main.async { self.ready = true; self.onReady?() }
                self.languageQueue.async { self.stage3 = try? LiveStage3() }
            } catch {
                DispatchQueue.main.async { self.onError?("Model load failed: \(error.localizedDescription)") }
            }
        }
    }

    // MARK: capture

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480        // the recognizer's local recordings are 640x480
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let connection = output.connection(with: .video) {
            // Upright and unmirrored for the model; only the preview is mirrored.
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
        session.commitConfiguration()
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        rotation = coordinator
        applyRotation()
        observers = [
            coordinator.observe(\.videoRotationAngleForHorizonLevelCapture) { [weak self] _, _ in
                DispatchQueue.main.async { self?.applyRotation() }
            },
            coordinator.observe(\.videoRotationAngleForHorizonLevelPreview) { [weak self] _, _ in
                DispatchQueue.main.async { self?.applyRotation() }
            },
        ]
    }

    private func applyRotation() {
        guard let rotation else { return }
        if let connection = output.connection(with: .video),
           connection.isVideoRotationAngleSupported(rotation.videoRotationAngleForHorizonLevelCapture) {
            let angle = rotation.videoRotationAngleForHorizonLevelCapture
            queue.async {
                if connection.videoRotationAngle != angle {
                    connection.videoRotationAngle = angle
                    if self.processing { self.engine?.runtime.reset() }   // geometry changed
                }
            }
        }
        if let connection = previewLayer.connection,
           connection.isVideoRotationAngleSupported(rotation.videoRotationAngleForHorizonLevelPreview) {
            connection.videoRotationAngle = rotation.videoRotationAngleForHorizonLevelPreview
        }
    }

    func startCapture() { queue.async { if !self.session.isRunning { self.session.startRunning() } } }

    /// Stops processing and the camera; synchronous so another user of the camera can start after.
    func stopCapture() {
        queue.sync {
            processing = false
            if session.isRunning { session.stopRunning() }
        }
    }

    // MARK: processing

    func setProcessing(_ on: Bool) {
        queue.async {
            if on && !self.processing {
                self.engine?.reset()
                self.utterance = []
                self.firstStamp = nil
                self.deadline = 0
                self.frameTimes = []
                self.visibleBody = nil; self.visibleFace = nil
            }
            self.processing = on
        }
    }

    /// Clear the stream and anything not yet finished (RESET, or a judged practice attempt).
    func reset() {
        queue.async {
            self.engine?.reset()
            self.utterance = []
        }
    }

    /// Flush the open segment; `completion` (main) gets the tail and the whole utterance.
    func finish(_ completion: @escaping (_ tail: [LiveWord], _ utterance: [LiveWord]) -> Void) {
        queue.async {
            let tail = (try? self.engine?.finish()) ?? []
            self.utterance += tail
            let words = self.utterance
            self.utterance = []
            DispatchQueue.main.async { completion(tail, words) }
        }
    }

    /// Clause-split Stage 3 rendering (main-thread completion).
    func render(_ words: [LiveWord], _ completion: @escaping (_ sentence: String, _ clauses: [[String]], _ ms: Double) -> Void) {
        languageQueue.async {
            let started = Date()
            let rendered = self.stage3?.renderUtterance(words)
                ?? (sentence: words.map { $0.gloss.hasPrefix("fs-") ? LiveStage3.spelledText($0.gloss) : $0.gloss.lowercased() }
                        .joined(separator: " ") + ".", clauses: [[String]]())
            let ms = 1000 * Date().timeIntervalSince(started)
            DispatchQueue.main.async { completion(rendered.sentence, rendered.clauses, ms) }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard processing, let engine, let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let stamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if firstStamp == nil { firstStamp = stamp }
        let seconds = CMTimeGetSeconds(CMTimeSubtract(stamp, firstStamp!))
        guard seconds + 1e-6 >= deadline else { return }
        deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
        do {
            let step = try engine.process(frame, seconds: seconds)
            processed += 1
            computeHistory.append(step.computeSeconds)
            if computeHistory.count > 40 { computeHistory.removeFirst() }
            frameTimes.append(seconds)
            if frameTimes.count > 30 { frameTimes.removeFirst() }
            utterance += step.committed
            let o = step.observation
            if o.body.contains(where: { $0.c > 0 }) { visibleBody = o.body }
            if o.faceForFeatures && o.face.contains(where: { $0.c > 0 }) { visibleFace = o.face }
            let sorted = computeHistory.sorted()
            let update = Update(
                committed: step.committed, preview: engine.runtime.preview,
                pending: engine.speller.letters.isEmpty ? nil : engine.speller.pending,
                observation: o, body: visibleBody, face: visibleFace,
                fps: frameTimes.count > 1 ? Double(frameTimes.count - 1) / max(frameTimes.last! - frameTimes.first!, 1e-6) : 0,
                frameMilliseconds: 1000 * sorted[sorted.count / 2], frames: processed)
            DispatchQueue.main.async { self.onUpdate?(update) }
        } catch {
            DispatchQueue.main.async { self.onError?("Frame error: \(error.localizedDescription)") }
        }
    }

    var timing: [String: Double] { queue.sync { engine?.runtime.timing ?? [:] } }
}

/// Phone session history: one JSON per LIVE session in Documents/live_reel_sessions.
final class ShellSessionStore {
    struct Row {
        let started: Date
        let seconds: Double
        let said: String
        let hasSentence: Bool
        let complete: Bool
    }

    static let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("live_reel_sessions", isDirectory: true)

    private var started: Date?
    private var events: [[String: Any]] = []
    private var sentences: [String] = []
    private var glosses: [String] = []

    var active: Bool { started != nil }

    func begin() {
        started = Date(); events = []; sentences = []; glosses = []
    }

    func word(_ w: LiveWord) {
        guard active else { return }
        glosses.append(w.gloss)
        events.append(["event": "word", "word": w.json])
    }

    func sentence(_ text: String, clauses: [[String]], ms: Double) {
        guard active else { return }
        sentences.append(text)
        events.append(["event": "sentence", "sentence": text, "clauses": clauses, "ms": ms])
    }

    func note(_ event: String) {
        guard active else { return }
        events.append(["event": event, "time": Date().timeIntervalSince1970])
    }

    func end(complete: Bool, timing: [String: Double]) {
        guard let started else { return }
        self.started = nil
        guard !glosses.isEmpty || !sentences.isEmpty else { return }
        try? FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        let iso = ISO8601DateFormatter()
        let value: [String: Any] = [
            "format": "slt_v17_phone_live_session", "started_utc": iso.string(from: started),
            "finished_utc": iso.string(from: Date()), "duration_seconds": Date().timeIntervalSince(started),
            "sentences": sentences, "glosses": glosses, "complete": complete, "events": events,
            "timing": timing, "system": UIDevice.current.systemVersion,
        ]
        let name = iso.string(from: started).replacingOccurrences(of: ":", with: "-") + ".json"
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted]) {
            try? data.write(to: Self.directory.appendingPathComponent(name))
        }
    }

    static func rows() -> [Row] {
        let iso = ISO8601DateFormatter()
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }.compactMap { url -> Row? in
            guard let data = try? Data(contentsOf: url),
                  let d = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
            let started = (d["started_utc"] as? String).flatMap { iso.date(from: $0) } ?? Date.distantPast
            let sentences = (d["sentences"] as? [String] ?? []).filter { !$0.isEmpty }
            let glosses = d["glosses"] as? [String] ?? []
            let said = sentences.last ?? (glosses.isEmpty ? "No recognized signs" : glosses.joined(separator: " "))
            return Row(started: started, seconds: d["duration_seconds"] as? Double ?? 0, said: said,
                       hasSentence: !sentences.isEmpty, complete: d["complete"] as? Bool ?? false)
        }.sorted { $0.started > $1.started }
    }
}
