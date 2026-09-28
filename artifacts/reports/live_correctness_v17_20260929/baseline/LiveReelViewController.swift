// LIVE and PRACTICE pages (landscape, full-screen camera) of the mobile app shell.
//
// Mirrors the desktop app shell (scripts/app_shell_v17.py, app_pages_v17.overlay_live, reel_hud_v17):
//   top          page nav (HOME LIVE GLOSSES PRACTICE HISTORY); another page closes this screen
//   below nav    state chip + FPS (left), committed-word rail with the amber "WORD?" preview (right)
//   under state  finished sentence (dot turns green while it is spoken)
//   centre       big current word (amber preview / white commit) with a confidence bar
//   bottom-left  live stats; bottom-right top-3 candidates
//   bottom       LIVE: RESET · START/STOP · FINISH      PRACTICE: SKIP THIS SIGN
// Practice: "SIGN THIS: X" banner, round/score chip, the reference clip ("HOW IT LOOKS"), a green
// "YEHEY!" flash on a match, "SAW X · TRY AGAIN" on a miss; every judged sign clears the buffer.
// LIVE processes nothing until START. PRACTICE runs as soon as the round opens (it was started).

import AVFoundation
import AudioToolbox
import Flutter
import UIKit

private enum ReelStyle {
    static let ink = UIColor(red: 14 / 255, green: 15 / 255, blue: 19 / 255, alpha: 1)
    static let white = UIColor(red: 245 / 255, green: 246 / 255, blue: 248 / 255, alpha: 1)
    static let muted = UIColor(red: 138 / 255, green: 143 / 255, blue: 152 / 255, alpha: 1)
    static let accent = UIColor(red: 61 / 255, green: 220 / 255, blue: 132 / 255, alpha: 1)
    static let amber = UIColor(red: 245 / 255, green: 165 / 255, blue: 36 / 255, alpha: 1)
    static let red = UIColor(red: 235 / 255, green: 77 / 255, blue: 75 / 255, alpha: 1)
    static let panelAlpha: CGFloat = 168 / 255
    static let chipAlpha: CGFloat = 150 / 255
    static let pages = ["HOME", "LIVE", "GLOSSES", "PRACTICE", "HISTORY"]
}

private typealias HUD = ReelStyle

private extension UIView {
    func shadowed(_ opacity: Float = 0.85, radius: CGFloat = 3) {
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = opacity
        layer.shadowRadius = radius
        layer.shadowOffset = CGSize(width: 0, height: 1)
    }
}

/// Rounded translucent pill with an optional leading dot (the desktop `_chip`).
private final class ReelChip: UIView {
    init(text: String, dotColor: UIColor?, textColor: UIColor, alpha: CGFloat = HUD.panelAlpha, size: CGFloat = 14) {
        super.init(frame: .zero)
        backgroundColor = HUD.ink.withAlphaComponent(alpha)
        layer.cornerRadius = 14
        let label = UILabel()
        label.text = text
        label.textColor = textColor
        label.font = .systemFont(ofSize: size, weight: .semibold)
        let dot = UIView()
        let stack = UIStackView(arrangedSubviews: dotColor == nil ? [label] : [dot, label])
        stack.spacing = 7
        stack.alignment = .center
        dot.backgroundColor = dotColor
        dot.layer.cornerRadius = 4
        dot.widthAnchor.constraint(equalToConstant: 8).isActive = true
        dot.heightAnchor.constraint(equalToConstant: 8).isActive = true
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 11),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -11),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }
}

final class LiveReelViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate,
    AVSpeechSynthesizerDelegate {
    enum Mode { case live, practice([String]) }

    /// Called once after the screen is dismissed with {"navigate": page, practice results...}.
    var onClose: (([String: Any]) -> Void)?

    private let mode: Mode
    private var isPractice: Bool { if case .practice = mode { return true } else { return false } }
    private var closeResult: [String: Any] = ["navigate": "HOME"]

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = LiveReelShared.queue
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var rotation: AVCaptureDevice.RotationCoordinator?
    private var rotationObservers: [NSKeyValueObservation] = []
    private let bones = CAShapeLayer()
    private let joints = CAShapeLayer()

    // Engine state: touched only on `queue`.
    private var running = false
    private var firstStamp: CMTime?
    private var deadline = 0.0
    private var computeHistory: [Double] = []
    private var frameTimes: [Double] = []
    private var processedFrames = 0
    private var utterance: [LiveWord] = []
    private var visibleBody: [LivePoint]?
    private var visibleFace: [LivePoint]?
    private var events: [[String: Any]] = []
    private var sessionGlosses: [String] = []
    private var sessionSentences: [String] = []
    private let openedAt = Date()

    // Display state: main thread.
    private var rail: [String] = []
    private var previewText: String?
    private var committedWord: (text: String, score: Double, at: Date)?
    private var centerPreview: (text: String, score: Double)?
    private var top3: [(gloss: String, score: Double)] = []
    private var finishGestureProgress: Double?
    private var finishing = false
    private var isRunningUI = false
    private var loaded = false
    private var fps = 0.0
    private var stateChipKey: String?
    private var railChipKey: [String]?
    private var candidateChipKey: [String]?
    private var roundChipKey: String?
    private var loadTimer: Timer?

    // Practice (main thread).
    private var targets: [String] = []
    private var practiceIndex = 0
    private var practiceScore = 0
    private var practiceAttempts = 0
    private var practiceMissed: [String] = []
    private var verdict: (kind: String, gloss: String, at: Date)?
    private var roundDone = false

    // Views.
    private let navStack = UIStackView()
    private let stateHost = UIView()
    private let roundHost = UIView()
    private let railStack = UIStackView()
    private let sentenceDot = UIView()
    private let sentenceLabel = UILabel()
    private let bigWord = UILabel()
    private let barTrack = UIView()
    private let barFill = UIView()
    private var barFillWidth: NSLayoutConstraint!
    private let scoreLabel = UILabel()
    private let statsLabel = UILabel()
    private let statsPanel = UIView()
    private let candidatesStack = UIStackView()
    private let controls = UIStackView()
    private let startButton = UIButton(type: .system)
    private let finishButton = UIButton(type: .system)
    private let resetButton = UIButton(type: .system)
    private let skipButton = UIButton(type: .system)
    private let banner = UILabel()
    private let bannerBox = UIView()
    private let pipView = UIView()
    private let pipLabel = UILabel()
    private var pipPlayer: AVQueuePlayer?
    private var pipLooper: AVPlayerLooper?
    private var pipLayer: AVPlayerLayer?
    private let flash = UIView()
    private let flashTitle = UILabel()
    private let flashGloss = UILabel()
    private let speech = AVSpeechSynthesizer()

    init(mode: Mode) {
        self.mode = mode
        if case .practice(let glosses) = mode { targets = glosses }
        super.init(nibName: nil, bundle: nil)
        modalPresentationStyle = .fullScreen
    }

    required init?(coder: NSCoder) { fatalError() }

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .landscape }
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation { .landscapeRight }
    override var prefersStatusBarHidden: Bool { true }
    override var prefersHomeIndicatorAutoHidden: Bool { true }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        speech.delegate = self
        configureCamera()
        buildInterface()
        render()
        LiveReelShared.warm()
        // The engine is usually warm already (built at launch); poll until it is.
        loadTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            let status = LiveReelShared.status
            if status["ready"] as? Bool == true {
                timer.invalidate()
                self.loaded = true
                self.queue.async {
                    LiveReelShared.engine?.reset()
                    if self.isPractice { self.startStream() }
                }
                if self.isPractice { self.isRunningUI = true }
                self.render()
            } else if status["state"] as? String == "failed" {
                timer.invalidate()
                self.sentenceLabel.text = "Model load failed: \(status["error"] as? String ?? "")"
                self.render()
            }
        }
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        UIApplication.shared.isIdleTimerDisabled = true
        setNeedsUpdateOfSupportedInterfaceOrientations()
        view.window?.windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: .landscape))
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        UIApplication.shared.isIdleTimerDisabled = false
        loadTimer?.invalidate()
        pipPlayer?.pause()
        // Synchronous so the camera is free before Flutter shows its next page.
        queue.sync {
            running = false
            session.stopRunning()
            LiveReelShared.engine?.reset()
            saveSession()
        }
        speech.stopSpeaking(at: .immediate)
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        if isBeingDismissed || presentingViewController == nil {
            let close = onClose
            onClose = nil
            close?(closeResult)
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        bones.frame = view.bounds
        joints.frame = view.bounds
        pipLayer?.frame = pipView.bounds
    }

    // MARK: - Camera

    private func configureCamera() {
        session.beginConfiguration()
        // The desktop live page captures 1280x720; the engine processes frames up to 1280 wide.
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .vga640x480
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
            // The model contract is upright and unmirrored; only the on-screen preview is mirrored.
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
        session.commitConfiguration()
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill      // full screen, no letterbox
        view.layer.insertSublayer(previewLayer, at: 0)
        bones.strokeColor = HUD.accent.withAlphaComponent(0.9).cgColor
        bones.fillColor = UIColor.clear.cgColor
        bones.lineWidth = 2
        bones.lineCap = .round
        joints.fillColor = HUD.white.cgColor
        view.layer.insertSublayer(bones, above: previewLayer)
        view.layer.insertSublayer(joints, above: bones)
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        rotation = coordinator
        applyRotation()
        rotationObservers = [
            coordinator.observe(\.videoRotationAngleForHorizonLevelCapture) { [weak self] _, _ in
                DispatchQueue.main.async { self?.applyRotation() }
            },
            coordinator.observe(\.videoRotationAngleForHorizonLevelPreview) { [weak self] _, _ in
                DispatchQueue.main.async { self?.applyRotation() }
            },
        ]
        queue.async { self.session.startRunning() }
    }

    private func applyRotation() {
        guard let rotation else { return }
        if let connection = output.connection(with: .video),
           connection.isVideoRotationAngleSupported(rotation.videoRotationAngleForHorizonLevelCapture) {
            let angle = rotation.videoRotationAngleForHorizonLevelCapture
            queue.async {
                if connection.videoRotationAngle != angle {
                    connection.videoRotationAngle = angle
                    // Frame geometry changed: start the stream fresh.
                    if self.running {
                        LiveReelShared.engine?.runtime.reset()
                        LiveReelShared.engine?.resetFinishGesture()
                    }
                }
            }
        }
        if let connection = previewLayer.connection,
           connection.isVideoRotationAngleSupported(rotation.videoRotationAngleForHorizonLevelPreview) {
            connection.videoRotationAngle = rotation.videoRotationAngleForHorizonLevelPreview
        }
    }

    /// On `queue`.
    private func startStream() {
        LiveReelShared.engine?.reset()
        firstStamp = nil
        deadline = 0
        frameTimes = []
        visibleBody = nil; visibleFace = nil
        running = true
        events.append(["event": "start", "time": Date().timeIntervalSince1970])
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard running, let engine = LiveReelShared.engine, let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let stamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if firstStamp == nil { firstStamp = stamp }
        let seconds = CMTimeGetSeconds(CMTimeSubtract(stamp, firstStamp!))
        guard seconds + 1e-6 >= deadline else { return }
        deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
        do {
            let step = try engine.process(frame, seconds: seconds, allowFinishGesture: !isPractice) { [weak self] observation in
                self?.publishObservation(observation)
            }
            processedFrames += 1
            computeHistory.append(step.computeSeconds)
            if computeHistory.count > 40 { computeHistory.removeFirst() }
            frameTimes.append(seconds)
            if frameTimes.count > 30 { frameTimes.removeFirst() }
            if !isPractice { utterance += step.committed }
            for w in step.committed {
                events.append(["event": "word", "seconds": seconds, "word": w.json])
                if !isPractice { sessionGlosses.append(w.gloss) }
            }
            let o = step.observation
            let preview = step.finishProgress == nil ? engine.runtime.preview : nil
            let pending = step.finishProgress != nil || engine.speller.letters.isEmpty ? nil : engine.speller.pending
            let sorted = computeHistory.sorted()
            let median = 1000 * sorted[sorted.count / 2]
            let fps = frameTimes.count > 1 ? Double(frameTimes.count - 1) / max(frameTimes.last! - frameTimes.first!, 1e-6) : 0
            let hands = [o.left, o.right].compactMap { $0 }.count
            let frames = processedFrames
            DispatchQueue.main.async {
                guard self.isRunningUI else { return }
                self.fps = fps
                self.finishGestureProgress = step.finishProgress
                self.apply(committed: step.committed, preview: preview, pending: pending)
                if self.isPractice { self.judge(step.committed) }
                self.statsLabel.attributedText = self.stats([
                    ("FPS", String(format: "%.1f", fps)), ("FRAME", String(format: "%.0f ms", median)),
                    ("HANDS", "\(hands)"), ("FRAMES", "\(frames)"),
                ])
                self.render()
            }
            if step.finishRequested { finishUtterance(reason: "two_hand_gesture") }
        } catch {
            DispatchQueue.main.async { self.sentenceLabel.text = "Frame error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Controls

    @objc private func toggleStart() {
        guard loaded, !isPractice else { return }
        let starting = !isRunningUI
        isRunningUI = starting
        render()
        queue.async {
            if starting {
                self.startStream()
            } else {
                self.running = false
                self.finishUtterance(reason: "stop")
                DispatchQueue.main.async {
                    self.bones.path = nil; self.joints.path = nil
                    self.fps = 0
                    self.render()
                }
            }
        }
    }

    @objc private func finishPressed() {
        guard loaded, !isPractice else { return }
        queue.async { self.finishUtterance(reason: "finish") }
    }

    @objc private func resetPressed() {
        queue.async {
            LiveReelShared.engine?.reset()
            self.utterance = []
            self.events.append(["event": "reset"])
            DispatchQueue.main.async {
                self.speech.stopSpeaking(at: .immediate)
                self.rail = []; self.previewText = nil; self.committedWord = nil
                self.centerPreview = nil; self.top3 = []; self.finishing = false
                self.finishGestureProgress = nil
                self.sentenceLabel.text = nil
                self.render()
            }
        }
    }

    @objc private func navPressed(_ sender: UIButton) {
        let page = HUD.pages[sender.tag]
        if page == (isPractice ? "PRACTICE" : "LIVE") { return }
        close(navigate: page)
    }

    private func close(navigate page: String) {
        guard presentingViewController != nil, !isBeingDismissed else { return }
        closeResult = ["navigate": page]
        if isPractice {
            closeResult.merge([
                "score": practiceScore, "attempts": practiceAttempts, "total": targets.count,
                "missed": practiceMissed, "completed": roundDone,
            ]) { $1 }
        }
        dismiss(animated: true)
    }

    /// On `queue`: flush the open segment, then render and speak the utterance.
    private func finishUtterance(reason: String) {
        guard let engine = LiveReelShared.engine else { return }
        let tail = (try? engine.finish()) ?? []
        utterance += tail
        sessionGlosses += tail.map(\.gloss)
        let words = utterance
        utterance = []
        events.append(["event": reason, "words": words.map(\.json)])
        DispatchQueue.main.async {
            self.apply(committed: tail, preview: nil, pending: nil)
            self.previewText = nil; self.centerPreview = nil
            if words.isEmpty {
                self.sentenceLabel.text = "No recognized signs to finish."
            } else {
                self.finishing = true
                self.sentenceLabel.text = "Finishing..."
            }
            self.render()
        }
        guard !words.isEmpty else { return }
        LiveReelShared.languageQueue.async {
            let started = Date()
            let rendered = LiveReelShared.stage3?.renderUtterance(words)
                ?? (sentence: words.map { $0.gloss.hasPrefix("fs-") ? LiveStage3.spelledText($0.gloss) : $0.gloss.lowercased() }
                        .joined(separator: " ") + ".", clauses: [[String]]())
            let ms = 1000 * Date().timeIntervalSince(started)
            self.queue.async {
                self.events.append(["event": "sentence", "sentence": rendered.sentence, "clauses": rendered.clauses, "ms": ms])
                self.sessionSentences.append(rendered.sentence)
            }
            DispatchQueue.main.async {
                self.finishing = false
                self.sentenceLabel.text = rendered.sentence
                self.rail = []
                self.say(rendered.sentence)
                self.render()
            }
        }
    }

    // MARK: - Practice (AppShell._check_practice / skip_practice)

    private var target: String? { !roundDone && practiceIndex < targets.count ? targets[practiceIndex] : nil }

    private func judge(_ committed: [LiveWord]) {
        guard let target, let first = committed.first else { return }
        let seen = Self.display(first.gloss)
        practiceAttempts += 1
        if seen == target {
            practiceScore += 1
            verdict = ("matched", seen, Date())
            celebrate(seen)
            AudioServicesPlaySystemSound(1057)
            advance()
        } else {
            verdict = ("missed", seen, Date())
        }
        clearBuffer()      // judged either way: the next attempt starts from nothing
    }

    @objc private func skipPressed() {
        guard let target else { return }
        practiceAttempts += 1
        practiceMissed.append(target)
        verdict = nil
        advance()
        clearBuffer()
        render()
    }

    private func advance() {
        practiceIndex += 1
        if practiceIndex >= targets.count {
            roundDone = true
            // Let the last flash play, then hand the score to the result page.
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.close(navigate: "PRACTICE") }
            return
        }
        loadReference()
    }

    private func clearBuffer() {
        rail = []; previewText = nil; centerPreview = nil; top3 = []
        queue.async { LiveReelShared.engine?.reset() }
    }

    private func celebrate(_ gloss: String) {
        flashGloss.text = "✓  " + gloss
        flash.layer.removeAllAnimations()
        flash.alpha = 1
        UIView.animate(withDuration: 1.4, delay: 0, options: [.curveEaseIn]) { self.flash.alpha = 0 }
    }

    /// The target's reference clip, looping (the desktop "HOW IT LOOKS" picture-in-picture).
    private func loadReference() {
        pipPlayer?.pause()
        pipLayer?.removeFromSuperlayer()
        pipPlayer = nil; pipLooper = nil; pipLayer = nil
        guard let target else { pipView.isHidden = true; return }
        let key = FlutterDartProject.lookupKey(forAsset: "assets/gloss_examples/\(target).mp4")
        guard let path = Bundle.main.path(forResource: key, ofType: nil) else { pipView.isHidden = true; return }
        let player = AVQueuePlayer()
        player.isMuted = true
        pipLooper = AVPlayerLooper(player: player, templateItem: AVPlayerItem(url: URL(fileURLWithPath: path)))
        let layer = AVPlayerLayer(player: player)
        layer.videoGravity = .resizeAspectFill
        layer.frame = pipView.bounds
        pipView.layer.insertSublayer(layer, at: 0)
        pipPlayer = player; pipLayer = layer
        pipView.isHidden = false
        player.play()
    }

    // MARK: - Display state

    private static func display(_ gloss: String) -> String {
        if gloss.hasPrefix("fs-") { return LiveStage3.spelledText(gloss).uppercased() }
        if gloss.hasPrefix("FS_") { return String(gloss.dropFirst(3)) }
        return gloss
    }

    /// show_preview / emit from the desktop runner.
    private func apply(committed: [LiveWord], preview: LivePreview?, pending: String?) {
        if !committed.isEmpty, !finishing, sentenceLabel.text != nil {
            sentenceLabel.text = nil        // a new utterance has started
        }
        for w in committed {
            rail.append(Self.display(w.gloss))
            committedWord = (Self.display(w.gloss), w.score, Date())
            say(w.gloss.hasPrefix("fs-") ? LiveStage3.spelledText(w.gloss) : w.gloss.lowercased())
        }
        if rail.count > 24 { rail.removeFirst(rail.count - 24) }
        if let pending {
            previewText = pending + "…"
            centerPreview = (pending + "…", 1)
        } else if !committed.isEmpty {
            previewText = nil
            centerPreview = nil
        } else if let preview, preview.emit {
            previewText = Self.display(preview.gloss) + "?"
            centerPreview = (Self.display(preview.gloss) + "?", preview.score)
            top3 = preview.top3.map { (Self.display($0.gloss), $0.score) }
        } else {
            previewText = nil
            centerPreview = nil
        }
    }

    /// On the engine queue, before the expensive recognition stages.
    private func publishObservation(_ o: LiveObservation) {
        if o.body.contains(where: { $0.c > 0 }) { visibleBody = o.body }
        if o.faceForFeatures && o.face.contains(where: { $0.c > 0 }) { visibleFace = o.face }
        let body = visibleBody, face = visibleFace
        DispatchQueue.main.async { [weak self] in
            guard let self, self.isRunningUI else { return }
            self.drawSkeleton(o, body: body, face: face)
        }
    }

    private func say(_ text: String) {
        speech.speak(AVSpeechUtterance(string: text))
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        sentenceDot.backgroundColor = HUD.accent
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        if !synthesizer.isSpeaking { sentenceDot.backgroundColor = HUD.white.withAlphaComponent(0.35) }
    }

    private func stats(_ rows: [(String, String)]) -> NSAttributedString {
        let text = NSMutableAttributedString()
        for (i, (label, value)) in rows.enumerated() {
            text.append(NSAttributedString(string: label.padding(toLength: 8, withPad: " ", startingAt: 0), attributes: [
                .font: UIFont.monospacedSystemFont(ofSize: 11, weight: .semibold), .foregroundColor: HUD.muted]))
            text.append(NSAttributedString(string: value + (i < rows.count - 1 ? "\n" : ""), attributes: [
                .font: UIFont.monospacedSystemFont(ofSize: 13, weight: .medium), .foregroundColor: HUD.white]))
        }
        return text
    }

    private func host(_ view: UIView, in hostView: UIView) {
        hostView.subviews.forEach { $0.removeFromSuperview() }
        view.translatesAutoresizingMaskIntoConstraints = false
        hostView.addSubview(view)
        NSLayoutConstraint.activate([
            view.leadingAnchor.constraint(equalTo: hostView.leadingAnchor),
            view.topAnchor.constraint(equalTo: hostView.topAnchor),
            view.bottomAnchor.constraint(equalTo: hostView.bottomAnchor),
            view.trailingAnchor.constraint(equalTo: hostView.trailingAnchor),
        ])
    }

    /// Redraw every overlay from the display state.
    private func render() {
        // State chip.
        let hasHands = bones.path.map { !$0.isEmpty } ?? false
        let (state, color): (String, UIColor) =
            !loaded ? ("WARMING UP", HUD.muted) :
            finishing ? ("FINISHING", HUD.amber) :
            !isRunningUI ? ("READY", HUD.muted) :
            finishGestureProgress != nil ? (finishGestureProgress! < 1
                ? "HOLD TO FINISH \(Int(finishGestureProgress! * 100))%" : "LOWER HANDS", HUD.amber) :
            centerPreview != nil ? ("PREVIEW", HUD.amber) :
            hasHands ? ("SIGNING", HUD.accent) : ("READY", HUD.muted)
        let stateText = String(format: "%@   %.1f FPS", state, fps)
        if stateChipKey != stateText {
            stateChipKey = stateText
            host(ReelChip(text: stateText, dotColor: color, textColor: HUD.white), in: stateHost)
        }

        // Word rail (live only): newest at the right, preview (amber) after it.
        let railKey = Array(rail.suffix(7)) + ["\u{0}", previewText ?? "\u{1}"]
        if !isPractice && railChipKey != railKey {
            railChipKey = railKey
            railStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
            let recent = Array(rail.suffix(7))
            var items = recent.enumerated().map { index, word -> ReelChip in
                let newest = index == recent.count - 1 && previewText == nil
                return ReelChip(text: word, dotColor: newest ? HUD.accent : HUD.muted,
                                textColor: newest ? HUD.white : HUD.muted,
                                alpha: newest ? HUD.panelAlpha : HUD.chipAlpha)
            }
            if let previewText {
                items.append(ReelChip(text: previewText, dotColor: HUD.amber, textColor: HUD.amber))
            }
            items.forEach { railStack.addArrangedSubview($0) }
        }

        // Centre word.
        var centerAlpha: CGFloat = 0
        if let p = centerPreview {
            bigWord.text = p.text
            bigWord.textColor = HUD.amber
            barFill.backgroundColor = HUD.amber
            scoreLabel.text = String(format: "%.2f", p.score)
            barFillWidth.constant = 220 * CGFloat(max(0, min(1, p.score)))
            centerAlpha = 1
        } else if let c = committedWord {
            let age = Date().timeIntervalSince(c.at)
            bigWord.text = c.text
            bigWord.textColor = HUD.white
            barFill.backgroundColor = HUD.accent
            scoreLabel.text = String(format: "%.2f", c.score)
            barFillWidth.constant = 220 * CGFloat(max(0, min(1, c.score)))
            centerAlpha = age < 1.6 ? 1 : max(0.38, 1 - CGFloat(age - 1.6) * 0.43)
        }
        for v in [bigWord, barTrack, scoreLabel] as [UIView] { v.alpha = centerAlpha }

        // Candidates.
        let candidateKey = top3.flatMap { [$0.gloss, String($0.score.bitPattern)] }
        if candidateChipKey != candidateKey {
            candidateChipKey = candidateKey
            candidatesStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
            for (i, c) in top3.enumerated() {
                candidatesStack.addArrangedSubview(ReelChip(
                    text: String(format: "%@  %.2f", c.gloss, c.score),
                    dotColor: HUD.accent.withAlphaComponent(0.45 + 0.55 * CGFloat(min(1, max(0, c.score)))),
                    textColor: i == 0 ? HUD.white : HUD.muted, alpha: i == 0 ? HUD.panelAlpha : HUD.chipAlpha, size: 13))
            }
        }
        candidatesStack.isHidden = !isRunningUI || top3.isEmpty

        if isPractice {
            // Round furniture (app_pages_v17.overlay_live).
            if let v = verdict, Date().timeIntervalSince(v.at) > 3 { verdict = nil }
            if let target {
                let missed = verdict?.kind == "missed"
                banner.text = missed ? "SAW  \(verdict!.gloss)  ·  TRY AGAIN" : "SIGN THIS:  \(target)"
                banner.textColor = missed ? HUD.amber : HUD.white
                bannerBox.layer.borderColor = (missed ? HUD.amber : HUD.white).cgColor
                bannerBox.isHidden = false
            } else {
                bannerBox.isHidden = true
            }
            let roundText = "ROUND \(min(practiceIndex + 1, targets.count)) OF \(targets.count)   ·   SCORE \(practiceScore)/\(practiceAttempts)"
            if roundChipKey != roundText {
                roundChipKey = roundText
                host(ReelChip(text: roundText, dotColor: nil, textColor: HUD.white, size: 13), in: roundHost)
            }
            skipButton.isEnabled = target != nil
            skipButton.alpha = target != nil ? 1 : 0.4
        } else {
            startButton.setTitle(isRunningUI ? "STOP" : "START", for: .normal)
            startButton.backgroundColor = (isRunningUI ? HUD.red : HUD.accent).withAlphaComponent(loaded ? 0.92 : 0.35)
            startButton.setTitleColor(isRunningUI ? HUD.white : HUD.ink, for: .normal)
            finishButton.setTitle(finishing ? "WORKING" : "FINISH", for: .normal)
            finishButton.backgroundColor = (finishing ? HUD.amber : HUD.accent).withAlphaComponent(0.92)
            sentenceDot.isHidden = sentenceLabel.text == nil
            sentenceLabel.textColor = finishing ? HUD.muted : HUD.white
        }
    }

    /// Hands, body and face on the mirrored, aspect-filled preview; body/face stay until re-detected.
    private func drawSkeleton(_ o: LiveObservation, body: [LivePoint]?, face: [LivePoint]?) {
        let bounds = view.bounds
        let w = CGFloat(o.width), h = CGFloat(o.height)
        let scale = max(bounds.width / w, bounds.height / h)
        let ox = (bounds.width - w * scale) / 2, oy = (bounds.height - h * scale) / 2
        func point(_ p: LivePoint) -> CGPoint {
            CGPoint(x: ox + (1 - CGFloat(p.x)) * w * scale, y: oy + CGFloat(p.y) * h * scale)
        }
        let lines = UIBezierPath(), dots = UIBezierPath()
        let links = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                     (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
        for hand in [o.left, o.right].compactMap({ $0 }) {
            for (a, b) in links where hand.points[a].c > 0 && hand.points[b].c > 0 {
                lines.move(to: point(hand.points[a])); lines.addLine(to: point(hand.points[b]))
            }
            for p in hand.points where p.c > 0 {
                dots.append(UIBezierPath(arcCenter: point(p), radius: 2.5, startAngle: 0, endAngle: 2 * .pi, clockwise: true))
            }
        }
        if let body {
            for (a, b) in [(0, 1), (0, 2), (1, 3)] where body[a].c > 0 && body[b].c > 0 {
                lines.move(to: point(body[a])); lines.addLine(to: point(body[b]))
            }
            for p in body where p.c > 0 {
                dots.append(UIBezierPath(arcCenter: point(p), radius: 3.5, startAngle: 0, endAngle: 2 * .pi, clockwise: true))
            }
        }
        for p in (face ?? []) where p.c > 0 {
            dots.append(UIBezierPath(arcCenter: point(p), radius: 2, startAngle: 0, endAngle: 2 * .pi, clockwise: true))
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        bones.path = lines.cgPath
        joints.path = dots.cgPath
        CATransaction.commit()
    }

    // MARK: - Layout

    private func pill(_ button: UIButton, _ title: String, _ action: Selector, width: CGFloat, primary: Bool) {
        button.setTitle(title, for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 16, weight: .bold)
        button.layer.cornerRadius = 23
        button.addTarget(self, action: action, for: .touchUpInside)
        button.translatesAutoresizingMaskIntoConstraints = false
        button.widthAnchor.constraint(equalToConstant: width).isActive = true
        button.heightAnchor.constraint(equalToConstant: 46).isActive = true
        if primary {
            button.setTitleColor(HUD.ink, for: .normal)
            button.backgroundColor = HUD.accent.withAlphaComponent(0.92)
        } else {
            button.setTitleColor(HUD.white, for: .normal)
            button.backgroundColor = HUD.ink.withAlphaComponent(HUD.panelAlpha)
            button.layer.borderColor = HUD.white.withAlphaComponent(0.35).cgColor
            button.layer.borderWidth = 1
        }
    }

    private func buildInterface() {
        let guide = view.safeAreaLayoutGuide
        let active = isPractice ? "PRACTICE" : "LIVE"

        // Page nav (draw_nav): the active page is a filled pill; the others are shadowed text.
        navStack.axis = .horizontal
        navStack.spacing = 6
        for (index, page) in HUD.pages.enumerated() {
            var configuration = UIButton.Configuration.plain()
            configuration.contentInsets = NSDirectionalEdgeInsets(top: 6, leading: 13, bottom: 6, trailing: 13)
            var title = AttributedString(page)
            title.font = .systemFont(ofSize: 14, weight: .semibold)
            title.foregroundColor = page == active ? HUD.ink : HUD.white.withAlphaComponent(0.85)
            configuration.attributedTitle = title
            let button = UIButton(configuration: configuration)
            button.tag = index
            button.layer.cornerRadius = 16
            if page == active {
                button.backgroundColor = HUD.accent.withAlphaComponent(0.92)
            } else {
                button.shadowed()
            }
            button.addTarget(self, action: #selector(navPressed(_:)), for: .touchUpInside)
            navStack.addArrangedSubview(button)
        }

        railStack.axis = .horizontal
        railStack.spacing = 8
        railStack.alignment = .center

        sentenceDot.backgroundColor = HUD.white.withAlphaComponent(0.35)
        sentenceDot.layer.cornerRadius = 9
        sentenceLabel.font = .systemFont(ofSize: 26, weight: .semibold)
        sentenceLabel.textColor = HUD.white
        sentenceLabel.numberOfLines = 3
        sentenceLabel.shadowed()

        bigWord.font = .systemFont(ofSize: 48, weight: .heavy)
        bigWord.textAlignment = .center
        bigWord.adjustsFontSizeToFitWidth = true
        bigWord.minimumScaleFactor = 0.4
        bigWord.shadowed(0.9, radius: 4)
        barTrack.backgroundColor = HUD.ink.withAlphaComponent(HUD.panelAlpha)
        barTrack.layer.cornerRadius = 3
        barFill.layer.cornerRadius = 3
        scoreLabel.font = .monospacedDigitSystemFont(ofSize: 13, weight: .medium)
        scoreLabel.textColor = HUD.muted
        scoreLabel.textAlignment = .center

        statsPanel.backgroundColor = HUD.ink.withAlphaComponent(HUD.panelAlpha)
        statsPanel.layer.cornerRadius = 14
        statsPanel.layer.borderColor = HUD.white.withAlphaComponent(0.35).cgColor
        statsPanel.layer.borderWidth = 0.5
        statsLabel.numberOfLines = 0
        statsLabel.attributedText = stats([("FPS", "--"), ("FRAME", "--"), ("HANDS", "--"), ("FRAMES", "0")])
        candidatesStack.axis = .vertical
        candidatesStack.spacing = 6
        candidatesStack.alignment = .trailing

        controls.axis = .horizontal
        controls.spacing = 12
        if isPractice {
            pill(skipButton, "SKIP THIS SIGN", #selector(skipPressed), width: 190, primary: false)
            controls.addArrangedSubview(skipButton)
        } else {
            pill(resetButton, "RESET", #selector(resetPressed), width: 120, primary: false)
            pill(startButton, "START", #selector(toggleStart), width: 170, primary: true)
            pill(finishButton, "FINISH", #selector(finishPressed), width: 130, primary: true)
            [resetButton, startButton, finishButton].forEach { controls.addArrangedSubview($0) }
        }

        // Practice furniture.
        bannerBox.backgroundColor = HUD.ink.withAlphaComponent(225 / 255)
        bannerBox.layer.cornerRadius = 22
        bannerBox.layer.borderWidth = 1
        banner.font = .systemFont(ofSize: 17, weight: .semibold)
        banner.textAlignment = .center
        banner.translatesAutoresizingMaskIntoConstraints = false
        bannerBox.addSubview(banner)
        pipView.backgroundColor = HUD.ink
        pipView.layer.borderColor = HUD.white.cgColor
        pipView.layer.borderWidth = 1
        pipView.clipsToBounds = true
        pipLabel.text = "HOW IT LOOKS"
        pipLabel.font = .systemFont(ofSize: 11, weight: .semibold)
        pipLabel.textColor = HUD.muted
        pipLabel.textAlignment = .center
        pipLabel.backgroundColor = HUD.ink.withAlphaComponent(225 / 255)
        pipLabel.translatesAutoresizingMaskIntoConstraints = false
        pipView.addSubview(pipLabel)
        flash.backgroundColor = HUD.accent.withAlphaComponent(150 / 255)
        flash.alpha = 0
        flash.isUserInteractionEnabled = false
        flashTitle.text = "YEHEY!"
        flashTitle.font = .systemFont(ofSize: 68, weight: .heavy)
        flashTitle.textColor = HUD.white
        flashTitle.shadowed(0.9, radius: 4)
        flashGloss.font = .systemFont(ofSize: 30, weight: .semibold)
        flashGloss.textColor = HUD.ink
        for label in [flashTitle, flashGloss] {
            label.textAlignment = .center
            label.translatesAutoresizingMaskIntoConstraints = false
            flash.addSubview(label)
        }

        var views: [UIView] = [stateHost, bigWord, barTrack, scoreLabel, statsPanel, statsLabel, candidatesStack, controls]
        views += isPractice ? [roundHost, bannerBox, pipView] : [railStack, sentenceDot, sentenceLabel]
        views += [flash, navStack]
        for v in views {
            v.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(v)
        }
        barFill.translatesAutoresizingMaskIntoConstraints = false
        barTrack.addSubview(barFill)
        barFillWidth = barFill.widthAnchor.constraint(equalToConstant: 0)

        NSLayoutConstraint.activate([
            navStack.topAnchor.constraint(equalTo: guide.topAnchor, constant: 8),
            navStack.centerXAnchor.constraint(equalTo: guide.centerXAnchor),

            stateHost.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 16),
            stateHost.topAnchor.constraint(equalTo: navStack.bottomAnchor, constant: 12),

            controls.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            controls.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),

            scoreLabel.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -12),
            scoreLabel.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            barTrack.bottomAnchor.constraint(equalTo: scoreLabel.topAnchor, constant: -6),
            barTrack.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            barTrack.widthAnchor.constraint(equalToConstant: 220),
            barTrack.heightAnchor.constraint(equalToConstant: 6),
            barFill.leadingAnchor.constraint(equalTo: barTrack.leadingAnchor),
            barFill.topAnchor.constraint(equalTo: barTrack.topAnchor),
            barFill.bottomAnchor.constraint(equalTo: barTrack.bottomAnchor),
            barFillWidth,
            bigWord.bottomAnchor.constraint(equalTo: barTrack.topAnchor, constant: -8),
            bigWord.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            bigWord.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.5),

            statsPanel.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 16),
            statsPanel.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),
            statsLabel.leadingAnchor.constraint(equalTo: statsPanel.leadingAnchor, constant: 12),
            statsLabel.trailingAnchor.constraint(equalTo: statsPanel.trailingAnchor, constant: -12),
            statsLabel.topAnchor.constraint(equalTo: statsPanel.topAnchor, constant: 10),
            statsLabel.bottomAnchor.constraint(equalTo: statsPanel.bottomAnchor, constant: -10),

            candidatesStack.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -16),
            candidatesStack.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),

            flash.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            flash.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            flash.topAnchor.constraint(equalTo: view.topAnchor),
            flash.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            flashTitle.centerXAnchor.constraint(equalTo: flash.centerXAnchor),
            flashTitle.centerYAnchor.constraint(equalTo: flash.centerYAnchor, constant: -30),
            flashGloss.centerXAnchor.constraint(equalTo: flash.centerXAnchor),
            flashGloss.topAnchor.constraint(equalTo: flashTitle.bottomAnchor, constant: 8),
        ])
        if isPractice {
            NSLayoutConstraint.activate([
                roundHost.leadingAnchor.constraint(equalTo: stateHost.leadingAnchor),
                roundHost.topAnchor.constraint(equalTo: stateHost.bottomAnchor, constant: 8),
                bannerBox.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
                bannerBox.topAnchor.constraint(equalTo: navStack.bottomAnchor, constant: 10),
                bannerBox.heightAnchor.constraint(equalToConstant: 44),
                bannerBox.widthAnchor.constraint(greaterThanOrEqualToConstant: 320),
                banner.leadingAnchor.constraint(equalTo: bannerBox.leadingAnchor, constant: 22),
                banner.trailingAnchor.constraint(equalTo: bannerBox.trailingAnchor, constant: -22),
                banner.centerYAnchor.constraint(equalTo: bannerBox.centerYAnchor),
                pipView.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -16),
                pipView.topAnchor.constraint(equalTo: navStack.bottomAnchor, constant: 12),
                pipView.widthAnchor.constraint(equalTo: guide.widthAnchor, multiplier: 0.24),
                pipView.heightAnchor.constraint(equalTo: pipView.widthAnchor, multiplier: 0.75, constant: 22),
                pipLabel.leadingAnchor.constraint(equalTo: pipView.leadingAnchor),
                pipLabel.trailingAnchor.constraint(equalTo: pipView.trailingAnchor),
                pipLabel.bottomAnchor.constraint(equalTo: pipView.bottomAnchor),
                pipLabel.heightAnchor.constraint(equalToConstant: 22),
            ])
            loadReference()
        } else {
            NSLayoutConstraint.activate([
                railStack.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -16),
                railStack.centerYAnchor.constraint(equalTo: stateHost.centerYAnchor),
                railStack.leadingAnchor.constraint(greaterThanOrEqualTo: stateHost.trailingAnchor, constant: 16),
                sentenceDot.leadingAnchor.constraint(equalTo: stateHost.leadingAnchor),
                sentenceDot.topAnchor.constraint(equalTo: sentenceLabel.topAnchor, constant: 7),
                sentenceDot.widthAnchor.constraint(equalToConstant: 18),
                sentenceDot.heightAnchor.constraint(equalToConstant: 18),
                sentenceLabel.leadingAnchor.constraint(equalTo: sentenceDot.trailingAnchor, constant: 12),
                sentenceLabel.topAnchor.constraint(equalTo: stateHost.bottomAnchor, constant: 16),
                sentenceLabel.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.78),
            ])
        }
    }

    // MARK: - Session history

    /// On `queue`: one History row per LIVE visit that recognised or said something.
    private func saveSession() {
        guard !isPractice, !sessionGlosses.isEmpty || !sessionSentences.isEmpty else { return }
        let value: [String: Any] = [
            "format": "slt_mobile_live_session_v1", "mode": "live",
            "started_utc": ISO8601DateFormatter().string(from: openedAt),
            "duration_seconds": Date().timeIntervalSince(openedAt),
            "glosses": sessionGlosses, "sentences": sessionSentences, "complete": true,
            "events": events, "timing": LiveReelShared.engine?.runtime.timing ?? [:],
            "device": UIDevice.current.model, "system": UIDevice.current.systemVersion,
        ]
        LiveReelSessions.save(value, started: openedAt)
        events = []
    }
}
