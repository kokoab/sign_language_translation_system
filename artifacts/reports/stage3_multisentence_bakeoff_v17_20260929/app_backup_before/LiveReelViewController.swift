// LIVE and PRACTICE (landscape, full-screen camera) of the ATLAS app. Both modes share one layout;
// panels are iOS 26 Liquid Glass (a dark blur before iOS 26) over the camera, buttons are gradients:
//   top bar      back · mode + status (dot while running) · LIVE Letters (blue = on) / PRACTICE score · gear
//   under bar    LIVE: committed-word pills (newest last, amber "WORD?" preview) on the right,
//                the finished sentence on the left (its dot turns green while it is spoken)
//                PRACTICE: the round + sign to perform (top centre), the reference clip (top right)
//   bottom-left  current word (amber preview / white commit / practice verdict) with confidence,
//                and the top-3 candidate pills under it
//   bottom-right round ↺ then the big button: LIVE Start → Finish (no Stop; Back leaves and stops the
//                camera), PRACTICE Skip → Next (the round starts from the set chosen on the setup page)
//   gear         diagnostics panel (FPS, frame time, hands, frames)
// LIVE processes nothing until Start. PRACTICE runs as soon as the round opens.
// ↺ LIVE clears the signs so far; PRACTICE retries the current sign. Long-press it for a hint.
// A matched practice sign holds for a moment (Next skips the wait); every judged sign clears the buffer.

import AVFoundation
import AudioToolbox
import CoreText
import Flutter
import UIKit

/// The ATLAS colour variables and the Inter faces bundled as Flutter assets.
private enum ReelStyle {
    static func rgb(_ hex: UInt32) -> UIColor {
        UIColor(red: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }

    static let ink = rgb(0x08265C)
    static let night = rgb(0x091B32)
    static let white = rgb(0xFFFFFF)
    static let muted = rgb(0x52698C)
    static let blue = rgb(0x0769E8)
    static let sky = rgb(0x27A8FF)
    static let pale = rgb(0xEAF5FF)
    static let gold = rgb(0xFFBD36)
    static let line = rgb(0xD9E4F1)
    static let green = rgb(0x17845F)
    static let danger = rgb(0xD73548)
    static let panelAlpha: CGFloat = 0.88

    private static let registered: Void = {
        for face in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"] {
            let key = FlutterDartProject.lookupKey(forAsset: "assets/fonts/Inter-\(face).ttf")
            if let path = Bundle.main.path(forResource: key, ofType: nil) {
                CTFontManagerRegisterFontsForURL(URL(fileURLWithPath: path) as CFURL, .process, nil)
            }
        }
    }()

    /// Inter at the given weight; the system font if the asset is missing.
    static func font(_ size: CGFloat, _ weight: UIFont.Weight) -> UIFont {
        _ = registered
        let face: String
        switch weight {
        case .heavy, .black: face = "ExtraBold"
        case .bold: face = "Bold"
        case .semibold: face = "SemiBold"
        case .medium: face = "Medium"
        default: face = "Regular"
        }
        return UIFont(name: "Inter-\(face)", size: size) ?? .systemFont(ofSize: size, weight: weight)
    }
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

private extension UIFont {
    /// Tabular figures, so scores and percentages do not jitter as they change.
    func withMonospacedDigits() -> UIFont {
        let feature: [UIFontDescriptor.FeatureKey: Int] = [.type: kNumberSpacingType, .selector: kMonospacedNumbersSelector]
        return UIFont(descriptor: fontDescriptor.addingAttributes([.featureSettings: [feature]]), size: pointSize)
    }
}

/// Control / Button variants.
private enum SlotStyle { case primary, secondary, disabled }

private extension UIColor {
    /// The same hue, `amount` darker (the lower stop of a button gradient).
    func darker(_ amount: CGFloat = 0.16) -> UIColor {
        var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        getHue(&h, saturation: &s, brightness: &b, alpha: &a)
        return UIColor(hue: h, saturation: min(1, s * 1.04), brightness: b * (1 - amount), alpha: a)
    }
}

/// Liquid Glass on iOS 26, an ultra-thin blur before it. Dark by default; `light` is white glass.
/// `radius == nil` is a capsule.
private final class GlassView: UIVisualEffectView {
    private var capsule = false

    init(radius: CGFloat? = nil, light: Bool = false) {
        if #available(iOS 26.0, *) {
            let glass = UIGlassEffect(style: .regular)
            glass.tintColor = light ? UIColor.white.withAlphaComponent(0.75) : UIColor.black.withAlphaComponent(0.22)
            super.init(effect: glass)
            cornerConfiguration = radius.map { .uniformCorners(radius: .fixed($0)) } ?? .capsule()
        } else {
            super.init(effect: UIBlurEffect(style: light ? .systemThinMaterialLight : .systemUltraThinMaterialDark))
            capsule = radius == nil
            layer.cornerRadius = radius ?? 0
            layer.cornerCurve = .continuous
            clipsToBounds = true
        }
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        if capsule { layer.cornerRadius = bounds.height / 2 }
    }
}

/// Control / Button with depth: a top-to-bottom gradient of its colour, a soft top sheen,
/// a hairline edge and a drop shadow.
private final class GradientButton: UIButton {
    private let fill = CAGradientLayer()
    private let sheen = CAGradientLayer()
    private let icon = UIImageView()
    private let caption = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        // The label, then its symbol.
        icon.contentMode = .scaleAspectFit
        caption.font = HUD.font(19, .semibold)
        let stack = UIStackView(arrangedSubviews: [caption, icon])
        stack.axis = .horizontal
        stack.spacing = 8
        stack.alignment = .center
        stack.isUserInteractionEnabled = false
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
        for l in [fill, sheen] {
            l.cornerRadius = 16
            l.cornerCurve = .continuous
        }
        sheen.colors = [UIColor.white.withAlphaComponent(0.26).cgColor, UIColor.white.withAlphaComponent(0).cgColor]
        sheen.locations = [0, 0.55]
        layer.cornerRadius = 16
        layer.cornerCurve = .continuous
        layer.borderWidth = 1
        layer.borderColor = UIColor.white.withAlphaComponent(0.22).cgColor
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.28
        layer.shadowRadius = 8
        layer.shadowOffset = CGSize(width: 0, height: 3)
    }

    required init?(coder: NSCoder) { fatalError() }

    func content(_ title: String, symbol: String, color: UIColor) {
        if caption.text != title { caption.text = title }
        caption.textColor = color
        icon.image = UIImage(systemName: symbol, withConfiguration: UIImage.SymbolConfiguration(pointSize: 16, weight: .bold))
        icon.tintColor = color
        accessibilityLabel = title
    }

    func paint(_ color: UIColor) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        fill.colors = [color.cgColor, color.darker().cgColor]
        CATransaction.commit()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        // Kept beneath the title, whenever UIKit (re)adds it.
        layer.insertSublayer(fill, at: 0)
        layer.insertSublayer(sheen, at: 1)
        fill.frame = bounds
        sheen.frame = bounds
    }
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
    // Pose tracking is owned by captureQueue, independently of model inference.
    private let captureQueue = DispatchQueue(label: "slt.live.capture", qos: .userInteractive)
    private let frameGate = LiveLatestFrameGate<CMSampleBuffer>()
    private let poseScaler = LiveDetectionScaler()
    private var poseVision = LiveVision(auxiliaryInterval: 4)
    private var poseDeadline = 0.0
    private var poseBody: [LivePoint]?
    private var poseFace: [LivePoint]?
    private var events: [[String: Any]] = []
    private var sessionGlosses: [String] = []
    private var sessionSentences: [String] = []
    private var lastHistorySave = Date.distantPast
    private var lastPreviewEvent: String?
    private var lastPreviewEventTime = -Double.infinity
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
    private var loadTimer: Timer?
    /// Banner text other than the live words: the Stage 3 sentence, or a load/frame notice.
    private var bannerMessage: (caption: String, text: String)?

    // Practice (main thread).
    private var targets: [String] = []
    private var practiceIndex = 0
    private var practiceScore = 0
    private var practiceAttempts = 0
    private var practiceMissed: [String] = []
    private var verdict: (kind: String, gloss: String, at: Date)?
    private var roundDone = false
    /// A matched sign waits here (Next, or a short pause) before the round moves on.
    private var awaitingNext = false

    // Views.
    private let backButton = UIButton(type: .system)
    private let liveDot = UIView()
    private let modeLabel = UILabel()
    private let statusLabel = UILabel()
    /// LIVE: fingerspelling on/off (filled blue while on). PRACTICE: the score.
    private let lettersButton = UIButton(type: .custom)
    private let lettersFill = UIView()
    private let scoreChipLabel = UILabel()
    private let gearButton = UIButton(type: .system)
    private let railStack = UIStackView()
    private var railChipKey: [String]?
    private let sentenceDot = UIView()
    private let sentenceLabel = UILabel()
    private let bannerBox = GlassView(radius: 18)
    private let bannerCaption = UILabel()
    private let bannerValue = UILabel()
    private let promptTitle = UILabel()
    private let promptDetail = UILabel()
    private let bigWord = UILabel()
    private let barTrack = UIView()
    private let barFill = UIView()
    private var barFillWidth: NSLayoutConstraint!
    private let scoreLabel = UILabel()
    private let feedbackPill = UIView()
    private let feedbackLabel = UILabel()
    private let candidatesStack = UIStackView()
    private var candidateChipKey: [String]?
    private let statsPanel = GlassView(radius: 16)
    private let statsLabel = UILabel()
    /// Round glass ↺: LIVE clears the signs so far, PRACTICE retries the current sign.
    private let resetButton = UIButton(type: .system)
    private let resetGlass = GlassView(light: true)
    /// LIVE Start → Finish; PRACTICE Skip → Next.
    private let primaryButton = GradientButton(type: .custom)
    private let hintGlass = GlassView()
    private let hintLabel = UILabel()
    /// Live only: fingerspelling on/off (the desktop --no-fingerspelling), remembered across launches.
    private static let lettersKey = "SLTLiveFingerspellingV17"
    private var lettersOn: Bool = UserDefaults.standard.object(forKey: LiveReelViewController.lettersKey) as? Bool ?? true
    private let pipView = GlassView(radius: 18)
    private let pipVideo = UIView()
    private let pipLabel = UILabel()
    private var pipPlayer: AVQueuePlayer?
    private var pipLooper: AVPlayerLooper?
    private var pipLayer: AVPlayerLayer?
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
                self.bannerMessage = ("MODELS", "Model load failed: \(status["error"] as? String ?? "")")
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
        isRunningUI = false
        frameGate.setEnabled(false)
        // Synchronous so the camera is free before Flutter shows its next page.
        queue.sync {
            running = false
            frameGate.setEnabled(false)
            session.stopRunning()
            saveSession(complete: true)
            LiveReelShared.engine?.reset()
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
        pipLayer?.frame = pipVideo.bounds
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
        output.setSampleBufferDelegate(self, queue: captureQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let connection = output.connection(with: .video) {
            // The model contract is upright and unmirrored; only the on-screen preview is mirrored.
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
        session.commitConfiguration()
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill      // full screen, no letterbox
        if let connection = previewLayer.connection {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = true
        }
        view.layer.insertSublayer(previewLayer, at: 0)
        bones.strokeColor = HUD.sky.withAlphaComponent(0.9).cgColor
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
            if connection.videoRotationAngle != angle { frameGate.setEnabled(false) }
            queue.async {
                if connection.videoRotationAngle != angle {
                    self.frameGate.setEnabled(false)
                    connection.videoRotationAngle = angle
                    // Frame geometry changed: start the stream fresh.
                    if self.running {
                        LiveReelShared.engine?.resetStream()
                        LiveReelShared.engine?.resetFinishGesture()
                        LiveReelShared.engine?.vision.resetTracking()
                    }
                    self.resetPoseTracking()
                    self.frameGate.setEnabled(self.running)
                    self.deadline = 0
                    self.frameTimes = []
                    DispatchQueue.main.async {
                        self.bones.path = nil; self.joints.path = nil
                        self.previewText = nil; self.centerPreview = nil; self.top3 = []
                        self.finishGestureProgress = nil
                        self.render()
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
        // Practice judges the 100 signs exactly as the desktop does (letters on); Live follows the switch.
        LiveReelShared.engine?.fingerspelling = isPractice || lettersSetting
        LiveReelShared.engine?.reset()
        firstStamp = nil
        deadline = 0
        frameTimes = []
        resetPoseTracking()
        frameGate.setEnabled(true)
        running = true
        events.append(["event": "start", "time": Date().timeIntervalSince1970])
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let generation = frameGate.activeGeneration,
              let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if frameGate.offer(sampleBuffer, generation: generation) { queue.async { self.processNextCapture() } }
        let seconds = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        guard seconds + 1e-6 >= poseDeadline else { return }
        poseDeadline = max(poseDeadline + 1 / LiveReelEngine.fps, seconds)
        do {
            let observation = try autoreleasepool {
                let image = try poseScaler.resize(frame)
                return try poseVision.observe(detection: image, width: CVPixelBufferGetWidth(frame),
                                              height: CVPixelBufferGetHeight(frame), seconds: seconds)
            }
            if observation.faceForFeatures { poseBody = observation.body; poseFace = observation.face }
            let body = poseBody, face = poseFace
            DispatchQueue.main.async {
                guard self.isRunningUI, self.frameGate.isCurrent(generation) else { return }
                self.drawSkeleton(observation, body: body, face: face)
            }
        } catch { /* A failed overlay frame must not interrupt recognition. */ }
    }

    private func resetPoseTracking() {
        captureQueue.async {
            self.poseVision = LiveVision(auxiliaryInterval: 4)
            self.poseDeadline = 0
            self.poseBody = nil; self.poseFace = nil
        }
    }

    /// One model job at a time; reschedule to let Start/Stop/Reset/Finish interleave.
    private func processNextCapture() {
        if let (buffer, generation) = frameGate.take(), frameGate.isCurrent(generation) {
            processCapturedFrame(buffer)
        }
        if frameGate.complete() { queue.async { self.processNextCapture() } }
    }

    private func processCapturedFrame(_ sampleBuffer: CMSampleBuffer) {
        guard running, let engine = LiveReelShared.engine, let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let stamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if firstStamp == nil { firstStamp = stamp }
        let seconds = CMTimeGetSeconds(CMTimeSubtract(stamp, firstStamp!))
        guard seconds + 1e-6 >= deadline else { return }
        deadline = max(deadline + 1 / LiveReelEngine.fps, seconds)
        do {
            let step = try engine.process(frame, seconds: seconds, allowFinishGesture: !isPractice)
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
            let preview = step.finishProgress == nil ? engine.preview : nil
            let pending = step.finishProgress != nil || engine.speller.letters.isEmpty ? nil : engine.speller.pending
            let previewKey = pending.map { "fs-" + $0 } ?? preview?.gloss ?? ""
            if !isPractice, previewKey != lastPreviewEvent, seconds - lastPreviewEventTime >= 0.25 {
                events.append(["event": "preview", "seconds": seconds, "gloss": previewKey,
                               "score": preview?.score ?? 0])
                lastPreviewEvent = previewKey; lastPreviewEventTime = seconds
            }
            if !isPractice, Date().timeIntervalSince(lastHistorySave) >= 5 { saveSession() }
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
            DispatchQueue.main.async { self.bannerMessage = ("NOTICE", "Frame error: \(error.localizedDescription)"); self.render() }
        }
    }

    // MARK: - Controls

    /// LIVE Start/Stop (Stop finishes the utterance). PRACTICE Stop pauses the round; Start resumes it
    /// through the same startStream() the round opens with.
    @objc private func toggleStart() {
        guard loaded else { return }
        let starting = !isRunningUI
        isRunningUI = starting
        if isPractice && !starting { previewText = nil; centerPreview = nil; top3 = [] }
        render()
        queue.async {
            if starting {
                self.startStream()
            } else {
                self.running = false
                self.frameGate.setEnabled(false)
                if self.isPractice {
                    LiveReelShared.engine?.reset()      // no utterance in Practice; the next attempt starts clean
                } else {
                    self.finishUtterance(reason: "stop")
                }
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

    /// Read on the engine queue.
    private var lettersSetting: Bool { UserDefaults.standard.object(forKey: Self.lettersKey) as? Bool ?? true }

    @objc private func lettersTapped() {
        guard !isPractice else { return }
        lettersOn.toggle()
        UserDefaults.standard.set(lettersOn, forKey: Self.lettersKey)
        let on = lettersOn
        render()
        // Switching decoders restarts the open utterance's recognition (committed words stay).
        queue.async {
            LiveReelShared.engine?.fingerspelling = on
            self.events.append(["event": "fingerspelling", "on": on])
        }
    }

    @objc private func resetPressed() {
        queue.async {
            LiveReelShared.engine?.reset()
            self.frameGate.setEnabled(self.running)
            self.resetPoseTracking()
            self.utterance = []
            self.events.append(["event": "reset"])
            DispatchQueue.main.async {
                self.speech.stopSpeaking(at: .immediate)
                self.rail = []; self.previewText = nil; self.committedWord = nil
                self.centerPreview = nil; self.top3 = []; self.finishing = false
                self.finishGestureProgress = nil
                self.bannerMessage = nil
                self.render()
            }
        }
    }

    /// The big button: LIVE Start, then Finish while running; PRACTICE Skip, or Next after a match.
    @objc private func primaryPressed() {
        if isPractice {
            awaitingNext ? nextPressed() : skipPressed()
        } else if isRunningUI {
            finishPressed()
        } else {
            toggleStart()
        }
    }

    /// PRACTICE ↺: drop what was read so far and try the same sign again.
    @objc private func retryPressed() {
        guard !awaitingNext, target != nil else { return }
        verdict = nil
        clearBuffer()
        render()
    }

    @objc private func resetHint(_ gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began else { return }
        hintLabel.text = isPractice ? "Try this sign again" : "Clear the signs so far"
        hintGlass.layer.removeAllAnimations()
        hintGlass.alpha = 1
        UIView.animate(withDuration: 0.4, delay: 1.6, options: [.beginFromCurrentState]) { self.hintGlass.alpha = 0 }
    }

    /// LIVE returns Home; PRACTICE returns to its setup (or the result page once the round is done).
    @objc private func backPressed() {
        close(navigate: isPractice ? "PRACTICE" : "HOME")
    }

    @objc private func gearPressed() {
        statsPanel.isHidden.toggle()
        render()
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
                self.bannerMessage = ("TRANSLATION", "No recognized signs to finish.")
            } else {
                self.finishing = true
                self.bannerMessage = nil
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
                self.bannerMessage = ("TRANSLATION", rendered.sentence)
                self.rail = []
                self.say(rendered.sentence)
                self.render()
            }
        }
    }

    // MARK: - Practice (AppShell._check_practice / skip_practice)

    private var target: String? { !roundDone && practiceIndex < targets.count ? targets[practiceIndex] : nil }

    private func judge(_ committed: [LiveWord]) {
        guard !awaitingNext, let target, let first = committed.first else { return }
        let seen = Self.display(first.gloss)
        practiceAttempts += 1
        if seen == target {
            practiceScore += 1
            verdict = ("matched", seen, Date())
            AudioServicesPlaySystemSound(1057)
            // Hold the match on screen; Next moves on at once, otherwise the round continues by itself.
            awaitingNext = true
            let index = practiceIndex
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                guard let self, self.awaitingNext, self.practiceIndex == index else { return }
                self.nextPressed()
            }
        } else {
            verdict = ("missed", seen, Date())
        }
        clearBuffer()      // judged either way: the next attempt starts from nothing
    }

    @objc private func nextPressed() {
        guard awaitingNext else { return }
        awaitingNext = false
        verdict = nil
        advance()
        clearBuffer()
        render()
    }

    @objc private func skipPressed() {
        guard !awaitingNext, let target else { return }
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
            // Let the last verdict show, then hand the score to the result page.
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.close(navigate: "PRACTICE") }
            return
        }
        loadReference()
    }

    private func clearBuffer() {
        rail = []; previewText = nil; centerPreview = nil; top3 = []
        queue.async { LiveReelShared.engine?.reset() }
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
        layer.frame = pipVideo.bounds
        pipVideo.layer.insertSublayer(layer, at: 0)
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
        if !committed.isEmpty, !finishing, bannerMessage != nil {
            bannerMessage = nil             // a new utterance has started
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

    private func say(_ text: String) {
        speech.speak(AVSpeechUtterance(string: text))
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        render()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        render()
    }

    private func stats(_ rows: [(String, String)]) -> NSAttributedString {
        let text = NSMutableAttributedString()
        for (i, (label, value)) in rows.enumerated() {
            text.append(NSAttributedString(string: label.padding(toLength: 8, withPad: " ", startingAt: 0), attributes: [
                .font: UIFont.monospacedSystemFont(ofSize: 11, weight: .semibold), .foregroundColor: HUD.white.withAlphaComponent(0.6)]))
            text.append(NSAttributedString(string: value + (i < rows.count - 1 ? "\n" : ""), attributes: [
                .font: UIFont.monospacedSystemFont(ofSize: 13, weight: .medium), .foregroundColor: HUD.white]))
        }
        return text
    }

    private func stylePrimary(_ kind: SlotStyle, _ title: String, _ symbol: String) {
        let (fill, text): (UIColor, UIColor) = kind == .primary ? (HUD.blue, HUD.white) : (HUD.pale, HUD.ink)
        primaryButton.paint(fill)
        primaryButton.content(title, symbol: symbol, color: text)
        primaryButton.isEnabled = kind != .disabled
        primaryButton.alpha = kind == .disabled ? 0.45 : 1
    }

    /// A glass capsule holding an optional coloured dot and a label (the desktop `_chip`).
    private func glassChip(_ title: String, dot: UIColor?, textColor: UIColor, size: CGFloat = 13) -> UIView {
        let chip = GlassView()
        let label = UILabel()
        text(label, size, .semibold, textColor)
        label.text = title
        var items: [UIView] = [label]
        if let dot {
            let d = UIView()
            d.backgroundColor = dot
            d.layer.cornerRadius = 4
            d.widthAnchor.constraint(equalToConstant: 8).isActive = true
            d.heightAnchor.constraint(equalToConstant: 8).isActive = true
            items.insert(d, at: 0)
        }
        let row = UIStackView(arrangedSubviews: items)
        row.spacing = 7
        row.alignment = .center
        inset(row, in: chip.contentView, 7, 12)
        return chip
    }

    /// Redraw every overlay from the display state.
    private func render() {
        // Top bar: mode and status.
        let hasHands = bones.path.map { !$0.isEmpty } ?? false
        let dim = HUD.white.withAlphaComponent(0.7)
        let (state, stateColor): (String, UIColor) =
            !loaded ? ("Warming up", dim) :
            finishing ? ("Translating…", HUD.gold) :
            !isRunningUI ? (isPractice ? "Starting" : "Ready", dim) :
            finishGestureProgress != nil ? (finishGestureProgress! < 1
                ? "Hold to finish \(Int(finishGestureProgress! * 100))%" : "Lower your hands", HUD.gold) :
            centerPreview != nil ? ("Reading sign…", HUD.gold) :
            hasHands ? ("Signing", HUD.white) : ("Watching", dim)
        statusLabel.text = state
        statusLabel.textColor = stateColor
        liveDot.backgroundColor = isRunningUI ? HUD.green : HUD.white.withAlphaComponent(0.4)
        if isPractice {
            scoreChipLabel.text = "Score \(practiceScore) / \(targets.count)"
        } else {
            lettersFill.isHidden = !lettersOn
            lettersButton.setTitleColor(lettersOn ? HUD.white : HUD.white.withAlphaComponent(0.7), for: .normal)
            lettersButton.accessibilityValue = lettersOn ? "On" : "Off"
            lettersButton.accessibilityTraits = lettersOn ? [.button, .selected] : .button
        }

        // Live: committed-word pills (newest at the right) and the amber preview after them.
        let railKey = Array(rail.suffix(7)) + ["\u{0}", previewText ?? "\u{1}"]
        if !isPractice && railChipKey != railKey {
            railChipKey = railKey
            railStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
            let recent = Array(rail.suffix(7))
            for (index, word) in recent.enumerated() {
                let newest = index == recent.count - 1 && previewText == nil
                railStack.addArrangedSubview(glassChip(word, dot: newest ? HUD.green : HUD.white.withAlphaComponent(0.4),
                                                       textColor: newest ? HUD.white : HUD.white.withAlphaComponent(0.7)))
            }
            if let previewText {
                railStack.addArrangedSubview(glassChip(previewText, dot: HUD.gold, textColor: HUD.gold))
            }
        }

        // Live: the finished sentence (or a notice) under the top bar.
        if !isPractice {
            sentenceLabel.text = finishing ? "Finishing…" : bannerMessage?.text
            sentenceLabel.textColor = finishing ? HUD.white.withAlphaComponent(0.6) : HUD.white
            sentenceDot.backgroundColor = speech.isSpeaking ? HUD.green : HUD.white.withAlphaComponent(0.35)
            sentenceDot.isHidden = sentenceLabel.text == nil
        }

        // Practice: the round and the sign to perform.
        if isPractice {
            var caption: String?, value: String?
            if let m = bannerMessage {
                (caption, value) = (m.caption, m.text)
            } else if let target {
                caption = "ROUND \(practiceIndex + 1) OF \(targets.count)  ·  " + (awaitingNext ? "MATCHED" : "SIGN THIS")
                value = target
            } else if roundDone {
                (caption, value) = ("ROUND COMPLETE", "\(practiceScore) of \(targets.count)")
            }
            bannerCaption.text = caption
            bannerValue.text = value
            bannerBox.isHidden = value == nil
        }

        // Bottom-left: the word in progress, the last commit, or the practice verdict.
        if isPractice, !awaitingNext, let v = verdict, Date().timeIntervalSince(v.at) > 3 { verdict = nil }
        var word: String?, wordColor = HUD.white, score = 0.0, level = HUD.sky, alpha: CGFloat = 1
        var feedback: (text: String, color: UIColor)?
        if isPractice, awaitingNext, let v = verdict {
            word = v.gloss; score = committedWord?.score ?? 1; level = HUD.green
            feedback = ("Matched  ·  +1", HUD.green)
        } else if let p = centerPreview {
            word = p.text; wordColor = HUD.gold; score = p.score; level = HUD.gold
        } else if isPractice, let v = verdict, v.kind == "missed" {
            word = v.gloss; score = committedWord?.score ?? 0; level = HUD.danger
            feedback = ("Saw \(v.gloss)  ·  try again", HUD.danger)
        } else if let c = committedWord {
            let age = Date().timeIntervalSince(c.at)
            word = c.text; score = c.score
            alpha = age < 1.6 ? 1 : max(0.38, 1 - CGFloat(age - 1.6) * 0.43)
        }
        bigWord.text = word
        bigWord.textColor = wordColor
        barFill.backgroundColor = level
        barFillWidth.constant = 120 * CGFloat(max(0, min(1, score)))
        scoreLabel.text = "\(Int((max(0, min(1, score)) * 100).rounded()))%"
        for v in [bigWord, barTrack.superview!] as [UIView] { v.alpha = word == nil ? 0 : alpha }
        bigWord.isHidden = word == nil
        barTrack.superview!.isHidden = word == nil
        feedbackLabel.text = feedback?.text
        feedbackPill.backgroundColor = feedback?.color
        feedbackPill.isHidden = feedback == nil

        // Centre prompt when nothing is being read.
        var prompt: (String, String)?
        if word == nil && centerPreview == nil {
            if !loaded {
                prompt = ("Warming up", "Loading the on-device models…")
            } else if !isRunningUI {
                prompt = isPractice ? ("Starting", "Get ready to sign.")
                                    : ("Ready to translate", "Position your hands inside the frame, then tap Start.")
            } else if isPractice, target != nil {
                prompt = ("Your turn", "Watch the reference, then sign it.")
            }
        }
        promptTitle.text = prompt?.0
        promptDetail.text = prompt?.1
        promptTitle.superview?.isHidden = prompt == nil

        // Candidate pills under the word, the leader brightest.
        let candidateKey = top3.flatMap { [$0.gloss, String($0.score.bitPattern)] }
        if candidateChipKey != candidateKey {
            candidateChipKey = candidateKey
            candidatesStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
            for (i, c) in top3.enumerated() {
                candidatesStack.addArrangedSubview(glassChip(
                    String(format: "%@  %.2f", c.gloss, c.score),
                    dot: (i == 0 ? HUD.gold : HUD.sky).withAlphaComponent(0.45 + 0.55 * CGFloat(min(1, max(0, c.score)))),
                    textColor: i == 0 ? HUD.white : HUD.white.withAlphaComponent(0.7), size: 12))
            }
        }
        candidatesStack.isHidden = !isRunningUI || top3.isEmpty

        // Controls: ↺ then the big button.
        let canReset: Bool
        if isPractice {
            canReset = !awaitingNext && target != nil && (centerPreview != nil || verdict?.kind == "missed" || !top3.isEmpty)
            if awaitingNext {
                stylePrimary(.primary, "Next", "arrow.right")
            } else {
                stylePrimary(target != nil ? .secondary : .disabled, "Skip", "forward.end.fill")
            }
        } else {
            canReset = loaded && (!rail.isEmpty || centerPreview != nil || bannerMessage != nil || committedWord != nil)
            if !isRunningUI {
                stylePrimary(loaded ? .primary : .disabled, "Start", "play.fill")
            } else {
                // Finish waits for something to translate.
                let ready = !finishing && (!rail.isEmpty || centerPreview != nil)
                stylePrimary(ready ? .primary : .disabled, finishing ? "Working…" : "Finish", "checkmark")
            }
        }
        resetButton.isEnabled = canReset
        resetGlass.alpha = canReset ? 1 : 0.4
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

    private func text(_ label: UILabel, _ size: CGFloat, _ weight: UIFont.Weight, _ color: UIColor,
                      align: NSTextAlignment = .left) {
        label.font = HUD.font(size, weight)
        label.textColor = color
        label.textAlignment = align
    }

    /// Pin `content` inside `box` with the given insets.
    private func inset(_ content: UIView, in box: UIView, _ v: CGFloat, _ h: CGFloat) {
        content.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(content)
        NSLayoutConstraint.activate([
            content.topAnchor.constraint(equalTo: box.topAnchor, constant: v),
            content.bottomAnchor.constraint(equalTo: box.bottomAnchor, constant: -v),
            content.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: h),
            content.trailingAnchor.constraint(equalTo: box.trailingAnchor, constant: -h),
        ])
    }

    /// A 44-pt round glass button with an SF Symbol.
    private func glassIcon(_ button: UIButton, _ symbol: String, _ label: String, _ action: Selector) -> UIView {
        let glass = GlassView()
        button.setImage(UIImage(systemName: symbol, withConfiguration: UIImage.SymbolConfiguration(pointSize: 17, weight: .semibold)),
                        for: .normal)
        button.tintColor = HUD.white
        button.accessibilityLabel = label
        button.addTarget(self, action: action, for: .touchUpInside)
        inset(button, in: glass.contentView, 0, 0)
        glass.translatesAutoresizingMaskIntoConstraints = false
        glass.widthAnchor.constraint(equalToConstant: 44).isActive = true
        glass.heightAnchor.constraint(equalToConstant: 44).isActive = true
        return glass
    }

    private func buildInterface() {
        let guide = view.safeAreaLayoutGuide

        // Top bar: back · mode/status · Letters switch or score · gear.
        let back = glassIcon(backButton, "chevron.left", "Back", #selector(backPressed))
        let gear = glassIcon(gearButton, "gearshape", "Diagnostics", #selector(gearPressed))
        liveDot.layer.cornerRadius = 4
        liveDot.isHidden = isPractice
        text(modeLabel, 16, .semibold, HUD.white)
        modeLabel.text = isPractice ? "Practice" : "LIVE"
        text(statusLabel, 11, .semibold, HUD.white)
        let modeRow = UIStackView(arrangedSubviews: [liveDot, modeLabel])
        modeRow.spacing = 8
        modeRow.alignment = .center
        let modeStack = UIStackView(arrangedSubviews: [modeRow, statusLabel])
        modeStack.axis = .vertical
        modeStack.alignment = .leading
        for v in [modeLabel, statusLabel] { v.shadowed(0.6, radius: 3) }
        let chip = GlassView()
        let chipContent: UIView
        if isPractice {
            text(scoreChipLabel, 13, .semibold, HUD.white)
            chipContent = scoreChipLabel
        } else {
            lettersFill.backgroundColor = HUD.blue
            lettersFill.layer.cornerRadius = 22
            lettersFill.layer.cornerCurve = .continuous
            inset(lettersFill, in: chip.contentView, 0, 0)
            lettersButton.setTitle("Letters", for: .normal)
            lettersButton.titleLabel?.font = HUD.font(14, .semibold)
            lettersButton.accessibilityLabel = "Letters"
            lettersButton.addTarget(self, action: #selector(lettersTapped), for: .touchUpInside)
            chipContent = lettersButton
        }
        inset(chipContent, in: chip.contentView, isPractice ? 12 : 0, isPractice ? 14 : 18)
        let spacer = UIView()
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        modeStack.setContentHuggingPriority(.required, for: .horizontal)
        let topBar = UIStackView(arrangedSubviews: [back, modeStack, spacer, chip, gear])
        topBar.spacing = 14
        topBar.alignment = .center

        // Live: word pills and the finished sentence.
        railStack.axis = .horizontal
        railStack.spacing = 8
        railStack.alignment = .center
        sentenceDot.layer.cornerRadius = 9
        sentenceLabel.font = HUD.font(26, .semibold)
        sentenceLabel.textColor = HUD.white
        sentenceLabel.numberOfLines = 3
        sentenceLabel.shadowed(0.9, radius: 4)

        // Practice: round banner (top centre) and the reference clip (left).
        text(bannerCaption, 11, .semibold, HUD.white.withAlphaComponent(0.75), align: .center)
        text(bannerValue, 24, .semibold, HUD.white, align: .center)
        bannerValue.adjustsFontSizeToFitWidth = true
        bannerValue.minimumScaleFactor = 0.6
        let bannerStack = UIStackView(arrangedSubviews: [bannerCaption, bannerValue])
        bannerStack.axis = .vertical
        bannerStack.spacing = 2
        inset(bannerStack, in: bannerBox.contentView, 10, 20)
        pipVideo.backgroundColor = HUD.night
        pipVideo.layer.cornerRadius = 12
        pipVideo.layer.cornerCurve = .continuous
        pipVideo.clipsToBounds = true
        text(pipLabel, 12, .semibold, HUD.white.withAlphaComponent(0.8))
        pipLabel.text = "Reference"
        let pipStack = UIStackView(arrangedSubviews: [pipVideo, pipLabel])
        pipStack.axis = .vertical
        pipStack.spacing = 4
        inset(pipStack, in: pipView.contentView, 8, 8)

        // Bottom-left: word, confidence bar and (Practice) verdict pill.
        bigWord.font = HUD.font(34, .heavy)
        bigWord.adjustsFontSizeToFitWidth = true
        bigWord.minimumScaleFactor = 0.5
        bigWord.shadowed(0.9, radius: 4)
        barTrack.backgroundColor = HUD.white.withAlphaComponent(0.3)
        barTrack.layer.cornerRadius = 2.5
        barFill.layer.cornerRadius = 2.5
        barFill.translatesAutoresizingMaskIntoConstraints = false
        barTrack.addSubview(barFill)
        barFillWidth = barFill.widthAnchor.constraint(equalToConstant: 0)
        text(scoreLabel, 13, .semibold, HUD.white)
        scoreLabel.font = HUD.font(13, .semibold).withMonospacedDigits()
        scoreLabel.shadowed(0.7, radius: 2)
        let confidence = UIStackView(arrangedSubviews: [barTrack, scoreLabel])
        confidence.spacing = 10
        confidence.alignment = .center
        feedbackPill.layer.cornerRadius = 13
        text(feedbackLabel, 11, .semibold, HUD.white, align: .center)
        inset(feedbackLabel, in: feedbackPill, 5, 12)
        feedbackPill.isHidden = true
        // Candidate pills, stacked under the word (leader first).
        candidatesStack.axis = .vertical
        candidatesStack.spacing = 6
        candidatesStack.alignment = .leading
        let wordStack = UIStackView(arrangedSubviews: [feedbackPill, bigWord, confidence, candidatesStack])
        wordStack.axis = .vertical
        wordStack.spacing = 4
        wordStack.alignment = .leading
        wordStack.setCustomSpacing(10, after: confidence)

        // Centre prompt (only while nothing is being read).
        text(promptTitle, 24, .semibold, HUD.white, align: .center)
        text(promptDetail, 13, .regular, HUD.white, align: .center)
        promptDetail.numberOfLines = 2
        promptTitle.shadowed(0.8, radius: 4)
        promptDetail.shadowed(0.8, radius: 3)
        let promptStack = UIStackView(arrangedSubviews: [promptTitle, promptDetail])
        promptStack.axis = .vertical
        promptStack.spacing = 8

        // Diagnostics (gear).
        statsPanel.isHidden = true
        statsLabel.numberOfLines = 0
        statsLabel.attributedText = stats([("FPS", "--"), ("FRAME", "--"), ("HANDS", "--"), ("FRAMES", "0")])
        inset(statsLabel, in: statsPanel.contentView, 10, 12)

        // Controls, bottom-right: ↺ then the big button.
        resetButton.setImage(UIImage(systemName: "arrow.counterclockwise",
                                     withConfiguration: UIImage.SymbolConfiguration(pointSize: 20, weight: .semibold)),
                             for: .normal)
        resetButton.tintColor = HUD.ink
        resetButton.accessibilityLabel = isPractice ? "Try again" : "Reset"
        resetButton.accessibilityHint = isPractice ? "Tries this sign again" : "Clears the signs so far"
        resetButton.addTarget(self, action: isPractice ? #selector(retryPressed) : #selector(resetPressed), for: .touchUpInside)
        inset(resetButton, in: resetGlass.contentView, 0, 0)
        resetGlass.addGestureRecognizer(UILongPressGestureRecognizer(target: self, action: #selector(resetHint(_:))))
        primaryButton.addTarget(self, action: #selector(primaryPressed), for: .touchUpInside)
        let controls = UIStackView(arrangedSubviews: [resetGlass, primaryButton])
        controls.spacing = 14
        controls.alignment = .center
        text(hintLabel, 12, .semibold, HUD.white)
        inset(hintLabel, in: hintGlass.contentView, 7, 12)
        hintGlass.alpha = 0
        hintGlass.isUserInteractionEnabled = false

        var views: [UIView] = [promptStack, wordStack, statsPanel, controls, hintGlass]
        views += isPractice ? [pipView, bannerBox] : [railStack, sentenceDot, sentenceLabel]
        views.append(topBar)
        for v in views {
            v.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(v)
        }

        NSLayoutConstraint.activate([
            topBar.topAnchor.constraint(equalTo: guide.topAnchor, constant: 12),
            topBar.leadingAnchor.constraint(equalTo: guide.leadingAnchor),
            topBar.trailingAnchor.constraint(equalTo: guide.trailingAnchor),
            topBar.heightAnchor.constraint(equalToConstant: 44),
            chip.heightAnchor.constraint(equalToConstant: 44),
            liveDot.widthAnchor.constraint(equalToConstant: 8),
            liveDot.heightAnchor.constraint(equalToConstant: 8),

            resetGlass.widthAnchor.constraint(equalToConstant: 56),
            resetGlass.heightAnchor.constraint(equalToConstant: 56),
            primaryButton.widthAnchor.constraint(equalToConstant: 160),
            primaryButton.heightAnchor.constraint(equalToConstant: 60),
            controls.trailingAnchor.constraint(equalTo: guide.trailingAnchor),
            controls.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),
            hintGlass.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -8),
            hintGlass.trailingAnchor.constraint(equalTo: controls.trailingAnchor),

            promptStack.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            promptStack.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            promptStack.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.5),

            barTrack.widthAnchor.constraint(equalToConstant: 120),
            barTrack.heightAnchor.constraint(equalToConstant: 5),
            barFill.leadingAnchor.constraint(equalTo: barTrack.leadingAnchor),
            barFill.topAnchor.constraint(equalTo: barTrack.topAnchor),
            barFill.bottomAnchor.constraint(equalTo: barTrack.bottomAnchor),
            barFillWidth,
            bigWord.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.34),
            wordStack.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 4),
            wordStack.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),

            // Diagnostics: above the controls.
            statsPanel.trailingAnchor.constraint(equalTo: guide.trailingAnchor),
            statsPanel.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -12),
        ])
        if isPractice {
            NSLayoutConstraint.activate([
                bannerBox.topAnchor.constraint(equalTo: guide.topAnchor, constant: 12),
                bannerBox.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
                bannerBox.widthAnchor.constraint(greaterThanOrEqualToConstant: 220),
                bannerBox.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.5),
                pipView.topAnchor.constraint(equalTo: topBar.bottomAnchor, constant: 12),
                pipView.trailingAnchor.constraint(equalTo: guide.trailingAnchor),
                pipVideo.widthAnchor.constraint(equalToConstant: 130),
                pipVideo.heightAnchor.constraint(equalToConstant: 110),
            ])
            loadReference()
        } else {
            NSLayoutConstraint.activate([
                railStack.topAnchor.constraint(equalTo: topBar.bottomAnchor, constant: 10),
                railStack.trailingAnchor.constraint(equalTo: guide.trailingAnchor),
                railStack.leadingAnchor.constraint(greaterThanOrEqualTo: guide.leadingAnchor),
                railStack.heightAnchor.constraint(equalToConstant: 30),
                sentenceDot.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 4),
                sentenceDot.topAnchor.constraint(equalTo: sentenceLabel.topAnchor, constant: 7),
                sentenceDot.widthAnchor.constraint(equalToConstant: 18),
                sentenceDot.heightAnchor.constraint(equalToConstant: 18),
                sentenceLabel.leadingAnchor.constraint(equalTo: sentenceDot.trailingAnchor, constant: 12),
                sentenceLabel.topAnchor.constraint(equalTo: railStack.bottomAnchor, constant: 10),
                sentenceLabel.trailingAnchor.constraint(lessThanOrEqualTo: guide.trailingAnchor),
                // A long sentence truncates rather than running into the word and candidates.
                sentenceLabel.bottomAnchor.constraint(lessThanOrEqualTo: wordStack.topAnchor, constant: -8),
            ])
        }
    }

    // MARK: - Session history

    /// On `queue`: one History row per LIVE visit that recognised or said something.
    private func saveSession(complete: Bool = false) {
        guard !isPractice, !events.isEmpty else { return }
        let value: [String: Any] = [
            "format": "slt_mobile_live_session_v1", "mode": "live",
            "started_utc": ISO8601DateFormatter().string(from: openedAt),
            "duration_seconds": Date().timeIntervalSince(openedAt),
            "glosses": sessionGlosses, "sentences": sessionSentences, "complete": complete,
            "stage3_checkpoint": LiveReelShared.stage3?.checkpointIdentifier ?? "unloaded",
            "events": events, "timing": LiveReelShared.engine?.timing ?? [:],
            "device": UIDevice.current.model, "system": UIDevice.current.systemVersion,
        ]
        LiveReelSessions.save(value, started: openedAt)
        lastHistorySave = Date()
    }
}
