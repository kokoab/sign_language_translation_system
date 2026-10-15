// LIVE and PRACTICE (portrait or landscape, full-screen camera) of the ATLAS app. Both modes share one layout;
// panels are iOS 26 Liquid Glass (a dark blur before iOS 26) over the camera, buttons are gradients:
//   top bar      back · mode + status (dot while running) · landmarks · camera flip (LIVE) · ?
//   under bar    LIVE: committed-word pills (newest last, amber "WORD?" preview) on the right,
//                the finished sentence on the left (its dot turns green while it is spoken)
//                PRACTICE: the round + sign to perform (top centre), the reference clip (top right)
//   bottom-left  current word (amber preview / white commit / practice verdict) with confidence,
//                and the top-3 candidate pills under it
//   bottom-right round ↺ then the big button: LIVE Start → Finish (Finish ends the run and speaks the
//                sentence; Back leaves and stops the camera), PRACTICE Skip → Next (the round starts from
//                the set chosen on the setup page)
//   ?            leaves like Back, opening the manual (LIVE: reading the screen, PRACTICE: practice)
// LIVE processes nothing until Start. PRACTICE runs as soon as the round opens.
// ↺ LIVE clears the signs so far; PRACTICE retries the current sign. Long-press it for a hint.
// LIVE records each Start → Finish take to Documents/live_reel_videos (Files app); ↺ restarts the take,
// and Back saves a take still recording.
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
    /// Confidence bar length: shorter in LIVE, where the word sits in a narrow top-right column.
    private var barWidth: CGFloat { isPractice ? 120 : 90 }
    private var closeResult: [String: Any] = ["navigate": "HOME"]

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = LiveReelShared.queue
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var rotation: AVCaptureDevice.RotationCoordinator?
    private var rotationObservers: [NSKeyValueObservation] = []
    private let bones = CAShapeLayer()
    private let joints = CAShapeLayer()
    private var cameraPosition: AVCaptureDevice.Position = .front
    private var switchingCamera = false
    private let flipButton = UIButton(type: .system)
    /// ⋯ menu: Show landmarks, (LIVE) Show signs, How to use ATLAS.
    private let menuButton = UIButton(type: .system)
    private var landmarksOn = UserDefaults.standard.object(forKey: "SLTShowLandmarksV17") as? Bool ?? true
    /// LIVE: the sign pills and the word column (word, confidence, top 3); off leaves only the English.
    private var signsOn = UserDefaults.standard.object(forKey: "SLTShowSignsV17") as? Bool ?? true
    private var portraitConstraints: [NSLayoutConstraint] = []
    private var landscapeConstraints: [NSLayoutConstraint] = []
    private var portraitLayout: Bool?
    private let orientationHint = UILabel()
    private var referenceOverlay: UIView?
    private var expandedVideo: UIView?

    // Engine state: touched only on `queue`.
    private var running = false
    private var firstStamp: CMTime?
    private var deadline = 0.0
    private var computeHistory: [Double] = []
    private var frameTimes: [Double] = []
    private var processedFrames = 0
    private var framing = LiveFraming()
    private var brightness: Double?
    private var utterance: [LiveWord] = []
    /// Live English captions (scripts/live_captions_v17.py): committed signs are pushed in order and
    /// every frame ticks the pause timer. Created lazily and used only on LiveReelShared.languageQueue.
    private var captions: LiveCaptions?
    // Pose tracking is owned by captureQueue, independently of model inference.
    private let captureQueue = DispatchQueue(label: "slt.live.capture", qos: .userInteractive)
    private let frameGate = LiveLatestFrameGate<CMSampleBuffer>()
    private let poseScaler = LiveDetectionScaler()
    private var poseVision = LiveVision(auxiliaryInterval: 4)
    private var poseDeadline = 0.0
    private var poseBody: [LivePoint]?
    private var poseFace: [LivePoint]?
    /// The LIVE take being recorded; owned by captureQueue.
    private var recorder: LiveVideoRecorder?
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
    private var videoTakes = 0
    private var fps = 0.0
    private var loadTimer: Timer?
    /// Banner text other than the live words: the finished caption, or a load/frame notice.
    private var bannerMessage: (caption: String, text: String)?
    /// The caption of the utterance being signed (main thread).
    private var captionText: String?
    /// The last finished sentence, while it is on screen (Speak again).
    private var finishedSentence: String?
    /// The current framing hint (main thread), or nil.
    private var advice: String?
    private var captionWidth: CGFloat = 0

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
    /// Fingerspelling on/off (filled blue while on), then (PRACTICE) the score.
    private let scoreChipLabel = UILabel()
    private let railStack = UIStackView()
    private let railScroll = UIScrollView()
    private var railChipKey: [String]?
    private let sentenceDot = UIView()
    private let sentenceLabel = UILabel()
    /// A dark panel behind the centre prompt (it sits over the framing outline before Start).
    private let promptPanel = UIView()
    /// LIVE: Speak again above a finished sentence.
    private let actionsRow = UIStackView()
    private let speakAgainButton = UIButton(type: .system)
    /// Framing: an upper-body outline and a one-line hint (too dark, step back, hands out of frame).
    private let framingGuide = CAShapeLayer()
    private let adviceGlass = GlassView()
    private let adviceLabel = UILabel()
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
    /// The word in progress with its confidence, candidates and (Practice) verdict.
    private let wordStack = UIStackView()
    private var candidateChipKey: [String]?
    private let statsPanel = GlassView(radius: 16)
    private let statsLabel = UILabel()
    /// Round glass ↺: LIVE clears the signs so far, PRACTICE retries the current sign.
    private let resetButton = UIButton(type: .system)
    private let resetGlass = GlassView(light: true)
    /// LIVE Start → Finish (ends the run); PRACTICE Skip → Next.
    private let primaryButton = GradientButton(type: .custom)
    private let hintGlass = GlassView()
    private let hintLabel = UILabel()
    /// The engine's spell mode, for the top-bar status.
    private var spellMode: LiveReelEngine.SpellMode = .words
    private let pipView = GlassView(radius: 18)
    private let pipVideo = UIView()
    private let pipLabel = UILabel()
    private var pipPlayer: AVQueuePlayer?
    private var pipLooper: AVPlayerLooper?
    private var pipLayer: AVPlayerLayer?
    private var referenceTask: Task<Void, Never>?
    private var referenceStatus: NSKeyValueObservation?
    private let speech = AVSpeechSynthesizer()

    init(mode: Mode) {
        self.mode = mode
        if case .practice(let glosses) = mode { targets = glosses }
        super.init(nibName: nil, bundle: nil)
        modalPresentationStyle = .fullScreen
    }

    required init?(coder: NSCoder) { fatalError() }

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .allButUpsideDown }
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation { .portrait }
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
                NSLog("Live models failed to load: %@", status["error"] as? String ?? "")
                self.bannerMessage = ("MODELS", "ATLAS couldn't load its sign models. Close the app and open it again.")
                self.render()
            }
        }
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        UIApplication.shared.isIdleTimerDisabled = true
        setNeedsUpdateOfSupportedInterfaceOrientations()
        view.window?.windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: .portrait))
        if !isPractice, !UserDefaults.standard.bool(forKey: LiveReelShared.liveTipsSeenKey) { presentTips() }
    }

    /// The first-LIVE tips (Settings → Show Live tips again resets them). Shown over the camera.
    func presentTips() {
        guard !view.subviews.contains(where: { $0 is LiveTipsView }) else { return }
        let tips = LiveTipsView()
        tips.onDone = { UserDefaults.standard.set(true, forKey: LiveReelShared.liveTipsSeenKey) }
        tips.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(tips)
        NSLayoutConstraint.activate([
            tips.leadingAnchor.constraint(equalTo: view.leadingAnchor), tips.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tips.topAnchor.constraint(equalTo: view.topAnchor), tips.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        UIApplication.shared.isIdleTimerDisabled = false
        loadTimer?.invalidate()
        referenceTask?.cancel()
        dismissReference()
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
        saveRecording()
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
        let portrait = view.bounds.height > view.bounds.width
        if portraitLayout != portrait {
            NSLayoutConstraint.deactivate(portraitConstraints + landscapeConstraints)
            NSLayoutConstraint.activate(portrait ? portraitConstraints : landscapeConstraints)
            portraitLayout = portrait
            orientationHint.isHidden = !portrait
            view.layoutIfNeeded()
        }
        pipLayer?.frame = (expandedVideo ?? pipVideo).bounds
        framingGuide.frame = view.bounds
        framingGuide.path = Self.framingPath(in: view.safeAreaLayoutGuide.layoutFrame)
        // The caption is fitted to the label's width; refit when that width changes (rotation).
        if !isPractice, sentenceLabel.bounds.width != captionWidth {
            captionWidth = sentenceLabel.bounds.width
            render()
            view.layoutIfNeeded()
            railScroll.setContentOffset(CGPoint(x: max(0, railScroll.contentSize.width - railScroll.bounds.width), y: 0),
                                        animated: false)
        }
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
        updateLandmarkVisibility()
        configureRotation(device)
        queue.async { self.session.startRunning() }
    }

    private func configureRotation(_ device: AVCaptureDevice) {
        rotationObservers = []
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
    }

    private func applyRotation() {
        guard let rotation else { return }
        if let connection = output.connection(with: .video),
           connection.isVideoRotationAngleSupported(rotation.videoRotationAngleForHorizonLevelCapture) {
            let angle = rotation.videoRotationAngleForHorizonLevelCapture
            let resumeRecording = isRunningUI && !switchingCamera
            if connection.videoRotationAngle != angle {
                frameGate.setEnabled(false)
                saveRecording()
            }
            queue.async {
                if connection.videoRotationAngle != angle {
                    self.frameGate.setEnabled(false)
                    self.captureQueue.sync { connection.videoRotationAngle = angle }
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
                        if resumeRecording && self.isRunningUI { self.startRecording() }
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

    @objc private func landmarksTapped() {
        landmarksOn.toggle()
        UserDefaults.standard.set(landmarksOn, forKey: "SLTShowLandmarksV17")
        updateLandmarkVisibility()
        updateMenu()
    }

    private func updateLandmarkVisibility() {
        bones.isHidden = !landmarksOn
        joints.isHidden = !landmarksOn
    }

    /// LIVE: show or hide the sign pills and the word column, leaving only the English.
    @objc private func signsTapped() {
        signsOn.toggle()
        UserDefaults.standard.set(signsOn, forKey: "SLTShowSignsV17")
        updateMenu()
        render()
    }

    /// The ••• menu's items; checkmarks show the current toggles. The toggles keep the menu open
    /// (tap outside to close it); How to use ATLAS closes it and leaves the camera.
    func menuItems() -> [UIMenuElement] {
        let symbol = UIImage.SymbolConfiguration(pointSize: 15, weight: .semibold)
        var items: [UIMenuElement] = [
            UIAction(title: "Show landmarks", image: UIImage(systemName: "point.3.connected.trianglepath.dotted", withConfiguration: symbol),
                     attributes: .keepsMenuPresented, state: landmarksOn ? .on : .off) { [weak self] _ in self?.landmarksTapped() },
        ]
        if !isPractice {
            items.append(UIAction(title: "Show signs", image: UIImage(systemName: "text.bubble", withConfiguration: symbol),
                                  attributes: .keepsMenuPresented, state: signsOn ? .on : .off) { [weak self] _ in self?.signsTapped() })
        }
        items.append(UIMenu(options: .displayInline, children: [
            UIAction(title: "How to use ATLAS", image: UIImage(systemName: "questionmark.circle", withConfiguration: symbol)) { [weak self] _ in
                self?.helpPressed()
            },
        ]))
        return items
    }

    /// Built once: the items are made each time the menu opens, so the menu is never replaced (which
    /// would close it) while it is showing.
    private func buildMenu() {
        menuButton.menu = UIMenu(children: [UIDeferredMenuElement.uncached { [weak self] completion in
            completion(self?.menuItems() ?? [])
        }])
        updateMenu()
    }

    /// After a toggle: refresh the open menu in place (new checkmarks, still open) and the button's value.
    private func updateMenu() {
        menuButton.contextMenuInteraction?.updateVisibleMenu { [weak self] menu in
            menu.replacingChildren(self?.menuItems() ?? menu.children)
        }
        menuButton.accessibilityValue = [landmarksOn ? "Landmarks on" : "Landmarks off",
                                         isPractice ? nil : signsOn ? "Signs on" : "Signs off"]
            .compactMap { $0 }.joined(separator: ", ")
    }

    @objc private func flipCamera() {
        guard !isPractice, !switchingCamera, !finishing else { return }
        let position: AVCaptureDevice.Position = cameraPosition == .front ? .back : .front
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        switchingCamera = true
        flipButton.isEnabled = false
        frameGate.setEnabled(false)
        bones.path = nil; joints.path = nil
        saveRecording()
        queue.async {
            self.session.beginConfiguration()
            let old = self.session.inputs
            old.forEach { self.session.removeInput($0) }
            let switched = self.session.canAddInput(input)
            if switched { self.session.addInput(input) }
            else { old.forEach { self.session.addInput($0) } }
            if let connection = self.output.connection(with: .video) {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
            self.session.commitConfiguration()
            LiveReelShared.engine?.reset()
            self.resetPoseTracking()
            self.deadline = 0
            DispatchQueue.main.async {
                if switched { self.cameraPosition = position }
                if let connection = self.previewLayer.connection {
                    connection.automaticallyAdjustsVideoMirroring = false
                    connection.isVideoMirrored = self.cameraPosition == .front
                }
                if switched { self.configureRotation(device) }
                self.previewText = nil; self.centerPreview = nil; self.top3 = []
                self.finishGestureProgress = nil
                self.switchingCamera = false
                self.flipButton.isEnabled = true
                self.queue.async { self.frameGate.setEnabled(self.running) }
                if self.isRunningUI { self.startRecording() }
                if !switched { self.bannerMessage = ("CAMERA", "Could not switch cameras. Try again.") }
                self.render()
            }
        }
    }

    /// On `queue`.
    private func startStream() {
        // Manual Letters override stays off in both modes, regardless of old saved preferences. The
        // FINGERSPELL sign is off; NAME spells only in LIVE with the Settings switch on.
        LiveReelShared.engine?.lettersAlways = false
        LiveReelShared.engine?.fingerspellSign = false
        LiveReelShared.engine?.nameLetters = !isPractice && UserDefaults.standard.bool(forKey: LiveReelShared.nameSpellingKey)
        framing.reset(); brightness = nil
        LiveReelShared.engine?.reset()
        firstStamp = nil
        deadline = 0
        frameTimes = []
        resetPoseTracking()
        frameGate.setEnabled(true)
        running = true
        events.append(["event": "start", "time": Date().timeIntervalSince1970])
        LiveReelShared.engine?.startTrace()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        recorder?.append(sampleBuffer)
        guard let generation = frameGate.activeGeneration,
              let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if frameGate.offer(sampleBuffer, generation: generation) { queue.async { self.processNextCapture() } }
        let captureAngle = connection.videoRotationAngle
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
                self.drawSkeleton(observation, body: body, face: face, captureAngle: captureAngle)
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
            if !isPractice {
                utterance += step.committed
                caption(step.committed, seconds: seconds, held: step.finishProgress != nil)
            }
            if let spellEvent = step.spellEvent {
                events.append(["event": "spell_mode", "seconds": seconds, "change": spellEvent])
            }
            for w in step.committed {
                events.append(["event": "word", "seconds": seconds, "word": w.json])
                if !isPractice { sessionGlosses.append(w.gloss) }
            }
            let o = step.observation
            if processedFrames % 5 == 1 { brightness = Self.brightness(frame) }
            let advice = step.finishProgress == nil ? framing.update(o, brightness: brightness, seconds: seconds) : nil
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
                self.advice = advice
                self.finishGestureProgress = step.finishProgress
                self.spellMode = step.spellMode
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
            NSLog("Live frame error: %@", error.localizedDescription)
            DispatchQueue.main.async {
                self.bannerMessage = ("NOTICE", "The camera skipped a moment. ATLAS is still translating.")
                self.render()
            }
        }
    }

    // MARK: - Controls

    /// LIVE Start/Stop (Stop finishes the utterance). PRACTICE Stop pauses the round; Start resumes it
    /// through the same startStream() the round opens with.
    @objc private func toggleStart() {
        guard loaded, !switchingCamera else { return }
        let starting = !isRunningUI
        isRunningUI = starting
        if isPractice && !starting { previewText = nil; centerPreview = nil; top3 = [] }
        if starting { startRecording() } else { saveRecording() }
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

    @objc private func resetPressed() {
        if isRunningUI { startRecording() }
        queue.async {
            LiveReelShared.engine?.reset()
            self.frameGate.setEnabled(self.running)
            self.resetPoseTracking()
            self.utterance = []
            LiveReelShared.languageQueue.async { self.captions?.reset() }
            self.events.append(["event": "reset"])
            DispatchQueue.main.async {
                self.speech.stopSpeaking(at: .immediate)
                self.rail = []; self.previewText = nil; self.committedWord = nil; self.captionText = nil
                self.finishedSentence = nil
                self.centerPreview = nil; self.top3 = []; self.finishing = false
                self.finishGestureProgress = nil
                self.bannerMessage = nil
                self.render()
            }
        }
    }

    /// The big button: LIVE Start, then Finish (stop, speak the sentence, save the take);
    /// PRACTICE Skip, or Next after a match.
    @objc private func primaryPressed() {
        if isPractice {
            awaitingNext ? nextPressed() : skipPressed()
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
        showHint(isPractice ? "Try this sign again" : "Clear the signs so far")
    }

    /// LIVE returns Home; PRACTICE returns to its setup (or the result page once the round is done).
    @objc private func backPressed() {
        close(navigate: isPractice ? "PRACTICE" : "HOME")
    }

    /// ?: leave (saving like Back) for the manual section about this screen.
    @objc private func helpPressed() {
        close(navigate: "MANUAL", section: isPractice ? "practice" : "screen")
    }

    private func close(navigate page: String, section: String? = nil) {
        guard presentingViewController != nil, !isBeingDismissed else { return }
        closeResult = ["navigate": page]
        if let section { closeResult["section"] = section }
        if isPractice {
            closeResult.merge([
                "score": practiceScore, "attempts": practiceAttempts, "total": targets.count,
                "missed": practiceMissed, "completed": roundDone,
            ]) { $1 }
        }
        dismiss(animated: true)
    }

    /// The utterance's captions, created on first use (the models load after the recognizer).
    /// languageQueue only; nil while the caption models are still loading or failed to load.
    private func liveCaptions() -> LiveCaptions? {
        if captions == nil, let models = LiveReelShared.captions { captions = LiveCaptions(models) }
        return captions
    }

    /// On `queue`: push committed signs, then (unless the Finish gesture is held) tick the pause timer.
    private func caption(_ committed: [LiveWord], seconds: Double, held: Bool) {
        LiveReelShared.languageQueue.async {
            guard let captions = self.liveCaptions() else { return }
            var updates: [(kind: String, text: String)] = []
            for w in committed { updates.append(("sign", captions.push(w.gloss, score: w.score, seconds: seconds))) }
            if !held, let text = captions.tick(seconds) { updates.append(("end_of_turn", text)) }
            guard let latest = updates.last?.text else { return }
            self.queue.async {
                for u in updates { self.events.append(["event": "caption", "kind": u.kind, "seconds": seconds, "text": u.text]) }
            }
            DispatchQueue.main.async {
                guard !self.finishing else { return }
                self.captionText = latest.isEmpty ? nil : latest
                self.render()
            }
        }
    }

    /// On `queue`: flush the open segment, then render and speak the utterance.
    private func finishUtterance(reason: String) {
        guard let engine = LiveReelShared.engine else { return }
        let tail = (try? engine.finish()) ?? []
        utterance += tail
        sessionGlosses += tail.map(\.gloss)
        let words = utterance
        utterance = []
        let seconds = tail.last?.commitSeconds ?? 0
        events.append(["event": reason, "words": words.map(\.json)])
        DispatchQueue.main.async {
            self.apply(committed: tail, preview: nil, pending: nil)
            self.previewText = nil; self.centerPreview = nil
            if words.isEmpty {
                self.bannerMessage = ("TRANSLATION", "No signs yet. Sign first, then tap Finish.")
                self.captionText = nil
            } else {
                self.finishing = true
                self.bannerMessage = nil
            }
            self.render()
        }
        LiveReelShared.languageQueue.async {
            let started = Date()
            let captions = self.liveCaptions()
            for w in tail { captions?.push(w.gloss, score: w.score, seconds: seconds) }
            guard !words.isEmpty else { captions?.reset(); return }
            // Without caption models (not yet loaded or failed) the signs are shown as written words.
            let sentence = captions?.finish() ?? words.map {
                $0.gloss.hasPrefix("fs-") ? LiveStage3.spelledText($0.gloss) : $0.gloss.lowercased()
            }.joined(separator: " ") + "."
            captions?.reset()
            let ms = 1000 * Date().timeIntervalSince(started)
            self.queue.async {
                self.events.append(["event": "sentence", "sentence": sentence, "ms": ms,
                                    "rendering_mode": captions == nil ? "literal" : "live_captions",
                                    "live_captions": LiveReelShared.captions?.name ?? LiveReelShared.captionsError ?? "unloaded"])
                self.sessionSentences.append(sentence)
            }
            DispatchQueue.main.async {
                self.finishing = false
                self.captionText = nil
                self.finishedSentence = sentence
                self.bannerMessage = ("TRANSLATION", sentence)
                self.rail = []
                self.say(sentence)
                self.render()
            }
        }
    }

    // MARK: - Practice (AppShell._check_practice / skip_practice)

    private var target: String? { !roundDone && practiceIndex < targets.count ? targets[practiceIndex] : nil }

    private func judge(_ committed: [LiveWord]) {
        guard referenceOverlay == nil, !awaitingNext, let target, let first = committed.first else { return }
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
                guard let self, self.referenceOverlay == nil, self.awaitingNext, self.practiceIndex == index else { return }
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
        referenceTask?.cancel()
        referenceStatus = nil
        dismissReference()
        pipPlayer?.pause()
        pipLooper?.disableLooping()
        pipPlayer?.removeAllItems()
        pipLayer?.removeFromSuperlayer()
        pipPlayer = nil; pipLooper = nil; pipLayer = nil
        guard let target else { pipView.isHidden = true; return }
        pipView.isHidden = false
        pipLabel.text = "Loading…"
        let key = FlutterDartProject.lookupKey(forAsset: "assets/gloss_examples/\(target).mp4")
        guard let path = Bundle.main.path(forResource: key, ofType: nil) else {
            pipLabel.text = "Unavailable"
            return
        }
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        // AVPlayerLooper requires duration to be loaded; never block the camera UI on asset loading.
        referenceTask = Task { @MainActor [weak self] in
            do {
                let (playable, duration) = try await asset.load(.isPlayable, .duration)
                guard !Task.isCancelled, let self, self.target == target else { return }
                guard playable, duration.seconds.isFinite, duration.seconds > 0 else {
                    self.pipLabel.text = "Unavailable"
                    return
                }
                let player = AVQueuePlayer()
                player.isMuted = true
                let looper = AVPlayerLooper(player: player, templateItem: AVPlayerItem(asset: asset))
                let layer = AVPlayerLayer(player: player)
                layer.videoGravity = .resizeAspect
                layer.frame = self.pipVideo.bounds
                self.pipVideo.layer.insertSublayer(layer, at: 0)
                self.pipPlayer = player; self.pipLooper = looper; self.pipLayer = layer
                self.pipLabel.text = "Reference"
                self.referenceStatus = looper.observe(\.status, options: [.initial, .new]) { [weak self] looper, _ in
                    guard looper.status == .failed else { return }
                    DispatchQueue.main.async {
                        guard let self, self.pipLooper === looper else { return }
                        self.pipLabel.text = "Unavailable"
                        NSLog("Practice reference %@: %@", target, String(describing: looper.error))
                    }
                }
                // Reference clips play at 0.75× so the handshape and movement are easy to follow.
                player.defaultRate = 0.75
                player.play()
            } catch {
                guard !Task.isCancelled, let self, self.target == target else { return }
                self.pipLabel.text = "Unavailable"
                NSLog("Practice reference %@: %@", target, error.localizedDescription)
            }
        }
    }

    @objc private func expandReference() {
        guard pipLayer != nil, referenceOverlay == nil else { return }
        let overlay = UIView()
        overlay.backgroundColor = UIColor.black.withAlphaComponent(0.8)
        overlay.accessibilityViewIsModal = true
        let backdrop = UIButton(type: .custom)
        backdrop.accessibilityLabel = "Close reference video"
        backdrop.addTarget(self, action: #selector(dismissReference), for: .touchUpInside)
        let video = UIView()
        video.accessibilityIdentifier = "expandedReferenceVideo"
        video.backgroundColor = .black
        video.layer.cornerRadius = 16
        video.clipsToBounds = true
        let close = UIButton(type: .system)
        close.setImage(UIImage(systemName: "xmark.circle.fill"), for: .normal)
        close.tintColor = .white
        close.accessibilityLabel = "Close reference video"
        close.addTarget(self, action: #selector(dismissReference), for: .touchUpInside)
        for child in [backdrop, video, close] {
            child.translatesAutoresizingMaskIntoConstraints = false
            overlay.addSubview(child)
        }
        overlay.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(overlay)
        let guide = overlay.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            overlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlay.topAnchor.constraint(equalTo: view.topAnchor),
            overlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            backdrop.leadingAnchor.constraint(equalTo: overlay.leadingAnchor),
            backdrop.trailingAnchor.constraint(equalTo: overlay.trailingAnchor),
            backdrop.topAnchor.constraint(equalTo: overlay.topAnchor),
            backdrop.bottomAnchor.constraint(equalTo: overlay.bottomAnchor),
            video.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 20),
            video.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -20),
            video.topAnchor.constraint(equalTo: guide.topAnchor, constant: 64),
            video.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -40),
            close.trailingAnchor.constraint(equalTo: video.trailingAnchor),
            close.bottomAnchor.constraint(equalTo: video.topAnchor, constant: -8),
            close.widthAnchor.constraint(equalToConstant: 44),
            close.heightAnchor.constraint(equalToConstant: 44),
        ])
        referenceOverlay = overlay
        expandedVideo = video
        pipLayer?.removeFromSuperlayer()
        video.layer.addSublayer(pipLayer!)
        view.setNeedsLayout()
        UIAccessibility.post(notification: .screenChanged, argument: close)
    }

    @objc private func dismissReference() {
        guard referenceOverlay != nil else { return }
        pipLayer?.removeFromSuperlayer()
        if let pipLayer { pipVideo.layer.addSublayer(pipLayer) }
        expandedVideo = nil
        referenceOverlay?.removeFromSuperview()
        referenceOverlay = nil
        pipLayer?.frame = pipVideo.bounds
        UIAccessibility.post(notification: .screenChanged, argument: pipVideo)
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
            finishedSentence = nil
        }
        for w in committed {
            rail.append(Self.display(w.gloss))
            committedWord = (Self.display(w.gloss), w.score, Date())
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
            spellMode == .spelling ? ("Spelling…", HUD.blue) :
            !isPractice && spellMode == .name ? ("Name? Letters on", HUD.blue) :
            centerPreview != nil ? ("Reading sign…", HUD.gold) :
            hasHands ? ("Signing", HUD.white) : ("Watching", dim)
        statusLabel.text = state
        statusLabel.textColor = stateColor
        liveDot.backgroundColor = isRunningUI ? HUD.green : HUD.white.withAlphaComponent(0.4)
        if isPractice { scoreChipLabel.text = "Score \(practiceScore) / \(targets.count)" }
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
            // Keep the newest pill in view.
            railScroll.layoutIfNeeded()
            railScroll.setContentOffset(CGPoint(x: max(0, railScroll.contentSize.width - railScroll.bounds.width), y: 0),
                                        animated: false)
        }

        // Live: the finished sentence (or a notice) under the top bar.
        if !isPractice {
            sentenceLabel.text = fittedCaption(finishing ? (captionText ?? "Finishing…") : bannerMessage?.text ?? captionText)
            sentenceLabel.textColor = finishing ? HUD.white.withAlphaComponent(0.6) : HUD.white
            sentenceDot.backgroundColor = speech.isSpeaking ? HUD.green : HUD.white.withAlphaComponent(0.35)
            sentenceDot.isHidden = sentenceLabel.text == nil
            actionsRow.isHidden = finishedSentence == nil || finishing || captionText != nil
        }

        // Framing: the hint while running; the outline before Start, or while no one is in view.
        let shownAdvice = isRunningUI && !finishing ? advice : nil
        adviceLabel.text = shownAdvice
        adviceGlass.isHidden = shownAdvice == nil
        framingGuide.isHidden = !loaded || (isRunningUI ? !(shownAdvice?.hasPrefix("Step") ?? false) : isPractice)

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
        barFillWidth.constant = barWidth * CGFloat(max(0, min(1, score)))
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
                                    : ("Ready to translate", "Line yourself up with the outline, then tap Start.")
            } else if isPractice, target != nil {
                prompt = ("Your turn", "Watch the reference, then sign it.")
            }
        }
        if adviceLabel.text != nil { prompt = nil }   // the framing hint takes the centre
        promptTitle.text = prompt?.0
        promptDetail.text = prompt?.1
        promptTitle.superview?.isHidden = prompt == nil
        promptPanel.isHidden = prompt == nil

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
        if !isPractice {
            railScroll.isHidden = !signsOn
            wordStack.isHidden = !signsOn
        }

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
                // Finish always ends the run (and saves the take), even with nothing recognized yet.
                stylePrimary(finishing ? .disabled : .primary, finishing ? "Working…" : "Finish", "checkmark")
            }
        }
        resetButton.isEnabled = canReset
        resetGlass.alpha = canReset ? 1 : 0.4
    }

    /// A glass pill with an icon and a title (Speak again).
    private func glassPill(_ button: UIButton, _ title: String, _ symbol: String, _ action: Selector) -> UIView {
        let glass = GlassView(radius: 20)
        var configuration = UIButton.Configuration.plain()
        configuration.title = title
        configuration.image = UIImage(systemName: symbol, withConfiguration: UIImage.SymbolConfiguration(pointSize: 14, weight: .semibold))
        configuration.imagePadding = 6
        configuration.baseForegroundColor = HUD.white
        configuration.contentInsets = NSDirectionalEdgeInsets(top: 0, leading: 14, bottom: 0, trailing: 14)
        configuration.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer {
            var attributes = $0; attributes.font = HUD.font(15, .semibold); return attributes
        }
        button.configuration = configuration
        button.addTarget(self, action: action, for: .touchUpInside)
        inset(button, in: glass.contentView, 0, 0)
        glass.translatesAutoresizingMaskIntoConstraints = false
        glass.heightAnchor.constraint(equalToConstant: 44).isActive = true
        return glass
    }

    /// LIVE: read the finished sentence aloud again.
    @objc private func speakAgainPressed() {
        guard let sentence = finishedSentence else { return }
        speech.stopSpeaking(at: .immediate)
        say(sentence)
    }

    private func showHint(_ text: String) {
        hintLabel.text = text
        hintGlass.layer.removeAllAnimations()
        hintGlass.alpha = 1
        UIView.animate(withDuration: 0.4, delay: 1.6, options: [.beginFromCurrentState]) { self.hintGlass.alpha = 0 }
    }

    /// Mean luma (0–255) over a coarse grid of a BGRA frame; nil for other formats.
    static func brightness(_ frame: CVPixelBuffer) -> Double? {
        guard CVPixelBufferGetPixelFormatType(frame) == kCVPixelFormatType_32BGRA else { return nil }
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(frame)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
        let row = CVPixelBufferGetBytesPerRow(frame)
        var total = 0.0, count = 0
        for y in stride(from: height / 24, to: height, by: max(1, height / 12)) {
            for x in stride(from: width / 32, to: width, by: max(1, width / 16)) {
                let p = base + y * row + x * 4
                total += 0.114 * Double(p[0]) + 0.587 * Double(p[1]) + 0.299 * Double(p[2])
                count += 1
            }
        }
        return count > 0 ? total / Double(count) : nil
    }

    /// A dashed head-and-shoulders outline centred in `area`, showing where the signer should be.
    static func framingPath(in area: CGRect) -> CGPath {
        let size = min(area.height * 0.62, area.width * 0.8)
        let cx = area.midX, top = area.midY - size * 0.5
        let radius = size * 0.13
        let path = UIBezierPath(arcCenter: CGPoint(x: cx, y: top + radius), radius: radius,
                                startAngle: 0, endAngle: 2 * .pi, clockwise: true)
        let shoulders = top + 2 * radius + size * 0.1, half = size * 0.42, neck = size * 0.12
        path.move(to: CGPoint(x: cx - half, y: top + size))
        path.addLine(to: CGPoint(x: cx - half, y: shoulders + size * 0.14))
        path.addQuadCurve(to: CGPoint(x: cx - neck, y: shoulders), controlPoint: CGPoint(x: cx - half, y: shoulders))
        path.addLine(to: CGPoint(x: cx + neck, y: shoulders))
        path.addQuadCurve(to: CGPoint(x: cx + half, y: shoulders + size * 0.14), controlPoint: CGPoint(x: cx + half, y: shoulders))
        path.addLine(to: CGPoint(x: cx + half, y: top + size))
        return path.cgPath
    }

    /// The caption as shown: whole words dropped from the front (after "…") until it fits the label's
    /// lines, so the newest words stay visible.
    private func fittedCaption(_ text: String?) -> String? {
        guard let text, let font = sentenceLabel.font, sentenceLabel.bounds.width > 0 else { return text }
        let width = sentenceLabel.bounds.width
        let limit = font.lineHeight * CGFloat(sentenceLabel.numberOfLines) + 1
        func fits(_ candidate: String) -> Bool {
            (candidate as NSString).boundingRect(with: CGSize(width: width, height: .greatestFiniteMagnitude),
                                                 options: .usesLineFragmentOrigin, attributes: [.font: font],
                                                 context: nil).height <= limit
        }
        if fits(text) { return text }
        var words = text.split(separator: " ")
        while words.count > 1 {
            words.removeFirst()
            let trimmed = "…" + words.joined(separator: " ")
            if fits(trimmed) { return trimmed }
        }
        return text
    }

    /// Undo capture rotation before converting through the preview layer.
    static func captureDevicePoint(_ point: CGPoint, rotation: CGFloat) -> CGPoint {
        switch Int(rotation.rounded()) % 360 {
        case 90: return CGPoint(x: point.y, y: 1 - point.x)
        case 180: return CGPoint(x: 1 - point.x, y: 1 - point.y)
        case 270: return CGPoint(x: 1 - point.y, y: point.x)
        default: return point
        }
    }

    private func drawSkeleton(_ o: LiveObservation, body: [LivePoint]?, face: [LivePoint]?, captureAngle: CGFloat) {
        func point(_ p: LivePoint) -> CGPoint {
            let sensor = Self.captureDevicePoint(CGPoint(x: CGFloat(p.x), y: CGFloat(p.y)), rotation: captureAngle)
            return previewLayer.layerPointConverted(fromCaptureDevicePoint: sensor)
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
    private func glassIcon(_ button: UIButton, _ symbol: String, _ label: String, _ action: Selector?) -> UIView {
        let glass = GlassView()
        button.setImage(UIImage(systemName: symbol, withConfiguration: UIImage.SymbolConfiguration(pointSize: 17, weight: .semibold)),
                        for: .normal)
        button.tintColor = HUD.white
        button.accessibilityLabel = label
        if let action { button.addTarget(self, action: action, for: .touchUpInside) }
        inset(button, in: glass.contentView, 0, 0)
        glass.translatesAutoresizingMaskIntoConstraints = false
        glass.widthAnchor.constraint(equalToConstant: 44).isActive = true
        glass.heightAnchor.constraint(equalToConstant: 44).isActive = true
        return glass
    }

    private func buildInterface() {
        let guide = view.safeAreaLayoutGuide

        // Top bar: back · mode/status · (Live) flip · ••• menu.
        let back = glassIcon(backButton, "chevron.left", "Back", #selector(backPressed))
        let menu = glassIcon(menuButton, "ellipsis", "More options", nil)
        menuButton.showsMenuAsPrimaryAction = true
        menuButton.accessibilityIdentifier = "cameraMenu"
        let flip = glassIcon(flipButton, "arrow.triangle.2.circlepath.camera", "Flip camera", #selector(flipCamera))
        updateLandmarkVisibility()
        buildMenu()
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
        let scoreChip = GlassView()
        text(scoreChipLabel, 13, .semibold, HUD.white)
        inset(scoreChipLabel, in: scoreChip.contentView, 12, 14)
        let spacer = UIView()
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        modeStack.setContentHuggingPriority(.required, for: .horizontal)
        let topBar = UIStackView(arrangedSubviews: [back, modeStack, spacer] + (isPractice ? [scoreChip] : []))
        topBar.spacing = isPractice ? 14 : 8
        statusLabel.lineBreakMode = .byTruncatingTail
        statusLabel.widthAnchor.constraint(lessThanOrEqualTo: modeStack.widthAnchor).isActive = true
        topBar.alignment = .center
        let tools = UIStackView(arrangedSubviews: (isPractice ? [] : [flip]) + [menu])
        tools.spacing = 8
        tools.alignment = .center
        text(orientationHint, 11, .medium, HUD.white, align: .center)
        orientationHint.text = "Landscape recommended"
        orientationHint.shadowed()

        // Live: word pills and the finished sentence.
        railScroll.showsHorizontalScrollIndicator = false
        railStack.translatesAutoresizingMaskIntoConstraints = false
        railScroll.addSubview(railStack)
        NSLayoutConstraint.activate([
            railStack.leadingAnchor.constraint(equalTo: railScroll.contentLayoutGuide.leadingAnchor),
            railStack.trailingAnchor.constraint(equalTo: railScroll.contentLayoutGuide.trailingAnchor),
            railStack.topAnchor.constraint(equalTo: railScroll.contentLayoutGuide.topAnchor),
            railStack.bottomAnchor.constraint(equalTo: railScroll.contentLayoutGuide.bottomAnchor),
            railStack.heightAnchor.constraint(equalTo: railScroll.frameLayoutGuide.heightAnchor),
        ])
        railStack.axis = .horizontal
        railStack.spacing = 8
        railStack.alignment = .center
        sentenceDot.layer.cornerRadius = 9
        // LIVE: the English is the largest text on screen (the recognized word sits small, top right).
        sentenceLabel.font = UIFontMetrics(forTextStyle: .title1).scaledFont(for: HUD.font(32, .bold), maximumPointSize: 44)
        sentenceLabel.adjustsFontForContentSizeCategory = true
        sentenceLabel.textColor = HUD.white
        sentenceLabel.numberOfLines = 4
        // A long caption is trimmed from the front, whole words at a time (fittedCaption).
        sentenceLabel.lineBreakMode = .byWordWrapping
        sentenceLabel.shadowed(0.9, radius: 4)

        speakAgainButton.accessibilityIdentifier = "speakAgain"
        actionsRow.addArrangedSubview(glassPill(speakAgainButton, "Speak again", "speaker.wave.2.fill", #selector(speakAgainPressed)))
        actionsRow.spacing = 8
        actionsRow.isHidden = true
        framingGuide.strokeColor = HUD.white.withAlphaComponent(0.45).cgColor
        framingGuide.fillColor = UIColor.clear.cgColor
        framingGuide.lineWidth = 2
        framingGuide.lineDashPattern = [8, 6]
        framingGuide.isHidden = true
        if let previewLayer { view.layer.insertSublayer(framingGuide, above: previewLayer) } else { view.layer.insertSublayer(framingGuide, at: 0) }
        text(adviceLabel, 15, .semibold, HUD.white, align: .center)
        adviceLabel.numberOfLines = 2
        adviceGlass.accessibilityIdentifier = "framingAdvice"
        inset(adviceLabel, in: adviceGlass.contentView, 10, 16)
        adviceGlass.isHidden = true

        // Practice: target banner and tappable reference clip.
        text(bannerCaption, 11, .semibold, HUD.white.withAlphaComponent(0.75), align: .center)
        bannerCaption.numberOfLines = 2
        text(bannerValue, 24, .semibold, HUD.white, align: .center)
        bannerValue.adjustsFontSizeToFitWidth = true
        bannerValue.minimumScaleFactor = 0.6
        // On a narrow phone the banner gives way, not the top-bar chips.
        for v in [bannerCaption, bannerValue] { v.setContentCompressionResistancePriority(.defaultLow, for: .horizontal) }
        let bannerStack = UIStackView(arrangedSubviews: [bannerCaption, bannerValue])
        bannerStack.axis = .vertical
        bannerStack.spacing = 2
        inset(bannerStack, in: bannerBox.contentView, 10, 20)
        pipVideo.backgroundColor = HUD.night
        pipVideo.layer.cornerRadius = 12
        pipVideo.layer.cornerCurve = .continuous
        pipVideo.clipsToBounds = true
        pipVideo.isAccessibilityElement = true
        pipVideo.accessibilityLabel = "Enlarge reference video"
        pipVideo.accessibilityTraits = .button
        pipVideo.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(expandReference)))
        text(pipLabel, 12, .semibold, HUD.white.withAlphaComponent(0.8))
        pipLabel.text = "Reference"
        let pipStack = UIStackView(arrangedSubviews: [pipVideo, pipLabel])
        pipStack.axis = .vertical
        pipStack.spacing = 4
        inset(pipStack, in: pipView.contentView, 8, 8)

        // The word in progress, confidence bar, candidates and (Practice) verdict pill:
        // bottom-left in PRACTICE, a small column top-right in LIVE.
        bigWord.font = HUD.font(isPractice ? 34 : 22, .heavy)
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
        candidatesStack.alignment = isPractice ? .leading : .trailing
        for v in [feedbackPill, bigWord, confidence, candidatesStack] { wordStack.addArrangedSubview(v) }
        wordStack.axis = .vertical
        wordStack.spacing = 4
        wordStack.alignment = isPractice ? .leading : .trailing
        wordStack.setCustomSpacing(10, after: confidence)
        wordStack.accessibilityIdentifier = "recognizedWord"
        sentenceLabel.accessibilityIdentifier = "englishCaption"

        // Centre prompt (only while nothing is being read).
        text(promptTitle, 24, .semibold, HUD.white, align: .center)
        text(promptDetail, 13, .regular, HUD.white, align: .center)
        promptDetail.numberOfLines = 2
        promptTitle.shadowed(0.8, radius: 4)
        promptDetail.shadowed(0.8, radius: 3)
        let promptStack = UIStackView(arrangedSubviews: [promptTitle, promptDetail])
        promptStack.axis = .vertical
        promptStack.spacing = 8
        promptPanel.backgroundColor = HUD.night.withAlphaComponent(0.7)
        promptPanel.layer.cornerRadius = 20
        promptPanel.layer.cornerCurve = .continuous
        promptPanel.isUserInteractionEnabled = false

        // Internal diagnostics (not exposed in the camera toolbar).
        statsPanel.isHidden = true
        statsLabel.numberOfLines = 0
        statsLabel.attributedText = stats([("FPS", "--"), ("FRAME", "--"), ("HANDS", "--"), ("FRAMES", "0")])
        inset(statsLabel, in: statsPanel.contentView, 10, 12)

        // Bottom controls: ↺ then the primary action.
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
        controls.accessibilityIdentifier = "cameraControls"
        primaryButton.accessibilityIdentifier = "cameraPrimaryAction"
        bannerBox.accessibilityIdentifier = "practiceTarget"
        pipView.accessibilityIdentifier = "practiceReference"
        controls.spacing = 14
        controls.alignment = .center
        text(hintLabel, 12, .semibold, HUD.white)
        inset(hintLabel, in: hintGlass.contentView, 7, 12)
        hintGlass.alpha = 0
        hintGlass.isUserInteractionEnabled = false

        var views: [UIView] = [promptPanel, promptStack, adviceGlass, wordStack, statsPanel, controls, hintGlass]
        views += isPractice ? [pipView, bannerBox] : [railScroll, sentenceDot, sentenceLabel, actionsRow]
        views += [topBar, tools, orientationHint]
        for v in views {
            v.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(v)
        }

        NSLayoutConstraint.activate([
            topBar.topAnchor.constraint(equalTo: guide.topAnchor, constant: 12),
            topBar.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 12),
            topBar.heightAnchor.constraint(equalToConstant: 44),
            tools.heightAnchor.constraint(equalToConstant: 44),
            orientationHint.topAnchor.constraint(equalTo: tools.bottomAnchor, constant: 6),
            orientationHint.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            scoreChip.heightAnchor.constraint(equalToConstant: 44),
            liveDot.widthAnchor.constraint(equalToConstant: 8),
            liveDot.heightAnchor.constraint(equalToConstant: 8),

            resetGlass.widthAnchor.constraint(equalToConstant: 56),
            resetGlass.heightAnchor.constraint(equalToConstant: 56),
            primaryButton.widthAnchor.constraint(equalToConstant: 160),
            primaryButton.heightAnchor.constraint(equalToConstant: 60),
            controls.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),
            hintGlass.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -8),
            hintGlass.trailingAnchor.constraint(equalTo: controls.trailingAnchor),

            promptStack.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            promptStack.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            promptStack.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.8),
            promptPanel.leadingAnchor.constraint(equalTo: promptStack.leadingAnchor, constant: -20),
            promptPanel.trailingAnchor.constraint(equalTo: promptStack.trailingAnchor, constant: 20),
            promptPanel.topAnchor.constraint(equalTo: promptStack.topAnchor, constant: -14),
            promptPanel.bottomAnchor.constraint(equalTo: promptStack.bottomAnchor, constant: 14),
            adviceGlass.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            adviceGlass.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            adviceGlass.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.85),

            barTrack.widthAnchor.constraint(equalToConstant: barWidth),
            barTrack.heightAnchor.constraint(equalToConstant: 5),
            barFill.leadingAnchor.constraint(equalTo: barTrack.leadingAnchor),
            barFill.topAnchor.constraint(equalTo: barTrack.topAnchor),
            barFill.bottomAnchor.constraint(equalTo: barTrack.bottomAnchor),
            barFillWidth,


            // Diagnostics: above the controls.
            statsPanel.trailingAnchor.constraint(equalTo: guide.trailingAnchor),
            statsPanel.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -12),
        ])
        portraitConstraints = [
            topBar.trailingAnchor.constraint(equalTo: isPractice ? guide.trailingAnchor : tools.leadingAnchor, constant: isPractice ? -12 : -8),
            tools.topAnchor.constraint(equalTo: isPractice ? topBar.bottomAnchor : topBar.topAnchor, constant: isPractice ? 8 : 0),
            tools.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -12),
            (isPractice ? primaryButton : controls).centerXAnchor.constraint(equalTo: guide.centerXAnchor),
        ]
        landscapeConstraints = [
            topBar.trailingAnchor.constraint(equalTo: tools.leadingAnchor, constant: -16),
            tools.topAnchor.constraint(equalTo: topBar.topAnchor),
            tools.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -12),
            controls.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -12),
        ]
        if isPractice {
            NSLayoutConstraint.activate([
                pipVideo.widthAnchor.constraint(equalToConstant: 110),
                pipVideo.heightAnchor.constraint(equalToConstant: 94),
            ])
            NSLayoutConstraint.activate([wordStack.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 16)])
            portraitConstraints += [
                wordStack.bottomAnchor.constraint(equalTo: bannerBox.topAnchor, constant: -16),
                bigWord.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.85),
                bannerBox.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
                bannerBox.widthAnchor.constraint(equalTo: guide.widthAnchor, constant: -24),
                bannerBox.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -12),
                pipView.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
                pipView.topAnchor.constraint(equalTo: tools.bottomAnchor, constant: 28),
            ]
            landscapeConstraints += [
                wordStack.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),
                bigWord.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.34),
                bannerBox.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 12),
                bannerBox.trailingAnchor.constraint(equalTo: pipView.leadingAnchor, constant: -12),
                bannerBox.topAnchor.constraint(equalTo: tools.bottomAnchor, constant: 8),
                pipView.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -12),
                pipView.topAnchor.constraint(equalTo: bannerBox.topAnchor),
            ]
            loadReference()
        } else {
            // LIVE: sign pills top-left beside the word column (top-right); the English bottom-left.
            NSLayoutConstraint.activate([
                wordStack.topAnchor.constraint(equalTo: tools.bottomAnchor, constant: 28),
                wordStack.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -12),
                wordStack.widthAnchor.constraint(equalToConstant: 150),
                bigWord.widthAnchor.constraint(lessThanOrEqualTo: wordStack.widthAnchor),
                railScroll.topAnchor.constraint(equalTo: tools.bottomAnchor, constant: 28),
                railScroll.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 12),
                railScroll.trailingAnchor.constraint(equalTo: wordStack.leadingAnchor, constant: -8),
                railScroll.heightAnchor.constraint(equalToConstant: 30),
                sentenceDot.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 12),
                sentenceDot.topAnchor.constraint(equalTo: sentenceLabel.topAnchor, constant: 11),
                sentenceDot.widthAnchor.constraint(equalToConstant: 18),
                sentenceDot.heightAnchor.constraint(equalToConstant: 18),
                sentenceLabel.leadingAnchor.constraint(equalTo: sentenceDot.trailingAnchor, constant: 12),
                // A long caption drops its oldest words rather than running into the pills.
                sentenceLabel.topAnchor.constraint(greaterThanOrEqualTo: railScroll.bottomAnchor, constant: 12),
                actionsRow.leadingAnchor.constraint(equalTo: sentenceDot.leadingAnchor),
                actionsRow.bottomAnchor.constraint(equalTo: sentenceLabel.topAnchor, constant: -10),
            ])
            portraitConstraints += [
                sentenceLabel.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -20),
                sentenceLabel.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -16),
            ]
            // Landscape: level with ↺ and Finish; a longer caption grows upwards.
            let level = sentenceLabel.centerYAnchor.constraint(equalTo: controls.centerYAnchor)
            level.priority = .defaultHigh
            landscapeConstraints += [
                level,
                sentenceLabel.bottomAnchor.constraint(lessThanOrEqualTo: guide.bottomAnchor, constant: -12),
                sentenceLabel.trailingAnchor.constraint(equalTo: controls.leadingAnchor, constant: -16),
            ]
        }
    }

    /// Layout tests only: LIVE mid-utterance (sign pills, recognized word with candidates, English caption).
    func showSampleUtterance(caption: String = "Hello, how are you doing today? I am happy to meet you") {
        loaded = true; isRunningUI = true
        rail = ["HELLO", "HOW"]; previewText = nil; centerPreview = nil
        committedWord = ("YOU", 0.91, Date())
        top3 = [("YOU", 0.91), ("YOUR", 0.05), ("WE", 0.02)]
        captionText = caption
        finishedSentence = nil; bannerMessage = nil
        render()
    }

    /// Layout tests only: LIVE after Finish (the sentence with Speak again).
    func showSampleFinished(_ sentence: String = "Hello, how are you?") {
        loaded = true; isRunningUI = false
        rail = []; committedWord = nil; centerPreview = nil; top3 = []; captionText = nil
        finishedSentence = sentence
        bannerMessage = ("TRANSLATION", sentence)
        render()
    }

    // MARK: - Recording

    /// Main thread, LIVE only: begin a take (Start), or begin it again (↺), dropping the old one.
    private func startRecording() {
        // Takes are saved only with the Settings switch "Save Live videos" on (default off).
        guard !isPractice, UserDefaults.standard.bool(forKey: LiveReelShared.saveVideosKey) else { return }
        videoTakes += 1
        let name = LiveReelSessions.stamp(openedAt) + String(format: "_%02d", videoTakes)
        let take = LiveVideoRecorder(name: name)
        captureQueue.async {
            self.recorder?.discard()
            self.recorder = take
        }
        queue.async { self.events.append(["event": "video_start", "file": name + ".mp4", "time": Date().timeIntervalSince1970]) }
    }

    /// Main thread: Finish or Back closes the take and saves it to Documents/live_reel_videos.
    private func saveRecording() {
        captureQueue.async {
            guard let take = self.recorder else { return }
            self.recorder = nil
            take.finish()
        }
    }

    // MARK: - Session history

    /// On `queue`: one History row per LIVE visit that recognised or said something.
    private var lastTraceSave = Date.distantPast
    private func saveSession(complete: Bool = false) {
        guard !isPractice, !events.isEmpty else { return }
        let value: [String: Any] = [
            "format": "slt_mobile_live_session_v1", "mode": "live",
            "started_utc": ISO8601DateFormatter().string(from: openedAt),
            "duration_seconds": Date().timeIntervalSince(openedAt),
            "glosses": sessionGlosses, "sentences": sessionSentences, "complete": complete,
            "rendering_mode": "live_captions", "live_captions": LiveReelShared.captions?.name ?? "unloaded",
            "events": events, "timing": LiveReelShared.engine?.timing ?? [:],
            "device": UIDevice.current.model, "system": UIDevice.current.systemVersion,
        ]
        LiveReelSessions.save(value, started: openedAt)
        if complete || Date().timeIntervalSince(lastTraceSave) >= 20, let rows = LiveReelShared.engine?.takeTrace() {
            LiveReelSessions.saveTrace(rows, started: openedAt)
            lastTraceSave = Date()
        }
        lastHistorySave = Date()
    }
}

/// Camera framing hints for LIVE and PRACTICE: light, the signer in view, both shoulders, hands at the
/// frame edge. A hint appears once its condition has held for a second and clears as soon as it is
/// fixed; the first failing check wins. Engine queue only.
struct LiveFraming {
    /// Mean frame luma below this reads as too dark (auto-exposure has run out of range).
    static let darkLuma = 45.0
    private var since: [Int: Double] = [:]
    private var checked = false, personSeen = false, shouldersSeen = false

    mutating func reset() { since = [:]; checked = false; personSeen = false; shouldersSeen = false }

    mutating func update(_ o: LiveObservation, brightness: Double?, seconds: Double) -> String? {
        if o.faceForFeatures {   // body and face are only detected on these frames
            shouldersSeen = o.body.count >= 2 && o.body[0].c > 0 && o.body[1].c > 0
            personSeen = shouldersSeen || o.face.contains { $0.c > 0 }
            checked = true
        }
        let hands = [o.left, o.right].compactMap { $0 }
        let atEdge = hands.contains { hand in
            hand.points.contains { $0.c > 0 && ($0.x < 0.02 || $0.x > 0.98 || $0.y < 0.02 || $0.y > 0.98) }
        }
        let checks: [(String, Bool)] = [
            ("Too dark. Move somewhere brighter or face the light.", (brightness ?? 255) < Self.darkLuma),
            ("Step into view so ATLAS can see you.", checked && !personSeen && hands.isEmpty),
            ("Step back so both shoulders are in view.", checked && personSeen && !shouldersSeen),
            ("Keep your hands inside the frame.", atEdge),
        ]
        var shown: String?
        for (i, (text, failing)) in checks.enumerated() {
            guard failing else { since[i] = nil; continue }
            let start = since[i] ?? seconds
            since[i] = start
            if shown == nil, seconds - start >= 1 { shown = text }
        }
        return shown
    }
}

/// The first-LIVE tips: swipeable cards over the camera with Skip, page dots and Next / Start.
private final class LiveTipsView: UIView, UIScrollViewDelegate {
    var onDone: (() -> Void)?

    static let pages: [(symbol: String, title: String, body: String)] = [
        ("hand.raised.fill", "ATLAS knows 100 signs",
         "It turns the 100 signs in Glosses into English. Other signs are not recognised."),
        ("camera.viewfinder", "Set up the camera",
         "Prop the phone up in landscape at chest height, about an arm's length away. Keep your head, shoulders and hands in view, in good light."),
        ("text.bubble.fill", "Sign, then Finish",
         "Tap Start and sign at a steady pace. The English appears as you sign. Tap Finish and ATLAS reads it aloud. Speak again repeats it."),
        ("hand.raised.fingers.spread.fill", "Finish hands-free",
         "Instead of tapping Finish, raise both hands above your shoulders, right hand open and left hand in a fist, and hold for a second."),
        ("ellipsis.circle.fill", "More options",
         "Tap ••• to hide the sign details and keep only the English, to show the tracking points, or to open the guide."),
    ]

    private let scroll = UIScrollView()
    private let pageControl = UIPageControl()
    private let nextButton = UIButton(type: .system)
    private let skipButton = UIButton(type: .system)
    private var page = 0

    init() {
        super.init(frame: .zero)
        backgroundColor = UIColor.black.withAlphaComponent(0.6)
        accessibilityIdentifier = "liveTips"
        let card = UIView()
        card.backgroundColor = ReelStyle.white
        card.layer.cornerRadius = 28
        card.layer.cornerCurve = .continuous
        scroll.isPagingEnabled = true
        scroll.showsHorizontalScrollIndicator = false
        scroll.delegate = self
        let pages = UIStackView()
        pages.axis = .horizontal
        pages.translatesAutoresizingMaskIntoConstraints = false
        scroll.addSubview(pages)
        for content in Self.pages {
            let pageView = Self.page(content)
            pages.addArrangedSubview(pageView)
            pageView.widthAnchor.constraint(equalTo: scroll.frameLayoutGuide.widthAnchor).isActive = true
        }
        pageControl.numberOfPages = Self.pages.count
        pageControl.currentPageIndicatorTintColor = ReelStyle.blue
        pageControl.pageIndicatorTintColor = ReelStyle.line
        pageControl.addTarget(self, action: #selector(pageTapped), for: .valueChanged)
        for (button, title, color) in [(skipButton, "Skip", ReelStyle.muted), (nextButton, "Next", ReelStyle.blue)] {
            button.setTitle(title, for: .normal)
            button.setTitleColor(color, for: .normal)
            button.titleLabel?.font = ReelStyle.font(17, .semibold)
        }
        skipButton.addTarget(self, action: #selector(finish), for: .touchUpInside)
        nextButton.addTarget(self, action: #selector(nextTapped), for: .touchUpInside)
        nextButton.accessibilityIdentifier = "liveTipsNext"
        let footer = UIStackView(arrangedSubviews: [skipButton, pageControl, nextButton])
        footer.distribution = .equalCentering
        footer.alignment = .center
        for v in [card, scroll, footer] as [UIView] { v.translatesAutoresizingMaskIntoConstraints = false }
        addSubview(card)
        card.addSubview(scroll)
        card.addSubview(footer)
        let width = card.widthAnchor.constraint(equalToConstant: 420)
        width.priority = .defaultHigh
        NSLayoutConstraint.activate([
            card.centerXAnchor.constraint(equalTo: safeAreaLayoutGuide.centerXAnchor),
            card.centerYAnchor.constraint(equalTo: safeAreaLayoutGuide.centerYAnchor),
            width,
            card.widthAnchor.constraint(lessThanOrEqualToConstant: 420),
            card.widthAnchor.constraint(lessThanOrEqualTo: safeAreaLayoutGuide.widthAnchor, constant: -32),
            card.heightAnchor.constraint(lessThanOrEqualTo: safeAreaLayoutGuide.heightAnchor, constant: -24),
            scroll.topAnchor.constraint(equalTo: card.topAnchor, constant: 24),
            scroll.leadingAnchor.constraint(equalTo: card.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: card.trailingAnchor),
            pages.leadingAnchor.constraint(equalTo: scroll.contentLayoutGuide.leadingAnchor),
            pages.trailingAnchor.constraint(equalTo: scroll.contentLayoutGuide.trailingAnchor),
            pages.topAnchor.constraint(equalTo: scroll.contentLayoutGuide.topAnchor),
            pages.bottomAnchor.constraint(equalTo: scroll.contentLayoutGuide.bottomAnchor),
            pages.heightAnchor.constraint(equalTo: scroll.frameLayoutGuide.heightAnchor),
            footer.topAnchor.constraint(equalTo: scroll.bottomAnchor, constant: 8),
            footer.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 24),
            footer.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -24),
            footer.bottomAnchor.constraint(equalTo: card.bottomAnchor, constant: -16),
            footer.heightAnchor.constraint(equalToConstant: 44),
        ])
        let tallest = scroll.heightAnchor.constraint(equalToConstant: 230)
        tallest.priority = .defaultHigh
        tallest.isActive = true
    }

    required init?(coder: NSCoder) { fatalError() }

    private static func page(_ content: (symbol: String, title: String, body: String)) -> UIView {
        let icon = UIImageView(image: UIImage(systemName: content.symbol,
                                              withConfiguration: UIImage.SymbolConfiguration(pointSize: 36, weight: .semibold)))
        icon.tintColor = ReelStyle.blue
        icon.contentMode = .center
        let title = UILabel(), body = UILabel()
        title.text = content.title
        title.font = ReelStyle.font(22, .bold)
        title.textColor = ReelStyle.ink
        title.textAlignment = .center
        title.numberOfLines = 0
        body.text = content.body
        body.font = ReelStyle.font(16, .regular)
        body.textColor = ReelStyle.muted
        body.textAlignment = .center
        body.numberOfLines = 0
        let stack = UIStackView(arrangedSubviews: [icon, title, body])
        stack.axis = .vertical
        stack.spacing = 10
        stack.setCustomSpacing(16, after: icon)
        stack.translatesAutoresizingMaskIntoConstraints = false
        let page = UIView()
        page.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: page.leadingAnchor, constant: 28),
            stack.trailingAnchor.constraint(equalTo: page.trailingAnchor, constant: -28),
            stack.topAnchor.constraint(greaterThanOrEqualTo: page.topAnchor),
            stack.centerYAnchor.constraint(equalTo: page.centerYAnchor),
        ])
        return page
    }

    private func show(page index: Int, animated: Bool) {
        page = max(0, min(Self.pages.count - 1, index))
        pageControl.currentPage = page
        nextButton.setTitle(page == Self.pages.count - 1 ? "Start" : "Next", for: .normal)
        scroll.setContentOffset(CGPoint(x: CGFloat(page) * scroll.bounds.width, y: 0), animated: animated)
    }

    func scrollViewDidEndDecelerating(_ scrollView: UIScrollView) {
        guard scrollView.bounds.width > 0 else { return }
        show(page: Int((scrollView.contentOffset.x / scrollView.bounds.width).rounded()), animated: false)
    }

    @objc private func pageTapped() { show(page: pageControl.currentPage, animated: true) }

    @objc private func nextTapped() {
        if page == Self.pages.count - 1 { finish() } else { show(page: page + 1, animated: true) }
    }

    @objc private func finish() {
        onDone?()
        UIView.animate(withDuration: 0.2, animations: { self.alpha = 0 }) { _ in self.removeFromSuperview() }
    }
}
