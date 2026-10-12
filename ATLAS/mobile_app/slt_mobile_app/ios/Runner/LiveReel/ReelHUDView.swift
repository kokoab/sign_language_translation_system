// The desktop Reel HUD (scripts/reel_hud_v17.py) over the camera preview:
//   top-left     state chip (READY / LISTENING / SIGNING / PREVIEW / FINISHING) + FPS
//   top-right    committed-word rail (newest green) with the amber "WORD?" preview at its end
//   below state  finished sentence (dot turns green while spoken)
//   centre       big current word (amber preview / white commit, fading) + confidence bar
//   bottom-left  stats; bottom-right top-3 candidates; bottom RESET · START/STOP · FINISH
// `minimal` (a practice round) keeps the centre word, stats and candidates only.

import UIKit

final class ReelHUDView: UIView {
    var onStart: (() -> Void)?
    var onFinish: (() -> Void)?
    var onReset: (() -> Void)?

    var minimal = false { didSet { layoutMode(); render() } }
    var loaded = false { didSet { render() } }
    var running = false { didSet { render() } }
    var finishing = false { didSet { render() } }
    var speaking = false { didSet { sentenceDot.backgroundColor = speaking ? HUD.accent : HUD.white.withAlphaComponent(0.35) } }

    private(set) var rail: [String] = []
    private var previewText: String?
    private var committedWord: (text: String, score: Double, at: Date)?
    private var centerPreview: (text: String, score: Double)?
    private var top3: [(gloss: String, score: Double)] = []
    private var fps = 0.0
    private var hasHands = false

    private let bones = CAShapeLayer()
    private let joints = CAShapeLayer()
    private let stateHost = UIView()
    private let railStack = UIStackView()
    private let sentenceDot = UIView()
    private let sentenceLabel = ShadowLabel(size: 24, weight: .semibold, shadow: 3)
    private let bigWord = ShadowLabel(size: 46, weight: .heavy, shadow: 4)
    private let barTrack = UIView()
    private let barFill = UIView()
    private var barFillWidth: NSLayoutConstraint!
    private let scoreLabel = UILabel()
    private let statsLabel = UILabel()
    private let statsPanel = UIView()
    private let candidatesStack = UIStackView()
    private let startButton = PillButton("START", kind: .primary, size: 16)
    private let finishButton = PillButton("FINISH", kind: .primary, size: 16)
    private let resetButton = PillButton("RESET", kind: .secondary, size: 16)
    private let controls = UIStackView()
    private var bigWordBottom: NSLayoutConstraint!
    private var bigWordAboveStats: NSLayoutConstraint!

    init(topInset: CGFloat) {
        super.init(frame: .zero)
        build(topInset: topInset)
        render()
    }

    required init?(coder: NSCoder) { fatalError() }

    /// Touches pass through to the page below except on the HUD's own controls.
    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        let view = super.hitTest(point, with: event)
        return view === self ? nil : view
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        bones.frame = bounds
        joints.frame = bounds
    }

    // MARK: state

    private static func display(_ gloss: String) -> String {
        if gloss.hasPrefix("fs-") { return LiveStage3.spelledText(gloss).uppercased() }
        if gloss.hasPrefix("FS_") { return String(gloss.dropFirst(3)) }
        return gloss
    }

    /// One processed frame (show_preview / emit of the desktop runner).
    func apply(_ u: ReelCamera.Update) {
        fps = u.fps
        commit(u.committed)
        if let pending = u.pending {
            previewText = pending + "…"
            centerPreview = (pending + "…", 1)
        } else if !u.committed.isEmpty {
            previewText = nil
            centerPreview = nil
        } else if let p = u.preview, p.emit {
            previewText = Self.display(p.gloss) + "?"
            centerPreview = (Self.display(p.gloss) + "?", p.score)
            top3 = p.top3.map { (Self.display($0.gloss), $0.score) }
        } else {
            previewText = nil
            centerPreview = nil
        }
        statsLabel.attributedText = Self.stats([
            ("FPS", String(format: "%.1f", u.fps)), ("FRAME", String(format: "%.0f ms", u.frameMilliseconds)),
            ("HANDS", "\([u.observation.left, u.observation.right].compactMap { $0 }.count)"), ("FRAMES", "\(u.frames)"),
        ])
        drawSkeleton(u.observation, body: u.body, face: u.face)
        render()
    }

    func commit(_ words: [LiveWord]) {
        if !words.isEmpty, !finishing, sentenceLabel.text != nil { sentenceLabel.text = nil }   // new utterance
        for w in words {
            rail.append(Self.display(w.gloss))
            committedWord = (Self.display(w.gloss), w.score, Date())
        }
        if rail.count > 24 { rail.removeFirst(rail.count - 24) }
        render()
    }

    func endOfStream() {
        previewText = nil
        centerPreview = nil
        render()
    }

    func setSentence(_ text: String?) {
        sentenceLabel.text = text
        if text != nil && !finishing { rail = [] }
        render()
    }

    func clear() {
        rail = []; previewText = nil; committedWord = nil; centerPreview = nil; top3 = []
        sentenceLabel.text = nil
        render()
    }

    func clearSkeleton() {
        bones.path = nil; joints.path = nil
        hasHands = false
        fps = 0
        render()
    }

    private static func stats(_ rows: [(String, String)]) -> NSAttributedString {
        let text = NSMutableAttributedString()
        for (i, (label, value)) in rows.enumerated() {
            text.append(NSAttributedString(string: label.padding(toLength: 8, withPad: " ", startingAt: 0), attributes: [
                .font: HUD.mono(10, .semibold), .foregroundColor: HUD.muted]))
            text.append(NSAttributedString(string: value + (i < rows.count - 1 ? "\n" : ""), attributes: [
                .font: HUD.mono(12, .medium), .foregroundColor: HUD.white]))
        }
        return text
    }

    // MARK: drawing

    private func render() {
        let (state, color): (String, UIColor) =
            !loaded ? ("WARMING UP", HUD.muted) :
            finishing ? ("FINISHING", HUD.amber) :
            !running ? ("READY", HUD.muted) :
            centerPreview != nil ? ("PREVIEW", HUD.amber) :
            hasHands ? ("SIGNING", HUD.accent) : ("LISTENING", HUD.muted)
        stateHost.subviews.forEach { $0.removeFromSuperview() }
        stateHost.pin(ChipView(text: String(format: "%@   %.1f FPS", state, fps), dotColor: color, textColor: HUD.white, size: 13))

        railStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
        if !minimal {
            let recent = Array(rail.suffix(7))
            for (index, word) in recent.enumerated() {
                let newest = index == recent.count - 1 && previewText == nil
                railStack.addArrangedSubview(ChipView(
                    text: word, dotColor: newest ? HUD.accent : HUD.muted, textColor: newest ? HUD.white : HUD.muted,
                    alpha: newest ? HUD.panelAlpha : HUD.chipAlpha, size: 13))
            }
            if let previewText {
                railStack.addArrangedSubview(ChipView(text: previewText, dotColor: HUD.amber, textColor: HUD.amber, size: 13))
            }
        }

        var centerAlpha: CGFloat = 0
        if let p = centerPreview {
            bigWord.text = p.text; bigWord.textColor = HUD.amber
            barFill.backgroundColor = HUD.amber
            scoreLabel.text = String(format: "%.2f", p.score)
            barFillWidth.constant = 200 * CGFloat(max(0, min(1, p.score)))
            centerAlpha = 1
        } else if let c = committedWord {
            let age = Date().timeIntervalSince(c.at)
            bigWord.text = c.text; bigWord.textColor = HUD.white
            barFill.backgroundColor = HUD.accent
            scoreLabel.text = String(format: "%.2f", c.score)
            barFillWidth.constant = 200 * CGFloat(max(0, min(1, c.score)))
            centerAlpha = age < 1.6 ? 1 : max(0.38, 1 - CGFloat(age - 1.6) * 0.43)
        }
        for v in [bigWord, barTrack, scoreLabel] as [UIView] { v.alpha = centerAlpha }

        candidatesStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
        for (i, c) in top3.enumerated() {
            candidatesStack.addArrangedSubview(ChipView(
                text: String(format: "%@  %.2f", c.gloss, c.score),
                dotColor: HUD.accent.withAlphaComponent(0.45 + 0.55 * CGFloat(min(1, max(0, c.score)))),
                textColor: i == 0 ? HUD.white : HUD.muted, alpha: i == 0 ? HUD.panelAlpha : HUD.chipAlpha, size: 12))
        }
        candidatesStack.isHidden = !running || top3.isEmpty

        startButton.setTitle(running ? "STOP" : "START", for: .normal)
        startButton.kind = !loaded ? .disabled : running ? .danger : .primary
        finishButton.setTitle(finishing ? "WORKING" : "FINISH", for: .normal)
        finishButton.kind = !loaded ? .disabled : finishing ? .working : .primary
        sentenceDot.isHidden = minimal || sentenceLabel.text == nil
        sentenceLabel.textColor = finishing ? HUD.muted : HUD.white
    }

    private func layoutMode() {
        controls.isHidden = minimal
        sentenceLabel.isHidden = minimal
        railStack.isHidden = minimal
        bigWordAboveStats.isActive = minimal     // no controls in a round: the word sits lower
    }

    /// Hands, body and face on the mirrored, aspect-fit preview; body/face persist until re-detected.
    private func drawSkeleton(_ o: LiveObservation, body: [LivePoint]?, face: [LivePoint]?) {
        let w = CGFloat(o.width), h = CGFloat(o.height)
        let scale = min(bounds.width / w, bounds.height / h)
        let ox = (bounds.width - w * scale) / 2, oy = (bounds.height - h * scale) / 2
        func point(_ p: LivePoint) -> CGPoint {
            CGPoint(x: ox + (1 - CGFloat(p.x)) * w * scale, y: oy + CGFloat(p.y) * h * scale)
        }
        let lines = UIBezierPath(), dots = UIBezierPath()
        let links = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                     (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
        let hands = [o.left, o.right].compactMap { $0 }
        hasHands = !hands.isEmpty
        for hand in hands {
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
        bones.path = lines.cgPath
        joints.path = dots.cgPath
    }

    // MARK: layout

    @objc private func startTapped() { onStart?() }
    @objc private func finishTapped() { onFinish?() }
    @objc private func resetTapped() { onReset?() }

    private func build(topInset: CGFloat) {
        bones.strokeColor = HUD.accent.withAlphaComponent(0.9).cgColor
        bones.fillColor = UIColor.clear.cgColor
        bones.lineWidth = 2
        bones.lineCap = .round
        joints.fillColor = HUD.white.cgColor
        layer.addSublayer(bones)
        layer.addSublayer(joints)

        railStack.axis = .horizontal
        railStack.spacing = 6
        railStack.alignment = .center
        sentenceDot.backgroundColor = HUD.white.withAlphaComponent(0.35)
        sentenceDot.layer.cornerRadius = 8
        sentenceLabel.numberOfLines = 3
        bigWord.textAlignment = .center
        bigWord.adjustsFontSizeToFitWidth = true
        bigWord.minimumScaleFactor = 0.4
        barTrack.backgroundColor = HUD.ink.withAlphaComponent(HUD.panelAlpha)
        barTrack.layer.cornerRadius = 3
        barFill.layer.cornerRadius = 3
        scoreLabel.font = HUD.mono(12, .medium)
        scoreLabel.textColor = HUD.muted
        scoreLabel.textAlignment = .center
        statsPanel.backgroundColor = HUD.ink.withAlphaComponent(HUD.panelAlpha)
        statsPanel.layer.cornerRadius = 12
        statsPanel.layer.borderColor = HUD.white.withAlphaComponent(90 / 255).cgColor
        statsPanel.layer.borderWidth = 1
        statsLabel.numberOfLines = 0
        statsLabel.attributedText = Self.stats([("FPS", "--"), ("FRAME", "--"), ("HANDS", "--"), ("FRAMES", "0")])
        candidatesStack.axis = .vertical
        candidatesStack.spacing = 5
        candidatesStack.alignment = .trailing

        startButton.addTarget(self, action: #selector(startTapped), for: .touchUpInside)
        finishButton.addTarget(self, action: #selector(finishTapped), for: .touchUpInside)
        resetButton.addTarget(self, action: #selector(resetTapped), for: .touchUpInside)
        for (button, width) in [(resetButton, 120.0), (startButton, 170.0), (finishButton, 130.0)] {
            button.widthAnchor.constraint(equalToConstant: width).isActive = true
            button.heightAnchor.constraint(equalToConstant: 44).isActive = true
            controls.addArrangedSubview(button)
        }
        controls.axis = .horizontal
        controls.spacing = 12

        for v in [stateHost, railStack, sentenceDot, sentenceLabel, bigWord, barTrack, scoreLabel,
                  statsPanel, statsLabel, candidatesStack, controls] as [UIView] {
            v.translatesAutoresizingMaskIntoConstraints = false
            addSubview(v)
        }
        barFill.translatesAutoresizingMaskIntoConstraints = false
        barTrack.addSubview(barFill)
        barFillWidth = barFill.widthAnchor.constraint(equalToConstant: 0)
        let guide = safeAreaLayoutGuide
        bigWordBottom = bigWord.bottomAnchor.constraint(equalTo: barTrack.topAnchor, constant: -6)
        bigWordAboveStats = scoreLabel.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -64)
        let scoreAboveControls = scoreLabel.bottomAnchor.constraint(equalTo: controls.topAnchor, constant: -10)
        scoreAboveControls.priority = .defaultHigh
        NSLayoutConstraint.activate([
            stateHost.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 14),
            stateHost.topAnchor.constraint(equalTo: guide.topAnchor, constant: topInset),

            railStack.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -14),
            railStack.centerYAnchor.constraint(equalTo: stateHost.centerYAnchor),
            railStack.leadingAnchor.constraint(greaterThanOrEqualTo: stateHost.trailingAnchor, constant: 14),

            sentenceDot.leadingAnchor.constraint(equalTo: stateHost.leadingAnchor),
            sentenceDot.topAnchor.constraint(equalTo: sentenceLabel.topAnchor, constant: 6),
            sentenceDot.widthAnchor.constraint(equalToConstant: 16),
            sentenceDot.heightAnchor.constraint(equalToConstant: 16),
            sentenceLabel.leadingAnchor.constraint(equalTo: sentenceDot.trailingAnchor, constant: 10),
            sentenceLabel.topAnchor.constraint(equalTo: stateHost.bottomAnchor, constant: 12),
            sentenceLabel.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.78),

            controls.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            controls.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -8),

            scoreAboveControls,
            scoreLabel.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            barTrack.bottomAnchor.constraint(equalTo: scoreLabel.topAnchor, constant: -4),
            barTrack.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            barTrack.widthAnchor.constraint(equalToConstant: 200),
            barTrack.heightAnchor.constraint(equalToConstant: 5),
            barFill.leadingAnchor.constraint(equalTo: barTrack.leadingAnchor),
            barFill.topAnchor.constraint(equalTo: barTrack.topAnchor),
            barFill.bottomAnchor.constraint(equalTo: barTrack.bottomAnchor),
            barFillWidth,
            bigWordBottom,
            bigWord.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            bigWord.widthAnchor.constraint(lessThanOrEqualTo: guide.widthAnchor, multiplier: 0.5),

            statsPanel.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 14),
            statsPanel.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -8),
            statsLabel.leadingAnchor.constraint(equalTo: statsPanel.leadingAnchor, constant: 10),
            statsLabel.trailingAnchor.constraint(equalTo: statsPanel.trailingAnchor, constant: -10),
            statsLabel.topAnchor.constraint(equalTo: statsPanel.topAnchor, constant: 8),
            statsLabel.bottomAnchor.constraint(equalTo: statsPanel.bottomAnchor, constant: -8),

            candidatesStack.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -14),
            candidatesStack.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -8),
        ])
        bigWordBottom.isActive = true
        layoutMode()
    }
}
