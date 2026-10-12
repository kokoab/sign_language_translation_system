// The app shell's pages, drawn like the desktop shell (scripts/app_pages_v17.py):
// nav bar, animated home, the 100-sign gallery and detail, practice setup / round / result,
// and session history.

import AVFoundation
import UIKit

// MARK: - Catalog

struct GlossExample {
    let label: String
    let category: String
    let seconds: Double
    let notes: [String]
    let poster: URL?
    let loop: URL?
}

enum GlossCatalog {
    static let examples: [GlossExample] = {
        guard let root = Bundle.main.url(forResource: "gloss_examples", withExtension: nil),
              let data = try? Data(contentsOf: root.appendingPathComponent("catalog.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let examples = json["examples"] as? [String: [String: Any]] else { return [] }
        return examples.keys.sorted().map { label in
            let row = examples[label]!
            let assets = row["assets"] as? [String: Any] ?? [:]
            return GlossExample(
                label: label, category: (row["category"] as? String ?? "").replacingOccurrences(of: "_", with: " "),
                seconds: row["seconds"] as? Double ?? 0, notes: row["notes"] as? [String] ?? [],
                poster: (assets["poster"] as? String).map { root.appendingPathComponent($0) },
                loop: (assets["loop"] as? String).map { root.appendingPathComponent($0) })
        }
    }()

    static func example(_ label: String) -> GlossExample? { examples.first { $0.label == label } }
}

// MARK: - Nav bar

final class ShellNavBar: UIView {
    static let pages = ["HOME", "LIVE", "GLOSSES", "PRACTICE", "HISTORY"]
    var onSelect: ((String) -> Void)?
    private var buttons: [String: UIButton] = [:]
    private let note = ShadowLabel(size: 12, weight: .medium, color: HUD.white.withAlphaComponent(0.67))

    init() {
        super.init(frame: .zero)
        let stack = UIStackView()
        stack.axis = .horizontal
        stack.spacing = 4
        for page in Self.pages {
            let button = UIButton(type: .custom)
            button.setTitle(page, for: .normal)
            button.titleLabel?.font = HUD.font(14, .semibold)
            button.contentEdgeInsets = UIEdgeInsets(top: 0, left: 13, bottom: 0, right: 13)
            button.layer.cornerRadius = 16
            button.titleLabel?.layer.shadowColor = UIColor.black.cgColor
            button.titleLabel?.layer.shadowRadius = 2
            button.titleLabel?.layer.shadowOffset = CGSize(width: 0, height: 1)
            button.heightAnchor.constraint(equalToConstant: 32).isActive = true
            button.addAction(UIAction { [weak self] _ in self?.onSelect?(page) }, for: .touchUpInside)
            buttons[page] = button
            stack.addArrangedSubview(button)
        }
        stack.translatesAutoresizingMaskIntoConstraints = false
        note.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        addSubview(note)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: centerYAnchor),
            note.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            note.centerYAnchor.constraint(equalTo: centerYAnchor),
            heightAnchor.constraint(equalToConstant: 46),
        ])
        setActive("HOME")
    }

    required init?(coder: NSCoder) { fatalError() }

    func setActive(_ page: String) {
        for (name, button) in buttons {
            let active = name == page
            button.backgroundColor = active ? HUD.accent.withAlphaComponent(235 / 255) : .clear
            button.setTitleColor(active ? HUD.ink : HUD.white.withAlphaComponent(205 / 255), for: .normal)
            button.titleLabel?.layer.shadowOpacity = active ? 0 : 0.85
        }
    }

    func setNote(_ text: String?) { note.text = text }
}

// MARK: - Home

/// A slow aurora: three soft colour fields drifting over a dark gradient, with a vignette.
final class AuroraView: UIView {
    private let base = CAGradientLayer()
    private var glows: [CAGradientLayer] = []
    private let vignette = CAGradientLayer()
    private let veil = CALayer()
    private var link: CADisplayLink?
    private let started = CACurrentMediaTime()
    // (rgb, rate x, rate y, phase, radius, strength) — the desktop Backdrop palette.
    private let palette: [(UIColor, Double, Double, Double, CGFloat, CGFloat)] = [
        (UIColor(red: 70 / 255, green: 210 / 255, blue: 120 / 255, alpha: 1), 0.055, 0.041, 0.0, 0.42, 0.55),
        (UIColor(red: 60 / 255, green: 120 / 255, blue: 190 / 255, alpha: 1), 0.037, 0.062, 2.1, 0.38, 0.40),
        (UIColor(red: 180 / 255, green: 90 / 255, blue: 90 / 255, alpha: 1), 0.047, 0.033, 4.2, 0.34, 0.30),
    ]

    override init(frame: CGRect) {
        super.init(frame: frame)
        base.colors = [HUD.ink.cgColor, UIColor(red: 24 / 255, green: 27 / 255, blue: 30 / 255, alpha: 1).cgColor]
        layer.addSublayer(base)
        for (color, _, _, _, _, strength) in palette {
            let glow = CAGradientLayer()
            glow.type = .radial
            glow.colors = [color.withAlphaComponent(strength).cgColor, color.withAlphaComponent(strength * 0.45).cgColor,
                           color.withAlphaComponent(0).cgColor]
            glow.locations = [0, 0.45, 1]
            glow.startPoint = CGPoint(x: 0.5, y: 0.5)
            glow.endPoint = CGPoint(x: 1, y: 1)
            glows.append(glow)
            layer.addSublayer(glow)
        }
        vignette.type = .radial
        vignette.colors = [UIColor.black.withAlphaComponent(0).cgColor, UIColor.black.withAlphaComponent(0.55).cgColor]
        vignette.locations = [0.35, 1]
        vignette.startPoint = CGPoint(x: 0.5, y: 0.5)
        vignette.endPoint = CGPoint(x: 1, y: 1)
        layer.addSublayer(vignette)
        veil.backgroundColor = HUD.ink.withAlphaComponent(58 / 255).cgColor
        layer.addSublayer(veil)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        link?.invalidate()
        link = nil
        if window != nil {
            let link = CADisplayLink(target: self, selector: #selector(step))
            link.preferredFrameRateRange = CAFrameRateRange(minimum: 15, maximum: 30, preferred: 30)
            link.add(to: .main, forMode: .common)
            self.link = link
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        base.frame = bounds; vignette.frame = bounds; veil.frame = bounds
        step()
    }

    @objc private func step() {
        let now = CACurrentMediaTime() - started + 20
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        for (glow, (_, rx, ry, phase, radius, _)) in zip(glows, palette) {
            let cx = 0.5 + 0.40 * sin(now * rx * 2 * .pi + phase)
            let cy = 0.5 + 0.32 * sin(now * ry * 2 * .pi + phase * 1.7)
            let size = CGSize(width: bounds.width * radius * 2.6, height: bounds.width * radius * 2.6)
            glow.frame = CGRect(x: bounds.width * cx - size.width / 2, y: bounds.height * cy - size.height / 2,
                                width: size.width, height: size.height)
        }
        CATransaction.commit()
    }
}

final class HomePage: UIView {
    var onAction: ((String) -> Void)?
    private let title = ShadowLabel(size: 34, weight: .heavy, shadow: 3)
    private let subtitle = ShadowLabel(size: 13, weight: .medium, color: HUD.white.withAlphaComponent(0.72))
    private var buttons: [(String, PillButton)] = []
    private var content: [UIView] = []

    init() {
        super.init(frame: .zero)
        pin(AuroraView())
        title.text = "Sign Language Translation"
        subtitle.text = "\(GlossCatalog.examples.count == 0 ? 100 : GlossCatalog.examples.count) signs  ·  Apple Vision  ·  entirely on this device"
        let stack = UIStackView()
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 4
        stack.addArrangedSubview(title)
        stack.addArrangedSubview(subtitle)
        stack.setCustomSpacing(22, after: subtitle)
        for (action, label) in [("LIVE", "START CAMERA"), ("PRACTICE", "PRACTICE"),
                                ("GLOSSES", "THE \(max(GlossCatalog.examples.count, 100)) SIGNS"), ("HISTORY", "SESSION HISTORY")] {
            let button = PillButton(label, kind: action == "LIVE" ? .primary : .secondary, size: 15)
            button.widthAnchor.constraint(equalToConstant: 340).isActive = true
            button.heightAnchor.constraint(equalToConstant: 44).isActive = true
            button.addAction(UIAction { [weak self] _ in self?.onAction?(action) }, for: .touchUpInside)
            stack.addArrangedSubview(button)
            stack.setCustomSpacing(10, after: button)
            buttons.append((action, button))
        }
        content = [title, subtitle] + buttons.map(\.1)
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: centerXAnchor),
            stack.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 58),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func update(ready: Bool, status: String, sessions: Int) {
        for (action, button) in buttons {
            switch action {
            case "LIVE":
                button.setTitle(ready ? "START CAMERA" : status.uppercased(), for: .normal)
                button.kind = ready ? .primary : .disabled
            case "PRACTICE":
                button.kind = ready ? .secondary : .disabled
            case "HISTORY":
                button.setTitle(sessions > 0 ? "SESSION HISTORY  ·  \(sessions)" : "SESSION HISTORY", for: .normal)
            default: break
            }
        }
    }

    /// Everything eases in together, each button a beat after the one above it.
    func enter() {
        for (index, view) in content.enumerated() {
            view.alpha = 0
            view.transform = CGAffineTransform(translationX: 0, y: 18)
            let delay = index < 2 ? 0 : 0.08 * Double(index - 2)
            UIView.animate(withDuration: 0.65, delay: delay, options: [.curveEaseOut]) {
                view.alpha = 1
                view.transform = .identity
            }
        }
    }
}

// MARK: - Glosses

private final class GlossCell: UICollectionViewCell {
    let image = UIImageView()
    let caption = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        image.contentMode = .scaleAspectFill
        image.clipsToBounds = true
        image.backgroundColor = HUD.ink
        caption.font = HUD.font(12, .semibold)
        caption.textColor = HUD.white
        caption.textAlignment = .center
        caption.backgroundColor = HUD.ink.withAlphaComponent(225 / 255)
        for v in [image, caption] { v.translatesAutoresizingMaskIntoConstraints = false; contentView.addSubview(v) }
        NSLayoutConstraint.activate([
            image.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            image.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            image.topAnchor.constraint(equalTo: contentView.topAnchor),
            caption.topAnchor.constraint(equalTo: image.bottomAnchor),
            caption.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            caption.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            caption.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            caption.heightAnchor.constraint(equalToConstant: 22),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }
}

final class GlossesPage: UIView, UICollectionViewDataSource, UICollectionViewDelegateFlowLayout {
    var onSelect: ((String) -> Void)?
    private let grid: UICollectionView
    private let cache = NSCache<NSString, UIImage>()

    init() {
        let layout = UICollectionViewFlowLayout()
        layout.minimumInteritemSpacing = 12
        layout.minimumLineSpacing = 14
        grid = UICollectionView(frame: .zero, collectionViewLayout: layout)
        super.init(frame: .zero)
        backgroundColor = HUD.ink
        let heading = UILabel()
        heading.text = "\(GlossCatalog.examples.count) signs the model can recognise"
        heading.font = HUD.font(14, .semibold)
        heading.textColor = HUD.white
        grid.backgroundColor = .clear
        grid.dataSource = self
        grid.delegate = self
        grid.register(GlossCell.self, forCellWithReuseIdentifier: "cell")
        grid.indicatorStyle = .white
        let footer = UILabel()
        footer.text = "Tap a sign to see it signed · scroll for more"
        footer.font = HUD.font(11, .medium)
        footer.textColor = HUD.muted
        let guide = safeAreaLayoutGuide
        for v in [heading, grid, footer] { v.translatesAutoresizingMaskIntoConstraints = false; addSubview(v) }
        NSLayoutConstraint.activate([
            heading.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 18),
            heading.topAnchor.constraint(equalTo: guide.topAnchor, constant: 52),
            grid.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 18),
            grid.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -18),
            grid.topAnchor.constraint(equalTo: heading.bottomAnchor, constant: 10),
            grid.bottomAnchor.constraint(equalTo: footer.topAnchor, constant: -6),
            footer.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 18),
            footer.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        GlossCatalog.examples.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "cell", for: indexPath) as! GlossCell
        let example = GlossCatalog.examples[indexPath.item]
        cell.caption.text = example.label
        let key = example.label as NSString
        if let image = cache.object(forKey: key) {
            cell.image.image = image
        } else if let url = example.poster, let image = UIImage(contentsOfFile: url.path) {
            cache.setObject(image, forKey: key)
            cell.image.image = image
        } else {
            cell.image.image = nil
        }
        return cell
    }

    func collectionView(_ collectionView: UICollectionView, layout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        let columns: CGFloat = 5
        let width = floor((collectionView.bounds.width - 12 * (columns - 1)) / columns)
        return CGSize(width: width, height: width * 0.75 + 22)
    }

    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        onSelect?(GlossCatalog.examples[indexPath.item].label)
    }
}

/// A muted, endlessly looping example clip.
final class LoopingClipView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    private var player: AVQueuePlayer?
    private var looper: AVPlayerLooper?

    func play(_ url: URL?) {
        stop()
        guard let url else { return }
        let player = AVQueuePlayer()
        player.isMuted = true
        looper = AVPlayerLooper(player: player, templateItem: AVPlayerItem(url: url))
        (layer as! AVPlayerLayer).player = player
        (layer as! AVPlayerLayer).videoGravity = .resizeAspect
        player.play()
        self.player = player
    }

    func stop() {
        player?.pause()
        looper = nil
        player = nil
        (layer as! AVPlayerLayer).player = nil
    }
}

final class GlossDetailPage: UIView {
    var onBack: (() -> Void)?
    var onPractice: ((String) -> Void)?
    private let clip = LoopingClipView()
    private let name = ShadowLabel(size: 40, weight: .heavy, shadow: 3)
    private let category = ChipView(text: "", dotColor: nil, textColor: HUD.muted, size: 12)
    private let length = UILabel()
    private let notes = UILabel()
    private var label = ""

    init() {
        super.init(frame: .zero)
        backgroundColor = HUD.ink
        clip.backgroundColor = .black
        length.font = HUD.font(13, .medium)
        length.textColor = HUD.muted
        notes.font = HUD.font(12, .medium)
        notes.textColor = HUD.amber
        notes.numberOfLines = 0
        let back = PillButton("BACK", kind: .secondary)
        back.addAction(UIAction { [weak self] _ in self?.onBack?() }, for: .touchUpInside)
        let practice = PillButton("SIGN THIS NOW", kind: .primary)
        practice.addAction(UIAction { [weak self] _ in
            guard let self else { return }
            self.onPractice?(self.label)
        }, for: .touchUpInside)
        let text = UIStackView(arrangedSubviews: [name, category, length, notes])
        text.axis = .vertical
        text.alignment = .leading
        text.spacing = 10
        let guide = safeAreaLayoutGuide
        for v in [clip, text, back, practice] as [UIView] { v.translatesAutoresizingMaskIntoConstraints = false; addSubview(v) }
        NSLayoutConstraint.activate([
            clip.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 30),
            clip.topAnchor.constraint(equalTo: guide.topAnchor, constant: 56),
            clip.bottomAnchor.constraint(equalTo: back.topAnchor, constant: -16),
            clip.widthAnchor.constraint(equalTo: clip.heightAnchor, multiplier: 4 / 3),
            text.leadingAnchor.constraint(equalTo: clip.trailingAnchor, constant: 34),
            text.topAnchor.constraint(equalTo: clip.topAnchor),
            text.trailingAnchor.constraint(lessThanOrEqualTo: guide.trailingAnchor, constant: -20),
            back.leadingAnchor.constraint(equalTo: clip.leadingAnchor),
            back.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -10),
            back.widthAnchor.constraint(equalToConstant: 130),
            back.heightAnchor.constraint(equalToConstant: 44),
            practice.leadingAnchor.constraint(equalTo: text.leadingAnchor),
            practice.bottomAnchor.constraint(equalTo: back.bottomAnchor),
            practice.widthAnchor.constraint(equalToConstant: 210),
            practice.heightAnchor.constraint(equalToConstant: 44),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func show(_ label: String) {
        self.label = label
        let example = GlossCatalog.example(label)
        name.text = label
        category.label.text = (example?.category ?? "").uppercased()
        category.isHidden = (example?.category ?? "").isEmpty
        length.text = String(format: "Clip  %.1fs", example?.seconds ?? 0)
        notes.text = (example?.notes ?? []).map { "· " + $0 }.joined(separator: "\n")
        clip.play(example?.loop)
    }

    func stop() { clip.stop() }
}

// MARK: - Practice

final class PracticeSetupPage: UIView {
    var onStart: ((Int) -> Void)?

    init() {
        super.init(frame: .zero)
        backgroundColor = HUD.ink
        let title = ShadowLabel(size: 34, weight: .heavy)
        title.text = "Practice"
        let subtitle = UILabel()
        subtitle.text = "The app picks signs at random.  Sign each one; the reference stays on screen."
        subtitle.font = HUD.font(14, .medium)
        subtitle.textColor = HUD.muted
        let row = UIStackView()
        row.axis = .horizontal
        row.spacing = 12
        for (index, size) in [5, 10, 20].enumerated() {
            let button = PillButton("\(size) SIGNS", kind: index == 0 ? .primary : .secondary)
            button.widthAnchor.constraint(equalToConstant: 118).isActive = true
            button.heightAnchor.constraint(equalToConstant: 48).isActive = true
            button.addAction(UIAction { [weak self] _ in self?.onStart?(size) }, for: .touchUpInside)
            row.addArrangedSubview(button)
        }
        let stack = UIStackView(arrangedSubviews: [title, subtitle, row])
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 6
        stack.setCustomSpacing(40, after: subtitle)
        let footer = UILabel()
        footer.text = "\(GlossCatalog.examples.count) signs available"
        footer.font = HUD.font(11, .medium)
        footer.textColor = HUD.muted
        for v in [stack, footer] as [UIView] { v.translatesAutoresizingMaskIntoConstraints = false; addSubview(v) }
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: centerXAnchor),
            stack.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 80),
            footer.leadingAnchor.constraint(equalTo: safeAreaLayoutGuide.leadingAnchor, constant: 18),
            footer.bottomAnchor.constraint(equalTo: safeAreaLayoutGuide.bottomAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }
}

final class PracticeResultPage: UIView {
    var onAgain: (() -> Void)?
    var onDone: (() -> Void)?
    private let score = ShadowLabel(size: 56, weight: .heavy)
    private let remark = UILabel()
    private let missed = UILabel()

    init() {
        super.init(frame: .zero)
        backgroundColor = HUD.ink
        remark.font = HUD.font(17, .medium)
        remark.textColor = HUD.muted
        missed.font = HUD.font(13, .medium)
        missed.textColor = HUD.amber
        missed.numberOfLines = 2
        missed.textAlignment = .center
        let again = PillButton("PRACTICE AGAIN", kind: .primary)
        again.addAction(UIAction { [weak self] _ in self?.onAgain?() }, for: .touchUpInside)
        let done = PillButton("DONE", kind: .secondary)
        done.addAction(UIAction { [weak self] _ in self?.onDone?() }, for: .touchUpInside)
        for (button, width) in [(again, 190.0), (done, 110.0)] {
            button.widthAnchor.constraint(equalToConstant: width).isActive = true
            button.heightAnchor.constraint(equalToConstant: 46).isActive = true
        }
        let row = UIStackView(arrangedSubviews: [again, done])
        row.spacing = 12
        let stack = UIStackView(arrangedSubviews: [score, remark, missed, row])
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 8
        stack.setCustomSpacing(26, after: missed)
        let footer = UILabel()
        footer.text = "A miss may be the model, not you — scores are uncalibrated"
        footer.font = HUD.font(11, .medium)
        footer.textColor = HUD.muted
        for v in [stack, footer] as [UIView] { v.translatesAutoresizingMaskIntoConstraints = false; addSubview(v) }
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: centerXAnchor),
            stack.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 70),
            stack.widthAnchor.constraint(lessThanOrEqualTo: widthAnchor, multiplier: 0.8),
            footer.leadingAnchor.constraint(equalTo: safeAreaLayoutGuide.leadingAnchor, constant: 18),
            footer.bottomAnchor.constraint(equalTo: safeAreaLayoutGuide.bottomAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func show(score value: Int, total: Int, missed list: [String]) {
        score.text = "\(value) of \(total)"
        score.textColor = value == total ? HUD.accent : HUD.white
        remark.text = total > 0 && value == total ? "Perfect round." : total > 0 && Double(value) >= Double(total) * 0.6 ? "Nice work." : "Keep going."
        missed.text = list.isEmpty ? nil : "Not recognised: " + list.prefix(8).joined(separator: ", ")
    }
}

/// A round's furniture over the camera: target banner, round/score chip, reference clip, skip,
/// and the full-frame "YEHEY!" flash on a match.
final class PracticeOverlay: UIView {
    var onSkip: (() -> Void)?
    private let banner = UIView()
    private let bannerLabel = UILabel()
    private let roundHost = UIView()
    private let clip = LoopingClipView()
    private let clipCaption = UILabel()
    private let skip = PillButton("SKIP THIS SIGN", kind: .secondary)
    private let flash = UIView()
    private let yehey = ShadowLabel(size: 64, weight: .heavy, shadow: 4)
    private let matched = UILabel()
    private var clipLabel: String?

    init(topInset: CGFloat) {
        super.init(frame: .zero)
        banner.backgroundColor = HUD.ink.withAlphaComponent(225 / 255)
        banner.layer.cornerRadius = 22
        banner.layer.borderWidth = 1
        bannerLabel.font = HUD.font(16, .semibold)
        bannerLabel.textAlignment = .center
        clip.backgroundColor = .black
        clip.layer.borderColor = HUD.white.withAlphaComponent(90 / 255).cgColor
        clip.layer.borderWidth = 1
        clipCaption.text = "HOW IT LOOKS"
        clipCaption.font = HUD.font(10, .semibold)
        clipCaption.textColor = HUD.muted
        clipCaption.textAlignment = .center
        clipCaption.backgroundColor = HUD.ink.withAlphaComponent(225 / 255)
        skip.addAction(UIAction { [weak self] _ in self?.onSkip?() }, for: .touchUpInside)
        flash.backgroundColor = HUD.accent.withAlphaComponent(150 / 255)
        flash.alpha = 0
        flash.isUserInteractionEnabled = false
        yehey.text = "YEHEY!"
        yehey.textAlignment = .center
        matched.font = HUD.font(30, .semibold)
        matched.textColor = HUD.ink
        matched.textAlignment = .center
        let guide = safeAreaLayoutGuide
        for v in [banner, roundHost, clip, clipCaption, skip, flash] as [UIView] {
            v.translatesAutoresizingMaskIntoConstraints = false
            addSubview(v)
        }
        banner.pin(bannerLabel, insets: UIEdgeInsets(top: 0, left: 18, bottom: 0, right: 18))
        let flashText = UIStackView(arrangedSubviews: [yehey, matched])
        flashText.axis = .vertical
        flashText.spacing = 8
        flashText.translatesAutoresizingMaskIntoConstraints = false
        flash.addSubview(flashText)
        NSLayoutConstraint.activate([
            banner.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            banner.topAnchor.constraint(equalTo: guide.topAnchor, constant: topInset),
            banner.widthAnchor.constraint(equalToConstant: 360),
            banner.heightAnchor.constraint(equalToConstant: 44),
            roundHost.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 14),
            roundHost.topAnchor.constraint(equalTo: guide.topAnchor, constant: topInset + 8),
            clip.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -14),
            clip.topAnchor.constraint(equalTo: banner.bottomAnchor, constant: 8),
            clip.widthAnchor.constraint(equalTo: guide.widthAnchor, multiplier: 0.24),
            clip.heightAnchor.constraint(equalTo: clip.widthAnchor, multiplier: 0.75),
            clipCaption.leadingAnchor.constraint(equalTo: clip.leadingAnchor),
            clipCaption.trailingAnchor.constraint(equalTo: clip.trailingAnchor),
            clipCaption.topAnchor.constraint(equalTo: clip.bottomAnchor),
            clipCaption.heightAnchor.constraint(equalToConstant: 20),
            skip.centerXAnchor.constraint(equalTo: guide.centerXAnchor),
            skip.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -8),
            skip.widthAnchor.constraint(equalToConstant: 180),
            skip.heightAnchor.constraint(equalToConstant: 44),
            flash.leadingAnchor.constraint(equalTo: leadingAnchor),
            flash.trailingAnchor.constraint(equalTo: trailingAnchor),
            flash.topAnchor.constraint(equalTo: topAnchor),
            flash.bottomAnchor.constraint(equalTo: bottomAnchor),
            flashText.centerXAnchor.constraint(equalTo: flash.centerXAnchor),
            flashText.centerYAnchor.constraint(equalTo: flash.centerYAnchor),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        let view = super.hitTest(point, with: event)
        return view === self || view === flash ? nil : view
    }

    func show(target: String, index: Int, total: Int, score: Int, attempts: Int, missed: String?) {
        if let missed {
            bannerLabel.text = "SAW  \(missed)  ·  TRY AGAIN"
            bannerLabel.textColor = HUD.amber
            banner.layer.borderColor = HUD.amber.withAlphaComponent(90 / 255).cgColor
        } else {
            bannerLabel.text = "SIGN THIS:  \(target)"
            bannerLabel.textColor = HUD.white
            banner.layer.borderColor = HUD.white.withAlphaComponent(90 / 255).cgColor
        }
        roundHost.subviews.forEach { $0.removeFromSuperview() }
        roundHost.pin(ChipView(text: "ROUND \(index + 1) OF \(total)   ·   SCORE \(score)/\(attempts)",
                               dotColor: nil, textColor: HUD.white, size: 12))
        if clipLabel != target {
            clipLabel = target
            clip.play(GlossCatalog.example(target)?.loop)
        }
    }

    func celebrate(_ gloss: String) {
        matched.text = "✓  \(gloss)"
        flash.layer.removeAllAnimations()
        flash.alpha = 1
        UIView.animate(withDuration: 1.4, delay: 0, options: [.curveEaseIn]) { self.flash.alpha = 0 }
    }

    func stop() {
        clip.stop()
        clipLabel = nil
    }
}

// MARK: - History

private final class HistoryCell: UITableViewCell {
    let box = UIView()
    let when = UILabel()
    let length = UILabel()
    let said = UILabel()
    let badge = ChipView(text: "unfinished", dotColor: nil, textColor: HUD.amber, size: 11)

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        backgroundColor = .clear
        selectionStyle = .none
        box.backgroundColor = HUD.ink.withAlphaComponent(HUD.panelAlpha)
        box.layer.cornerRadius = 10
        box.layer.borderWidth = 1
        box.layer.borderColor = HUD.muted.withAlphaComponent(90 / 255).cgColor
        when.font = HUD.font(14, .semibold)
        when.textColor = HUD.white
        length.font = HUD.mono(13, .medium)
        length.textColor = HUD.muted
        said.font = HUD.font(13, .medium)
        for v in [box, when, length, said, badge] as [UIView] { v.translatesAutoresizingMaskIntoConstraints = false; contentView.addSubview(v) }
        NSLayoutConstraint.activate([
            box.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            box.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            box.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 4),
            box.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -4),
            when.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: 16),
            when.topAnchor.constraint(equalTo: box.topAnchor, constant: 10),
            length.trailingAnchor.constraint(equalTo: box.trailingAnchor, constant: -16),
            length.centerYAnchor.constraint(equalTo: when.centerYAnchor),
            said.leadingAnchor.constraint(equalTo: when.leadingAnchor),
            said.topAnchor.constraint(equalTo: when.bottomAnchor, constant: 8),
            said.trailingAnchor.constraint(lessThanOrEqualTo: badge.leadingAnchor, constant: -12),
            badge.trailingAnchor.constraint(equalTo: length.trailingAnchor),
            badge.centerYAnchor.constraint(equalTo: said.centerYAnchor),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }
}

final class HistoryPage: UIView, UITableViewDataSource {
    private let table = UITableView(frame: .zero, style: .plain)
    private let heading = UILabel()
    private let empty = UILabel()
    private var rows: [ShellSessionStore.Row] = []
    private let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "dd MMM yyyy · HH:mm"
        return f
    }()

    init() {
        super.init(frame: .zero)
        backgroundColor = HUD.ink
        heading.font = HUD.font(14, .semibold)
        heading.textColor = HUD.white
        table.backgroundColor = .clear
        table.separatorStyle = .none
        table.rowHeight = 74
        table.dataSource = self
        table.indicatorStyle = .white
        table.register(HistoryCell.self, forCellReuseIdentifier: "row")
        empty.numberOfLines = 2
        empty.textAlignment = .center
        empty.attributedText = {
            let text = NSMutableAttributedString(string: "No sessions yet\n", attributes: [.font: HUD.font(20, .semibold), .foregroundColor: HUD.muted])
            text.append(NSAttributedString(string: "Recordings appear here after you use the camera", attributes: [.font: HUD.font(13, .medium), .foregroundColor: HUD.muted]))
            return text
        }()
        let footer = UILabel()
        footer.text = "Saved on this iPhone · scores are uncalibrated"
        footer.font = HUD.font(11, .medium)
        footer.textColor = HUD.muted
        let guide = safeAreaLayoutGuide
        for v in [heading, table, empty, footer] as [UIView] { v.translatesAutoresizingMaskIntoConstraints = false; addSubview(v) }
        NSLayoutConstraint.activate([
            heading.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 18),
            heading.topAnchor.constraint(equalTo: guide.topAnchor, constant: 52),
            table.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 18),
            table.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -18),
            table.topAnchor.constraint(equalTo: heading.bottomAnchor, constant: 8),
            table.bottomAnchor.constraint(equalTo: footer.topAnchor, constant: -6),
            empty.centerXAnchor.constraint(equalTo: table.centerXAnchor),
            empty.centerYAnchor.constraint(equalTo: table.centerYAnchor),
            footer.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 18),
            footer.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -6),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func reload() {
        rows = ShellSessionStore.rows()
        heading.text = "\(rows.count) recorded session\(rows.count == 1 ? "" : "s")"
        empty.isHidden = !rows.isEmpty
        table.reloadData()
    }

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int { rows.count }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "row", for: indexPath) as! HistoryCell
        let row = rows[indexPath.row]
        cell.when.text = formatter.string(from: row.started)
        cell.length.text = String(format: "%d:%02d", Int(row.seconds) / 60, Int(row.seconds) % 60)
        cell.said.text = row.said
        cell.said.textColor = row.hasSentence ? HUD.white : HUD.muted
        cell.badge.isHidden = row.complete
        return cell
    }
}
