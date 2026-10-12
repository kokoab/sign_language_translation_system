// Visual tokens and primitives shared by every page, taken from the desktop app shell
// (scripts/reel_hud_v17.py): INK / WHITE / MUTED / ACCENT / AMBER, translucent panels,
// dotted chips, pill buttons (primary = accent, secondary = ink with a hairline), shadowed text.

import UIKit

enum HUD {
    static let ink = UIColor(red: 14 / 255, green: 15 / 255, blue: 19 / 255, alpha: 1)
    static let white = UIColor(red: 245 / 255, green: 246 / 255, blue: 248 / 255, alpha: 1)
    static let muted = UIColor(red: 138 / 255, green: 143 / 255, blue: 152 / 255, alpha: 1)
    static let accent = UIColor(red: 61 / 255, green: 220 / 255, blue: 132 / 255, alpha: 1)
    static let amber = UIColor(red: 245 / 255, green: 165 / 255, blue: 36 / 255, alpha: 1)
    static let red = UIColor(red: 235 / 255, green: 77 / 255, blue: 75 / 255, alpha: 1)
    static let panelAlpha: CGFloat = 168 / 255
    static let chipAlpha: CGFloat = 150 / 255

    static func font(_ size: CGFloat, _ weight: UIFont.Weight) -> UIFont { .systemFont(ofSize: size, weight: weight) }
    static func mono(_ size: CGFloat, _ weight: UIFont.Weight) -> UIFont { .monospacedSystemFont(ofSize: size, weight: weight) }
}

/// Rounded translucent pill with an optional leading dot (the desktop `chip`).
final class ChipView: UIView {
    let dot = UIView()
    let label = UILabel()

    init(text: String, dotColor: UIColor?, textColor: UIColor, alpha: CGFloat = HUD.panelAlpha, size: CGFloat = 14) {
        super.init(frame: .zero)
        backgroundColor = HUD.ink.withAlphaComponent(alpha)
        layer.cornerRadius = 13
        label.text = text
        label.textColor = textColor
        label.font = HUD.font(size, .semibold)
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
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 5),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -5),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }
}

/// The desktop `_button`: primary (accent fill, ink text), secondary (ink panel, white text,
/// hairline), disabled (muted), danger (red, for STOP).
final class PillButton: UIButton {
    enum Kind { case primary, secondary, disabled, danger, working }

    var kind: Kind = .secondary { didSet { style() } }

    init(_ title: String, kind: Kind = .secondary, size: CGFloat = 15) {
        super.init(frame: .zero)
        setTitle(title, for: .normal)
        titleLabel?.font = HUD.font(size, .semibold)
        self.kind = kind
        style()
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        layer.cornerRadius = bounds.height / 2
    }

    override var isHighlighted: Bool {
        didSet { alpha = isHighlighted ? 0.75 : 1 }
    }

    private func style() {
        layer.borderWidth = 0
        switch kind {
        case .primary:
            backgroundColor = HUD.accent.withAlphaComponent(235 / 255)
            setTitleColor(HUD.ink, for: .normal)
        case .secondary:
            backgroundColor = HUD.ink.withAlphaComponent(HUD.panelAlpha)
            setTitleColor(HUD.white, for: .normal)
            layer.borderWidth = 1
            layer.borderColor = HUD.white.withAlphaComponent(90 / 255).cgColor
        case .disabled:
            backgroundColor = HUD.ink.withAlphaComponent(HUD.chipAlpha)
            setTitleColor(HUD.muted, for: .normal)
        case .danger:
            backgroundColor = HUD.red.withAlphaComponent(235 / 255)
            setTitleColor(HUD.white, for: .normal)
        case .working:
            backgroundColor = HUD.amber.withAlphaComponent(235 / 255)
            setTitleColor(HUD.ink, for: .normal)
        }
        isEnabled = kind != .disabled
    }
}

/// Text with the desktop's drop shadow, readable on dark pages and bright camera frames.
final class ShadowLabel: UILabel {
    init(size: CGFloat, weight: UIFont.Weight, color: UIColor = HUD.white, shadow: CGFloat = 2) {
        super.init(frame: .zero)
        font = HUD.font(size, weight)
        textColor = color
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.85
        layer.shadowRadius = shadow
        layer.shadowOffset = CGSize(width: 0, height: shadow / 2)
    }

    required init?(coder: NSCoder) { fatalError() }
}

extension UIView {
    func pin(_ view: UIView, insets: UIEdgeInsets = .zero) {
        view.translatesAutoresizingMaskIntoConstraints = false
        addSubview(view)
        NSLayoutConstraint.activate([
            view.leadingAnchor.constraint(equalTo: leadingAnchor, constant: insets.left),
            view.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -insets.right),
            view.topAnchor.constraint(equalTo: topAnchor, constant: insets.top),
            view.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -insets.bottom),
        ])
    }
}
