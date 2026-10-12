// ATLAS visual tokens and primitives (Figma "ATLAS — Brand + Mobile Experience", 03 · Components):
// the colour and radius variables, the Inter type ramp, Control / Button (primary, secondary,
// danger, disabled), Chrome / Navigation, chips and the shield note.
import 'package:flutter/material.dart';

class Atlas {
  static const ink = Color(0xFF08265C);
  static const muted = Color(0xFF52698C);
  static const blue = Color(0xFF0769E8);
  static const sky = Color(0xFF27A8FF);
  static const pale = Color(0xFFEAF5FF);
  static const gold = Color(0xFFFFBD36);
  static const cream = Color(0xFFFFF4E0);
  static const white = Color(0xFFFFFFFF);
  static const canvas = Color(0xFFF5F9FE);
  static const line = Color(0xFFD9E4F1);
  static const green = Color(0xFF17845F);
  static const danger = Color(0xFFD73548);
  static const night = Color(0xFF091B32);

  static const font = 'Inter';

  static TextStyle _t(double size, FontWeight weight, double height, Color color, [double spacing = 0]) =>
      TextStyle(
        fontFamily: font,
        fontSize: size,
        fontWeight: weight,
        height: height / size,
        color: color,
        letterSpacing: spacing,
      );

  // The eight ATLAS text styles.
  static TextStyle display([Color c = ink]) => _t(56, FontWeight.w800, 64, c);
  static TextStyle title([Color c = ink]) => _t(32, FontWeight.w700, 40, c);
  static TextStyle heading([Color c = ink]) => _t(24, FontWeight.w600, 32, c);
  static TextStyle subheading([Color c = ink]) => _t(20, FontWeight.w600, 28, c);
  static TextStyle body([Color c = muted]) => _t(16, FontWeight.w400, 24, c);
  static TextStyle button([Color c = ink]) => _t(16, FontWeight.w600, 24, c);
  static TextStyle caption([Color c = muted]) => _t(13, FontWeight.w400, 20, c);
  static TextStyle label([Color c = muted]) => _t(11, FontWeight.w600, 16, c);

  /// The wordmark: Extra Bold with 8% tracking.
  static TextStyle wordmark(double size) => _t(size, FontWeight.w800, size * 8 / 7, ink, size * 0.08);
}

enum AtlasButtonStyle { primary, secondary, danger }

/// Control / Button. `onPressed == null` draws the Disabled variant.
class AtlasButton extends StatelessWidget {
  const AtlasButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.style = AtlasButtonStyle.primary,
    this.height = 56,
    this.width,
  });

  final String label;
  final VoidCallback? onPressed;
  final AtlasButtonStyle style;
  final double height;
  final double? width;

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null;
    final (fill, text) = !enabled
        ? (Atlas.line, Atlas.ink)
        : switch (style) {
            AtlasButtonStyle.primary => (Atlas.blue, Atlas.white),
            AtlasButtonStyle.secondary => (Atlas.pale, Atlas.ink),
            AtlasButtonStyle.danger => (Atlas.danger, Atlas.white),
          };
    return Opacity(
      opacity: enabled ? 1 : 0.6,
      child: Material(
        color: fill,
        borderRadius: BorderRadius.circular(16),
        child: InkWell(
          onTap: onPressed,
          borderRadius: BorderRadius.circular(16),
          child: Container(
            width: width,
            height: height,
            alignment: Alignment.center,
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Text(label, style: Atlas.button(text), maxLines: 1, overflow: TextOverflow.ellipsis),
          ),
        ),
      ),
    );
  }
}

/// Chrome / Navigation: back chevron and the page title.
class NavHeader extends StatelessWidget {
  const NavHeader({super.key, required this.title, required this.onBack});

  final String title;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) => SizedBox(
    height: 52,
    child: Row(
      children: [
        Semantics(
          button: true,
          label: 'Back',
          child: InkWell(
            onTap: onBack,
            customBorder: const CircleBorder(),
            child: const SizedBox(
              width: 36,
              height: 44,
              child: Align(
                alignment: Alignment.centerLeft,
                child: Icon(Icons.arrow_back_ios_new_rounded, size: 20, color: Atlas.ink),
              ),
            ),
          ),
        ),
        Expanded(child: Text(title, style: Atlas.subheading(), overflow: TextOverflow.ellipsis)),
      ],
    ),
  );
}

/// A small rounded label: pale/blue by default, cream/ink for warnings.
class AtlasChip extends StatelessWidget {
  const AtlasChip(this.text, {super.key, this.fill = Atlas.pale, this.color = Atlas.blue});

  const AtlasChip.warm(this.text, {super.key}) : fill = Atlas.cream, color = Atlas.ink;

  final String text;
  final Color fill;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(8),
    decoration: BoxDecoration(color: fill, borderRadius: BorderRadius.circular(12)),
    child: Text(text, style: Atlas.label(color)),
  );
}

/// The shield reassurance strip at the foot of Home and History.
class ShieldNote extends StatelessWidget {
  const ShieldNote(this.text, {super.key, this.fill = Atlas.canvas});

  final String text;
  final Color fill;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(color: fill, borderRadius: BorderRadius.circular(16)),
    child: Row(
      children: [
        const Icon(Icons.verified_user_outlined, size: 18, color: Atlas.green),
        const SizedBox(width: 8),
        Expanded(child: Text(text, style: Atlas.label())),
      ],
    ),
  );
}

/// Centred two-line empty state.
class EmptyState extends StatelessWidget {
  const EmptyState({super.key, required this.title, required this.hint});

  final String title;
  final String hint;

  @override
  Widget build(BuildContext context) => Center(
    child: Padding(
      padding: const EdgeInsets.symmetric(vertical: 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(title, style: Atlas.subheading(Atlas.muted), textAlign: TextAlign.center),
          const SizedBox(height: 8),
          Text(hint, style: Atlas.caption(), textAlign: TextAlign.center),
        ],
      ),
    ),
  );
}

/// Screen content padding shared by every portrait page (24 sides, 8 top, 16 bottom).
const pagePadding = EdgeInsets.fromLTRB(24, 8, 24, 16);

/// White page with the navigation header, a body that fills, and an optional pinned footer.
class PortraitScaffold extends StatelessWidget {
  const PortraitScaffold({super.key, required this.header, required this.body, this.footer});

  final Widget header;
  final Widget body;
  final Widget? footer;

  @override
  Widget build(BuildContext context) => Material(
    color: Atlas.white,
    child: SafeArea(
      child: Padding(
        padding: pagePadding,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            header,
            Expanded(child: body),
            if (footer != null) ...[const SizedBox(height: 16), footer!],
          ],
        ),
      ),
    ),
  );
}
