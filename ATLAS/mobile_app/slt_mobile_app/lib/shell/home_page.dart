// SPLASH and HOME (Figma 01 / SPLASH, 02 / HOME). Home keeps the four entry points: Live
// ("Start translating" waits until the Reel models are warm), Practice, Glosses and History,
// then the "How to use ATLAS" row into the manual.
import 'package:flutter/material.dart';

import 'style.dart';

const brandSymbol = 'assets/brand/atlas_symbol.png';

class SplashPage extends StatefulWidget {
  const SplashPage({super.key});

  @override
  State<SplashPage> createState() => _SplashPageState();
}

class _SplashPageState extends State<SplashPage> with SingleTickerProviderStateMixin {
  late final AnimationController _progress = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  )..forward();

  @override
  void dispose() {
    _progress.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SizedBox.expand(
    child: ColoredBox(
      color: Atlas.pale,
      child: SafeArea(
        child: Padding(
          padding: pagePadding,
          child: Column(
            children: [
              const Spacer(),
              Image.asset(brandSymbol, width: 164, height: 164),
              const SizedBox(height: 16),
              Text('ATLAS', style: Atlas.wordmark(56), textAlign: TextAlign.center),
              const SizedBox(height: 16),
              Text('Sign Language Translation', style: Atlas.caption(), textAlign: TextAlign.center),
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(2),
                child: SizedBox(
                  width: 120,
                  height: 4,
                  child: AnimatedBuilder(
                    animation: _progress,
                    builder: (context, _) => LinearProgressIndicator(
                      value: Curves.easeOut.transform(_progress.value),
                      backgroundColor: Atlas.line,
                      color: Atlas.sky,
                    ),
                  ),
                ),
              ),
              const Spacer(),
              Text('Building a more\nconnected tomorrow.', style: Atlas.body(), textAlign: TextAlign.center),
              const Spacer(),
            ],
          ),
        ),
      ),
    ),
  );
}

class HomePage extends StatelessWidget {
  const HomePage({
    super.key,
    required this.ready,
    required this.status,
    required this.sessionCount,
    required this.glossCount,
    required this.onAction,
    required this.onSettings,
  });

  final bool ready;
  final String status;
  final int sessionCount;
  final int glossCount;
  final ValueChanged<String> onAction;
  final VoidCallback onSettings;

  @override
  Widget build(BuildContext context) {
    final count = glossCount == 0 ? 100 : glossCount;
    return SafeArea(
      child: LayoutBuilder(
        builder: (context, constraints) => SingleChildScrollView(
          padding: pagePadding,
          child: ConstrainedBox(
            constraints: BoxConstraints(minHeight: constraints.maxHeight - pagePadding.vertical),
            child: IntrinsicHeight(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  SizedBox(
                    height: 42,
                    child: Row(
                      children: [
                        Image.asset(brandSymbol, width: 42, height: 42),
                        const SizedBox(width: 12),
                        Expanded(child: Text('ATLAS', style: Atlas.wordmark(20))),
                        IconButton(
                          tooltip: 'Settings',
                          onPressed: onSettings,
                          icon: const Icon(Icons.settings_outlined, color: Atlas.ink),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text('Hello.', style: Atlas.title()),
                  const SizedBox(height: 8),
                  Text('Sign, learn, and connect\nwith a more inclusive world.', style: Atlas.body()),
                  const SizedBox(height: 16),
                  _ActionCard(
                    icon: Icons.back_hand_outlined,
                    title: 'Start translating',
                    subtitle: ready ? 'LIVE · Ready when you are' : 'LIVE · ${status.isEmpty ? 'Warming up…' : status}',
                    gradient: const LinearGradient(colors: [Atlas.sky, Atlas.blue]),
                    foreground: Atlas.white,
                    muted: Atlas.white,
                    height: 136,
                    chevron: true,
                    onTap: ready ? () => onAction('LIVE') : null,
                  ),
                  const SizedBox(height: 16),
                  _ActionCard(
                    icon: Icons.signal_cellular_alt_rounded,
                    title: 'Practice',
                    subtitle: 'Build your confidence',
                    fill: Atlas.pale,
                    height: 118,
                    chevron: true,
                    onTap: () => onAction('PRACTICE'),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: _ActionCard(
                          icon: Icons.menu_book_outlined,
                          title: 'Glosses',
                          subtitle: '$count signs',
                          fill: Atlas.cream,
                          height: 128,
                          onTap: () => onAction('GLOSSES'),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: _ActionCard(
                          icon: Icons.schedule_rounded,
                          title: 'History',
                          subtitle: sessionCount > 0
                              ? '$sessionCount session${sessionCount == 1 ? '' : 's'}'
                              : 'Your sessions',
                          fill: Atlas.cream,
                          height: 128,
                          onTap: () => onAction('HISTORY'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  _GuideRow(onTap: () => onAction('MANUAL')),
                  const Spacer(),
                  const SizedBox(height: 16),
                  const ShieldNote('All processing stays on your device.'),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Row / Home guide: the way into the manual.
class _GuideRow extends StatelessWidget {
  const _GuideRow({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Material(
    color: Atlas.white,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(20),
      side: const BorderSide(color: Atlas.line),
    ),
    child: InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: SizedBox(
        height: 72,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Row(
            children: [
              const Icon(Icons.help_outline_rounded, size: 24, color: Atlas.blue),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('How to use ATLAS', style: Atlas.button(), maxLines: 1, overflow: TextOverflow.ellipsis),
                    Text('A guide to every screen and sign', style: Atlas.caption(), maxLines: 1, overflow: TextOverflow.ellipsis),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              const Icon(Icons.chevron_right_rounded, size: 24, color: Atlas.ink),
            ],
          ),
        ),
      ),
    ),
  );
}

/// Card / Home action.
class _ActionCard extends StatelessWidget {
  const _ActionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.height,
    required this.onTap,
    this.fill,
    this.gradient,
    this.foreground = Atlas.ink,
    this.muted = Atlas.muted,
    this.chevron = false,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final double height;
  final VoidCallback? onTap;
  final Color? fill;
  final Gradient? gradient;
  final Color foreground;
  final Color muted;
  final bool chevron;

  @override
  Widget build(BuildContext context) => Opacity(
    opacity: onTap == null ? 0.7 : 1,
    child: Container(
      height: height,
      decoration: BoxDecoration(color: fill, gradient: gradient, borderRadius: BorderRadius.circular(24)),
      child: Material(
        type: MaterialType.transparency,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(24),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(icon, size: 28, color: foreground),
                      const SizedBox(height: 8),
                      Text(title, style: Atlas.subheading(foreground), maxLines: 1, overflow: TextOverflow.ellipsis),
                      Text(subtitle, style: Atlas.caption(muted), maxLines: 1, overflow: TextOverflow.ellipsis),
                    ],
                  ),
                ),
                if (chevron) ...[
                  const SizedBox(width: 16),
                  Icon(Icons.chevron_right_rounded, size: 24, color: foreground),
                ],
              ],
            ),
          ),
        ),
      ),
    ),
  );
}
