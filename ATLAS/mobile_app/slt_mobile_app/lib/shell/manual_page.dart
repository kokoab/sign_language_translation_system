// MANUAL: "How to use ATLAS", opened on demand from Home or the ? on the Live/Practice camera
// screen. A hub of section cards, then one page per section built from the copy below. The camera
// screens are drawn here (not screenshots) with numbered callouts, and the control signs are drawn
// hands. Copy is plain language for the general public and names no platform.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'style.dart';

// MARK: - Copy

sealed class ManualBlock {
  const ManualBlock();
}

class ManualPara extends ManualBlock {
  const ManualPara(this.text);

  final String text;
}

class ManualSteps extends ManualBlock {
  const ManualSteps(this.steps, {this.title});

  final String? title;
  final List<String> steps;
}

class ManualTip extends ManualBlock {
  const ManualTip(this.text);

  final String text;
}

/// Term → meaning rows under a small caps title.
class ManualTerms extends ManualBlock {
  const ManualTerms(this.title, this.rows);

  final String title;
  final List<(String, String)> rows;
}

enum ManualScreen { live, practice }

/// A drawn camera screen with numbered callouts and their legend.
class ManualDiagram extends ManualBlock {
  const ManualDiagram(this.screen);

  final ManualScreen screen;
}

enum SignalTone { dim, white, gold, blue }

/// The status labels at the top of the camera screen, in their own colours.
class ManualSignals extends ManualBlock {
  const ManualSignals(this.title, this.rows);

  final String title;
  final List<(String, SignalTone, String)> rows;
}

enum GestureArt { finish }

class ManualGesture extends ManualBlock {
  const ManualGesture(this.art, this.title, this.how, this.does, {this.watch});

  final GestureArt art;
  final String title;
  final String how;
  final String does;

  /// A gloss with a reference clip in Glosses.
  final String? watch;
}

class ManualSection {
  const ManualSection(this.id, this.icon, this.title, this.summary, this.blocks);

  final String id;
  final IconData icon;
  final String title;
  final String summary;
  final List<ManualBlock> blocks;
}

ManualSection? manualSection(String? id) => manualSections.where((s) => s.id == id).firstOrNull;

const manualSections = [
  ManualSection('start', Icons.flag_outlined, 'Before you start', 'Set up your phone and your space.', [
    ManualPara('ATLAS reads American Sign Language (ASL) from your camera and turns it into English '
        'words and sentences.'),
    ManualSteps(title: 'SET UP', [
      'Open ATLAS and wait until Start translating says “Ready when you are”. It takes a few seconds to get ready.',
      'Portrait works too; landscape is recommended to fit both hands. Prop your phone up so it stays still.',
      'Step back until your head, your shoulders and both hands fit on the screen.',
      'Face the light. A bright window behind you makes your hands hard to see.',
    ]),
    ManualTerms('GOOD TO KNOW', [
      ('100 signs', 'ATLAS knows 100 signs. Browse them in Glosses. Other signs will not be recognised.'),
      ('One signer', 'If several people are in view, ATLAS follows the largest face, usually the person nearest the camera.'),
      ('Private', 'Everything is processed on your phone.'),
      ('Tips', 'The first time you open Live, a few tips explain the screen. Settings (the gear on Home) can '
          'show them again.'),
      ('Hints', 'If it is too dark, or ATLAS cannot see you or your hands, Live shows a hint in the middle '
          'of the screen. Follow it and it goes away.'),
    ]),
  ]),
  ManualSection('live', Icons.back_hand_outlined, 'Live translation', 'Turn your signing into English.', [
    ManualSteps([
      'On Home, tap Start translating. It opens in portrait; landscape is recommended.',
      'Tap Start. ATLAS does not translate anything until you do.',
      'Sign one sign at a time at a steady pace, with a short pause between signs.',
      'Your signs appear as word pills at the top left, and the English appears at the bottom as you sign.',
      'Tap Finish when your sentence is done. ATLAS completes the sentence and reads it aloud.',
      'Tap Start for the next sentence, or Back to go Home.',
    ]),
    ManualPara('Use Flip camera in Live to switch between front and back. Practice uses the front camera. '
        'The More menu (•••, top right) holds Show landmarks (the tracking points), Show signs (Live: '
        'turn it off to keep only the English on screen) and How to use ATLAS. ATLAS remembers your choices.'),
    ManualTip('Hands-free: instead of tapping Finish, hold the finish pose. ATLAS finishes the sentence '
        'and keeps translating. See Gestures and special signs.'),
    ManualTerms('BUTTONS', [
      ('Start / Finish', 'Start translating, or finish the sentence and stop.'),
      ('Reset (↺)', 'Clears the signs so far so you can start over. Press and hold it to see what it does.'),
      ('Speak again', 'After Finish, reads the sentence aloud again.'),
      ('Back', 'Leaves the camera. Your session is saved to History.'),
      ('More (•••)', 'Show landmarks, Show signs and How to use ATLAS. The guide, like Back, leaves the camera and '
          'saves your session.'),
    ]),
  ]),
  ManualSection('screen', Icons.visibility_outlined, 'Reading the screen', 'What each label and colour means.', [
    ManualDiagram(ManualScreen.live),
    ManualSignals('STATUS', [
      ('Warming up', SignalTone.dim, 'ATLAS is getting ready. Wait a moment.'),
      ('Ready', SignalTone.dim, 'Tap Start when you are ready.'),
      ('Watching', SignalTone.dim, 'Translating, but no hands are in view.'),
      ('Signing', SignalTone.white, 'ATLAS can see your hands.'),
      ('Reading sign…', SignalTone.gold, 'ATLAS is working out a sign.'),
      ('Hold to finish 60%', SignalTone.gold, 'You are holding the finish pose. Keep still.'),
      ('Lower your hands', SignalTone.gold, 'The sentence is finishing. Relax your hands to carry on.'),
      ('Translating…', SignalTone.gold, 'Turning your signs into a sentence.'),
    ]),
    ManualTerms('COLOURS', [
      ('Amber', 'Still deciding. The sign may change.'),
      ('White', 'Accepted.'),
      ('Green dot', 'The newest sign, or the sentence being read aloud.'),
    ]),
  ]),
  ManualSection('gestures', Icons.pan_tool_outlined, 'Gestures and special signs', 'Finish a sentence without touching the phone.', [
    ManualGesture(
      GestureArt.finish,
      'Finish pose',
      'Raise both hands above your shoulders: your right hand open with the fingers up, your left hand '
          'in a fist. Hold still for one second.',
      'Finishes the sentence and reads it aloud without touching the phone. The status counts up '
          '“Hold to finish”, then shows “Lower your hands”. Lower them to carry on. Works in Live only.',
    ),
  ]),
  ManualSection('practice', Icons.signal_cellular_alt_rounded, 'Practice', 'Learn signs with instant feedback.', [
    ManualSteps([
      'On Home, tap Practice and choose 5, 10 or 20 signs.',
      'Tap Start practice. It opens in portrait and starts straight away; landscape is recommended.',
      'Watch the reference clip (it plays at 0.75× speed), then sign the word shown. Tap the clip to enlarge it; close with X or tap outside.',
      'A match shows “Matched · +1” in green and moves on by itself. Tap Next to move on sooner.',
      'If ATLAS saw a different sign, it shows “Saw … · try again” in red. Sign it again.',
      'Tap Skip to move on. A skipped sign counts as missed.',
    ]),
    ManualDiagram(ManualScreen.practice),
    ManualPara('At the end you see your score and the signs that were not recognised. Tap a missed sign to '
        'watch it in Glosses, or tap Practice again for a new round.'),
    ManualTip('A miss may be ATLAS, not you. Treat the score as a guide, not a grade.'),
  ]),
  ManualSection('glosses', Icons.menu_book_outlined, 'Glosses', 'Browse the 100 signs ATLAS knows.', [
    ManualPara('A gloss is the written name of a sign, in capital letters, such as HELLO. Glosses lists '
        'every sign ATLAS knows.'),
    ManualSteps([
      'Search by name, or scroll through the list.',
      'Tap a sign to watch its example clip. Tap the clip to pause it.',
      'Clips play at 0.75× speed. Choose another speed to slow down or speed up.',
      'Tap Practice signs to practise just that sign.',
    ]),
  ]),
  ManualSection('history', Icons.schedule_rounded, 'History', 'Look back at past sessions.', [
    ManualPara('Every Live session is saved in History when you leave the camera.'),
    ManualSteps([
      'Tap a session to see what was said, then its signs and a timeline. Tap Speak again to hear it.',
      'A session marked Unfinished was left before Finish.',
      'Delete session removes one session; Clear all removes them all.',
    ]),
    ManualTerms('GOOD TO KNOW', [
      ('Saved on this phone', 'Sessions stay on your phone. Videos are kept only if you turn on Save Live videos '
          'in Settings.'),
    ]),
  ]),
  ManualSection('help', Icons.help_outline_rounded, 'Troubleshooting', 'Fixes for common problems.', [
    ManualTerms('IF SOMETHING GOES WRONG', [
      ('Start translating is greyed out', 'ATLAS is still getting ready. Wait a few seconds.'),
      ('A sign is not recognised', 'Check it is in Glosses. Sign at a steady pace, keep both hands on screen and add light.'),
      ('The wrong word appeared', 'Tap ↺ to clear it and sign again.'),
      ('A hint keeps showing', 'Follow it: add light, step back so both shoulders show, or bring your hands '
          'back inside the frame.'),
      ('The sentence finished by itself', 'You may have held the finish pose. Lower your hands between sentences.'),
      ('Someone else is being followed', 'Move closer to the camera, or ask others to step out of view.'),
    ]),
  ]),
];

// MARK: - Pages

class ManualPage extends StatelessWidget {
  const ManualPage({super.key, required this.onSelect, required this.onBack});

  final ValueChanged<String> onSelect;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) => PortraitScaffold(
    header: NavHeader(title: 'How to use ATLAS', onBack: onBack),
    body: ListView(
      padding: EdgeInsets.zero,
      children: [
        const SizedBox(height: 16),
        Text('Your guide\nto ATLAS.', style: Atlas.title()),
        const SizedBox(height: 8),
        Text('Each screen, the signs that control ATLAS, and what every colour means.', style: Atlas.body()),
        const SizedBox(height: 16),
        for (final (index, section) in manualSections.indexed) ...[
          if (index > 0) const SizedBox(height: 12),
          _SectionCard(section: section, onTap: () => onSelect(section.id)),
        ],
        const SizedBox(height: 16),
      ],
    ),
  );
}

/// Card / Manual section.
class _SectionCard extends StatelessWidget {
  const _SectionCard({required this.section, required this.onTap});

  final ManualSection section;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Material(
    color: Atlas.canvas,
    borderRadius: BorderRadius.circular(20),
    child: InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: const BoxDecoration(color: Atlas.white, shape: BoxShape.circle),
              child: Icon(section.icon, size: 22, color: Atlas.blue),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(section.title, style: Atlas.button()),
                  Text(section.summary, style: Atlas.caption()),
                ],
              ),
            ),
            const SizedBox(width: 8),
            const Icon(Icons.chevron_right_rounded, size: 24, color: Atlas.ink),
          ],
        ),
      ),
    ),
  );
}

class ManualSectionPage extends StatelessWidget {
  const ManualSectionPage({
    super.key,
    required this.section,
    required this.onOpen,
    required this.onWatch,
    required this.onBack,
  });

  final ManualSection section;

  /// Another section, by id (the Next button).
  final ValueChanged<String> onOpen;

  /// A gloss's reference clip.
  final ValueChanged<String> onWatch;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    final index = manualSections.indexOf(section);
    final next = index + 1 < manualSections.length ? manualSections[index + 1] : null;
    return PortraitScaffold(
      header: NavHeader(title: section.title, onBack: onBack),
      body: ListView(
        padding: EdgeInsets.zero,
        children: [
          const SizedBox(height: 16),
          Align(alignment: Alignment.centerLeft, child: AtlasChip('${index + 1} OF ${manualSections.length}')),
          const SizedBox(height: 16),
          Text(section.summary, style: Atlas.heading()),
          for (final block in section.blocks) ...[const SizedBox(height: 24), _block(block)],
          const SizedBox(height: 32),
          if (next != null)
            AtlasButton(label: 'Next: ${next.title}', style: AtlasButtonStyle.secondary, onPressed: () => onOpen(next.id))
          else
            AtlasButton(label: 'Back to the guide', style: AtlasButtonStyle.secondary, onPressed: onBack),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _block(ManualBlock block) => switch (block) {
    ManualPara(:final text) => Text(text, style: Atlas.body()),
    ManualSteps() => _Steps(block),
    ManualTip(:final text) => _Tip(text),
    ManualTerms() => _Terms(block),
    ManualDiagram(:final screen) => ScreenDiagram(screen: screen),
    ManualSignals() => _Signals(block),
    ManualGesture() => _GestureCard(block, onWatch: onWatch),
  };
}

class _Steps extends StatelessWidget {
  const _Steps(this.block);

  final ManualSteps block;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      if (block.title != null) ...[Text(block.title!, style: Atlas.label()), const SizedBox(height: 12)],
      for (final (index, step) in block.steps.indexed) ...[
        if (index > 0) const SizedBox(height: 12),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 28,
              height: 28,
              alignment: Alignment.center,
              decoration: const BoxDecoration(color: Atlas.pale, shape: BoxShape.circle),
              child: Text('${index + 1}', style: Atlas.label(Atlas.blue)),
            ),
            const SizedBox(width: 12),
            Expanded(child: Text(step, style: Atlas.body(Atlas.ink))),
          ],
        ),
      ],
    ],
  );
}

class _Tip extends StatelessWidget {
  const _Tip(this.text);

  final String text;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(color: Atlas.cream, borderRadius: BorderRadius.circular(20)),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Icon(Icons.lightbulb_outline_rounded, size: 22, color: Atlas.ink),
        const SizedBox(width: 12),
        Expanded(child: Text(text, style: Atlas.body(Atlas.ink))),
      ],
    ),
  );
}

class _Terms extends StatelessWidget {
  const _Terms(this.block);

  final ManualTerms block;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Text(block.title, style: Atlas.label()),
      for (final (term, meaning) in block.rows) ...[
        const SizedBox(height: 12),
        Text(term, style: Atlas.button()),
        Text(meaning, style: Atlas.body()),
      ],
    ],
  );
}

class _Signals extends StatelessWidget {
  const _Signals(this.block);

  final ManualSignals block;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Text(block.title, style: Atlas.label()),
      for (final (text, tone, meaning) in block.rows) ...[
        const SizedBox(height: 12),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 136,
              child: Align(
                alignment: Alignment.centerLeft,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(color: Atlas.night, borderRadius: BorderRadius.circular(12)),
                  child: Text(text, style: Atlas.label(_toneColor(tone))),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(child: Text(meaning, style: Atlas.caption(Atlas.ink))),
          ],
        ),
      ],
    ],
  );
}

Color _toneColor(SignalTone tone) => switch (tone) {
  SignalTone.dim => Atlas.white.withValues(alpha: 0.7),
  SignalTone.white => Atlas.white,
  SignalTone.gold => Atlas.gold,
  SignalTone.blue => Atlas.sky,
};

class _GestureCard extends StatelessWidget {
  const _GestureCard(this.block, {required this.onWatch});

  final ManualGesture block;
  final ValueChanged<String> onWatch;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(color: Atlas.canvas, borderRadius: BorderRadius.circular(24)),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        GestureIllustration(art: block.art),
        const SizedBox(height: 16),
        Text(block.title, style: Atlas.subheading()),
        const SizedBox(height: 12),
        Text('HOW', style: Atlas.label()),
        const SizedBox(height: 4),
        Text(block.how, style: Atlas.body(Atlas.ink)),
        const SizedBox(height: 12),
        Text('WHAT IT DOES', style: Atlas.label()),
        const SizedBox(height: 4),
        Text(block.does, style: Atlas.body()),
        if (block.watch != null) ...[
          const SizedBox(height: 16),
          AtlasButton(
            label: 'Watch ${block.watch} in Glosses',
            style: AtlasButtonStyle.secondary,
            height: 48,
            onPressed: () => onWatch(block.watch!),
          ),
        ],
      ],
    ),
  );
}

// MARK: - Camera screen diagrams

/// The landscape camera screen, drawn at 760 × 352 and scaled to the page width, with numbered
/// callouts and their legend underneath.
class ScreenDiagram extends StatelessWidget {
  const ScreenDiagram({super.key, required this.screen});

  final ManualScreen screen;

  static const _liveLegend = [
    ('Status', 'What ATLAS is doing right now. See Status below.'),
    ('Flip camera', 'Switches between front and back cameras in Live.'),
    ('More (•••)', 'Show landmarks, Show signs and this guide.'),
    ('Word pills', 'Your signs so far. Amber with a ? is still being read.'),
    ('Current sign', 'Amber while reading, white once accepted. The bar shows how sure ATLAS is.'),
    ('Other guesses', 'The next most likely signs.'),
    ('English', 'The translation, written as you sign. Its dot turns green while it is read aloud. '
        'With Show signs off, only this stays on screen.'),
    ('Blue lines', 'ATLAS can see your hands.'),
    ('Reset (↺)', 'Clears the signs so far.'),
    ('Start / Finish', 'Starts translating, or finishes the sentence and stops.'),
  ];

  static const _practiceLegend = [
    ('Sign this', 'The round number and the sign to perform.'),
    ('Score', 'Signs matched so far.'),
    ('More (•••)', 'Show landmarks and this guide.'),
    ('Reference', 'A looping clip of the sign. Copy it.'),
    ('Result', 'Green “Matched · +1”, or red “Saw … · try again” when ATLAS saw a different sign.'),
    ('Try again (↺)', 'Tries the same sign again.'),
    ('Skip / Next', 'Skip moves on and counts as missed. After a match, Next moves on straight away.'),
  ];

  @override
  Widget build(BuildContext context) {
    final legend = screen == ManualScreen.live ? _liveLegend : _practiceLegend;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: AspectRatio(
            aspectRatio: 760 / 352,
            child: FittedBox(
              child: SizedBox(
                width: 760,
                height: 352,
                child: screen == ManualScreen.live ? const _LiveMock() : const _PracticeMock(),
              ),
            ),
          ),
        ),
        const SizedBox(height: 16),
        for (final (index, (title, meaning)) in legend.indexed) ...[
          if (index > 0) const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Pin(index + 1),
              const SizedBox(width: 12),
              Expanded(
                child: Text.rich(
                  TextSpan(children: [
                    TextSpan(text: '$title  ', style: Atlas.button()),
                    TextSpan(text: meaning, style: Atlas.body()),
                  ]),
                ),
              ),
            ],
          ),
        ],
      ],
    );
  }
}

class _Pin extends StatelessWidget {
  const _Pin(this.number);

  final int number;

  @override
  Widget build(BuildContext context) => Container(
    width: 24,
    height: 24,
    alignment: Alignment.center,
    decoration: BoxDecoration(
      color: Atlas.gold,
      shape: BoxShape.circle,
      border: Border.all(color: Atlas.white, width: 2),
    ),
    child: Text('$number', style: Atlas.label(Atlas.ink)),
  );
}

TextStyle _hud(double size, FontWeight weight, [Color color = Atlas.white]) =>
    TextStyle(fontFamily: Atlas.font, fontSize: size, fontWeight: weight, color: color, height: 1.2);

final _dim = Atlas.white.withValues(alpha: 0.7);

/// A glass panel over the camera.
Widget _glass({double? width, double height = 40, double radius = 20, Color? fill, Widget? child, EdgeInsets? padding}) =>
    Container(
      width: width,
      height: height,
      padding: padding,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: fill ?? Atlas.white.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(color: Atlas.white.withValues(alpha: 0.2)),
      ),
      child: child,
    );

Widget _icon(IconData icon) => _glass(width: 40, child: Icon(icon, size: 20, color: Atlas.white));

Widget _dot(Color color, [double size = 7]) =>
    Container(width: size, height: size, decoration: BoxDecoration(color: color, shape: BoxShape.circle));

Widget _pill(String text, Color dot, Color color) => _glass(
  height: 26,
  radius: 13,
  padding: const EdgeInsets.symmetric(horizontal: 10),
  child: Row(mainAxisSize: MainAxisSize.min, children: [
    _dot(dot),
    const SizedBox(width: 6),
    Text(text, style: _hud(11, FontWeight.w600, color)),
  ]),
);

Widget _confidence(double value, Color color) => Row(
  mainAxisSize: MainAxisSize.min,
  children: [
    Container(
      width: 100,
      height: 4,
      alignment: Alignment.centerLeft,
      decoration: BoxDecoration(color: Atlas.white.withValues(alpha: 0.25), borderRadius: BorderRadius.circular(2)),
      child: Container(
        width: 100 * value,
        decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2)),
      ),
    ),
    const SizedBox(width: 8),
    Text('${(value * 100).round()}%', style: _hud(11, FontWeight.w600)),
  ],
);

Widget _bigButton(String label, IconData icon) => Container(
  width: 150,
  height: 56,
  decoration: BoxDecoration(
    gradient: const LinearGradient(colors: [Atlas.sky, Atlas.blue]),
    borderRadius: BorderRadius.circular(28),
  ),
  child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
    Icon(icon, size: 20, color: Atlas.white),
    const SizedBox(width: 8),
    Text(label, style: _hud(16, FontWeight.w600)),
  ]),
);

Widget _reset({bool enabled = true}) => Opacity(
  opacity: enabled ? 1 : 0.4,
  child: _glass(width: 52, height: 52, radius: 26, child: const Icon(Icons.refresh_rounded, size: 24, color: Atlas.white)),
);

Widget _modeBlock(String mode, {required bool live}) => Column(
  crossAxisAlignment: CrossAxisAlignment.start,
  children: [
    Row(children: [
      if (live) ...[_dot(Atlas.green, 8), const SizedBox(width: 6)],
      Text(mode, style: _hud(15, FontWeight.w600)),
    ]),
    const SizedBox(height: 2),
    Text('Signing', style: _hud(10.5, FontWeight.w600)),
  ],
);

const _cameraBackground = BoxDecoration(
  gradient: LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Atlas.night, Color(0xFF1B3A5C)],
  ),
);

Widget _pinAt(int number, double left, double top) => Positioned(left: left, top: top, child: _Pin(number));

class _LiveMock extends StatelessWidget {
  const _LiveMock();

  @override
  Widget build(BuildContext context) => DecoratedBox(
    decoration: _cameraBackground,
    child: Stack(
      children: [
        const Positioned(left: 260, top: 96, width: 280, height: 256, child: CustomPaint(painter: SignerPainter())),
        Positioned(left: 24, top: 16, child: _icon(Icons.chevron_left_rounded)),
        Positioned(left: 76, top: 18, child: _modeBlock('LIVE', live: true)),
        Positioned(left: 644, top: 16, child: _icon(Icons.flip_camera_ios_outlined)),
        Positioned(left: 696, top: 16, child: _icon(Icons.more_horiz_rounded)),
        Positioned(
          left: 24,
          top: 72,
          child: Row(children: [
            _pill('HELLO', _dim, _dim),
            const SizedBox(width: 8),
            _pill('MY', Atlas.green, Atlas.white),
          ]),
        ),
        Positioned(
          right: 24,
          top: 68,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text('NAME', style: _hud(20, FontWeight.w700, Atlas.gold)),
              const SizedBox(height: 4),
              _confidence(0.72, Atlas.gold),
              const SizedBox(height: 8),
              _pill('NAME  0.72', Atlas.gold, Atlas.white),
              const SizedBox(height: 6),
              _pill('MY  0.14', Atlas.sky, _dim),
              const SizedBox(height: 6),
              _pill('HELLO  0.05', Atlas.sky, _dim),
            ],
          ),
        ),
        Positioned(
          left: 24,
          bottom: 34,
          child: Row(children: [
            _dot(Atlas.green, 14),
            const SizedBox(width: 10),
            Text('Hello, my', style: _hud(28, FontWeight.w700)),
          ]),
        ),
        Positioned(
          right: 24,
          bottom: 20,
          child: Row(children: [_reset(), const SizedBox(width: 12), _bigButton('Finish', Icons.check_rounded)]),
        ),
        _pinAt(1, 134, 30),
        _pinAt(2, 632, 4),
        _pinAt(3, 684, 4),
        _pinAt(4, 176, 73),
        _pinAt(5, 584, 66),
        _pinAt(6, 566, 150),
        _pinAt(7, 196, 270),
        _pinAt(8, 434, 108),
        _pinAt(9, 508, 266),
        _pinAt(10, 572, 264),
      ],
    ),
  );
}

class _PracticeMock extends StatelessWidget {
  const _PracticeMock();

  @override
  Widget build(BuildContext context) => DecoratedBox(
    decoration: _cameraBackground,
    child: Stack(
      children: [
        const Positioned(left: 220, top: 96, width: 280, height: 256, child: CustomPaint(painter: SignerPainter())),
        Positioned(left: 24, top: 16, child: _icon(Icons.chevron_left_rounded)),
        Positioned(left: 76, top: 18, child: _modeBlock('Practice', live: false)),
        Positioned(
          left: 236,
          top: 10,
          child: _glass(
            width: 186,
            height: 52,
            radius: 18,
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              Text('ROUND 3 OF 10  ·  SIGN THIS', style: _hud(8.5, FontWeight.w600, _dim)),
              const SizedBox(height: 2),
              Text('HELLO', style: _hud(20, FontWeight.w600)),
            ]),
          ),
        ),
        Positioned(left: 534, top: 16, child: _glass(width: 98, child: Text('Score 2 / 10', style: _hud(12, FontWeight.w600)))),
        Positioned(left: 644, top: 16, child: _icon(Icons.more_horiz_rounded)),
        Positioned(
          right: 24,
          top: 68,
          child: _glass(
            width: 150,
            height: 136,
            radius: 18,
            padding: const EdgeInsets.all(10),
            child: Column(children: [
              Expanded(
                child: Container(
                  decoration: BoxDecoration(color: Atlas.night, borderRadius: BorderRadius.circular(10)),
                  child: const SizedBox.expand(child: CustomPaint(painter: SignerPainter(tracked: false))),
                ),
              ),
              const SizedBox(height: 6),
              Text('Reference', style: _hud(10, FontWeight.w600, _dim)),
            ]),
          ),
        ),
        Positioned(
          left: 28,
          bottom: 20,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                height: 26,
                padding: const EdgeInsets.symmetric(horizontal: 12),
                alignment: Alignment.center,
                decoration: BoxDecoration(color: Atlas.green, borderRadius: BorderRadius.circular(13)),
                child: Text('Matched  ·  +1', style: _hud(11, FontWeight.w600)),
              ),
              const SizedBox(height: 8),
              Text('HELLO', style: _hud(34, FontWeight.w700)),
              const SizedBox(height: 6),
              _confidence(0.94, Atlas.green),
            ],
          ),
        ),
        Positioned(
          right: 24,
          bottom: 20,
          child: Row(children: [_reset(enabled: false), const SizedBox(width: 12), _bigButton('Next', Icons.arrow_forward_rounded)]),
        ),
        _pinAt(1, 224, 2),
        _pinAt(2, 522, 4),
        _pinAt(3, 632, 4),
        _pinAt(4, 574, 60),
        _pinAt(5, 144, 238),
        _pinAt(6, 508, 266),
        _pinAt(7, 572, 264),
      ],
    ),
  );
}

// MARK: - Drawn hands

enum HandPose { open, fist, h }

/// (x, width, extended tip y, fan) for index, middle, ring and little finger, in a 100 × 100 box
/// with the hand upright, palm towards the viewer and the thumb on the left.
const _fingers = [(35.5, 10.0, 16.0, -4.0), (46.5, 10.5, 10.0, -1.0), (57.5, 10.0, 14.0, 2.0), (68.0, 8.5, 26.0, 5.0)];

Path _capsule(Offset a, Offset b, double width) {
  final d = b - a;
  final c = math.cos(d.direction), s = math.sin(d.direction);
  final path = Path()
    ..addRRect(RRect.fromRectAndRadius(
      Rect.fromLTWH(-width / 2, -width / 2, d.distance + width, width),
      Radius.circular(width / 2),
    ));
  return path.transform(Float64List.fromList([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, a.dx, a.dy, 0, 1]));
}

bool _extended(HandPose pose, int finger) => switch (pose) {
  HandPose.open => true,
  HandPose.fist => false,
  HandPose.h => finger < 2,
};

/// Draws one hand centred on [center], [size] across. [draw] runs afterwards in the hand's
/// 100 × 100 box (for marks that move with the hand).
void paintHand(
  Canvas canvas,
  Offset center,
  double size,
  HandPose pose, {
  double angle = 0,
  bool mirror = false,
  Color fill = Atlas.white,
  Color stroke = Atlas.ink,
  double strokeWidth = 2.6,
  bool fillHand = true,
  void Function(Canvas canvas, Paint line)? draw,
}) {
  final scale = size / 100;
  canvas
    ..save()
    ..translate(center.dx, center.dy)
    ..rotate(angle)
    ..scale(mirror ? -scale : scale, scale)
    ..translate(-50, -50);
  final line = Paint()
    ..color = stroke
    ..style = PaintingStyle.stroke
    ..strokeWidth = strokeWidth / scale
    ..strokeCap = StrokeCap.round
    ..strokeJoin = StrokeJoin.round;
  final body = Paint()..color = fill;

  var shape = Path()
    ..addRRect(RRect.fromLTRBR(30, 46, 74, 90, const Radius.circular(13)));
  shape = Path.combine(
    PathOperation.union,
    shape,
    Path()
      ..addRRect(RRect.fromLTRBAndCorners(38, 80, 66, 100,
          bottomLeft: const Radius.circular(6), bottomRight: const Radius.circular(6))),
  );
  for (final (i, (x, w, tip, fan)) in _fingers.indexed) {
    final finger = _extended(pose, i)
        ? _capsule(Offset(x, 56), Offset(x + (pose == HandPose.open ? fan : 0), tip + w / 2), w)
        : _capsule(Offset(x, 50), Offset(x, 45), w);
    shape = Path.combine(PathOperation.union, shape, finger);
  }
  if (pose == HandPose.open) shape = Path.combine(PathOperation.union, shape, _capsule(const Offset(32, 76), const Offset(15, 54), 11.5));

  if (fillHand) canvas.drawPath(shape, body);
  canvas.drawPath(shape, line);

  // Folded fingers: the creases between knuckles, then the thumb across the front.
  for (var i = 0; i < 3; i++) {
    if (_extended(pose, i) || _extended(pose, i + 1)) continue;
    final x = (_fingers[i].$1 + _fingers[i + 1].$1) / 2;
    canvas.drawLine(Offset(x, 42), Offset(x, 54), line);
  }
  if (pose != HandPose.open) {
    final thumb = _capsule(const Offset(31, 68), Offset(pose == HandPose.h ? 52 : 58, pose == HandPose.h ? 64 : 62), 11.5);
    if (fillHand) canvas.drawPath(thumb, body);
    canvas.drawPath(thumb, line);
  }
  draw?.call(canvas, line);
  canvas.restore();
}

/// The person on the drawn camera screen: head, shoulders and raised hands, with the tracking lines
/// ([tracked]) drawn over the hands like the live screen does.
class SignerPainter extends CustomPainter {
  const SignerPainter({this.tracked = true});

  final bool tracked;

  @override
  void paint(Canvas canvas, Size size) {
    // Drawn on a 280 × 256 box, scaled to fit.
    final s = math.min(size.width / 280, size.height / 256);
    canvas
      ..save()
      ..translate((size.width - 280 * s) / 2, size.height - 256 * s)
      ..scale(s);
    const cx = 140.0;
    final person = Paint()..color = Atlas.white.withValues(alpha: 0.12);
    canvas.drawCircle(const Offset(cx, 70), 28, person);
    canvas.drawPath(
      Path()
        ..moveTo(cx - 92, 256)
        ..quadraticBezierTo(cx - 88, 132, cx - 44, 124)
        ..lineTo(cx + 44, 124)
        ..quadraticBezierTo(cx + 88, 132, cx + 92, 256)
        ..close(),
      person,
    );
    for (final side in [-1.0, 1.0]) {
      canvas.drawPath(_capsule(Offset(cx + side * 54, 138), Offset(cx + side * 102, 180), 22), person);
      canvas.drawPath(_capsule(Offset(cx + side * 102, 180), Offset(cx + side * 100, 100), 20), person);
      paintHand(
        canvas,
        Offset(cx + side * 100, 66),
        72,
        HandPose.open,
        mirror: side < 0,
        angle: side * 0.12,
        fill: Atlas.white.withValues(alpha: 0.16),
        stroke: tracked ? Atlas.sky.withValues(alpha: 0.95) : Atlas.white.withValues(alpha: 0.3),
        strokeWidth: 2,
      );
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(SignerPainter oldDelegate) => oldDelegate.tracked != tracked;
}

/// A control sign drawn on a 320 × 200 card, with its short caption.
class GestureIllustration extends StatelessWidget {
  const GestureIllustration({super.key, required this.art});

  final GestureArt art;

  @override
  Widget build(BuildContext context) {
    final (badge, left, right) = switch (art) {
      GestureArt.finish => ('HOLD 1 SECOND', 'LEFT HAND · FIST', 'RIGHT HAND · OPEN'),
    };
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: AspectRatio(
        aspectRatio: 320 / 200,
        child: ColoredBox(
          color: Atlas.pale,
          child: Stack(
            fit: StackFit.expand,
            children: [
              CustomPaint(painter: _GesturePainter(art)),
              Align(alignment: const Alignment(0, -0.86), child: AtlasChip(badge, fill: Atlas.white)),
              Align(alignment: const Alignment(-0.92, 0.9), child: Text(left, style: Atlas.label(Atlas.ink))),
              Align(alignment: const Alignment(0.92, 0.9), child: Text(right, style: Atlas.label(Atlas.ink))),
            ],
          ),
        ),
      ),
    );
  }
}

class _GesturePainter extends CustomPainter {
  const _GesturePainter(this.art);

  final GestureArt art;

  @override
  void paint(Canvas canvas, Size size) {
    canvas
      ..save()
      ..scale(size.width / 320, size.height / 200);
    switch (art) {
      case GestureArt.finish:
        _finish(canvas);
    }
    canvas.restore();
  }

  /// Shown as on the (mirrored) camera screen: the signer's right hand is on the right.
  void _finish(Canvas canvas) {
    final person = Paint()..color = Atlas.line;
    canvas.drawCircle(const Offset(160, 104), 24, person);
    canvas.drawPath(
      Path()
        ..moveTo(92, 200)
        ..quadraticBezierTo(96, 136, 132, 130)
        ..lineTo(188, 130)
        ..quadraticBezierTo(224, 136, 228, 200)
        ..close(),
      person,
    );
    for (final side in [-1.0, 1.0]) {
      canvas.drawPath(_capsule(Offset(160 + side * 34, 140), Offset(160 + side * 76, 160), 18), person);
      canvas.drawPath(_capsule(Offset(160 + side * 76, 160), Offset(160 + side * 86, 100), 16), person);
    }
    // The shoulder line both wrists must be above.
    final dash = Paint()
      ..color = Atlas.gold
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    for (var x = 14.0; x < 306; x += 14) {
      canvas.drawLine(Offset(x, 132), Offset(math.min(x + 7, 306), 132), dash);
    }
    paintHand(canvas, const Offset(74, 70), 58, HandPose.fist, mirror: true);
    paintHand(canvas, const Offset(246, 64), 70, HandPose.open);
  }

  @override
  bool shouldRepaint(_GesturePainter oldDelegate) => oldDelegate.art != art;
}
