// PRACTICE setup and result (Figma 03 / PRACTICE SETUP), portrait. The round itself is the
// native landscape camera screen; it returns the score here.
import 'package:flutter/material.dart';

import 'style.dart';

const practiceSizes = [5, 10, 20];
const _practiceBlurbs = ['A quick session', 'A balanced session', 'A longer challenge'];

class PracticeResult {
  const PracticeResult({required this.score, required this.total, required this.missed});

  final int score;
  final int total;
  final List<String> missed;
}

class PracticeSetupPage extends StatefulWidget {
  const PracticeSetupPage({
    super.key,
    required this.available,
    required this.ready,
    required this.onStart,
    required this.onBack,
  });

  final int available;
  final bool ready;
  final ValueChanged<int> onStart;
  final VoidCallback onBack;

  @override
  State<PracticeSetupPage> createState() => _PracticeSetupPageState();
}

class _PracticeSetupPageState extends State<PracticeSetupPage> {
  int _choice = 0;

  @override
  Widget build(BuildContext context) {
    final available = widget.available;
    return _PortraitPage(
      title: 'Practice',
      onBack: widget.onBack,
      footer: [
        AtlasButton(
          label: widget.ready ? 'Start practice' : 'Warming up…',
          onPressed: widget.ready && available > 0
              ? () => widget.onStart(practiceSizes[_choice].clamp(1, available))
              : null,
        ),
        const SizedBox(height: 16),
        Text('Rotate your phone when practice begins.', style: Atlas.caption(), textAlign: TextAlign.center),
      ],
      children: [
        Align(alignment: Alignment.centerLeft, child: AtlasChip('$available signs available')),
        const SizedBox(height: 16),
        Text('Choose your\npractice set.', style: Atlas.title()),
        const SizedBox(height: 12),
        Text('Try randomly selected signs from\nthe supported set, at your own pace.', style: Atlas.body()),
        const SizedBox(height: 16),
        if (available == 0)
          const EmptyState(title: 'No examples built yet', hint: 'Run scripts/build_gloss_examples_v17.py')
        else
          for (final (index, size) in practiceSizes.indexed) ...[
            if (index > 0) const SizedBox(height: 12),
            _ChoiceCard(
              title: '${size.clamp(1, available)} signs',
              description: _practiceBlurbs[index],
              selected: index == _choice,
              onTap: () => setState(() => _choice = index),
            ),
          ],
      ],
    );
  }
}

/// Choice / Practice set.
class _ChoiceCard extends StatelessWidget {
  const _ChoiceCard({required this.title, required this.description, required this.selected, required this.onTap});

  final String title;
  final String description;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Semantics(
    selected: selected,
    button: true,
    child: Material(
      color: selected ? Atlas.pale : Atlas.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        side: BorderSide(color: selected ? Atlas.blue : Atlas.line, width: selected ? 2 : 1),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: SizedBox(
          height: 88,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(title, style: Atlas.button()),
                      const SizedBox(height: 4),
                      Text(description, style: Atlas.caption()),
                    ],
                  ),
                ),
                const SizedBox(width: 16),
                Container(
                  width: 24,
                  height: 24,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: selected ? Atlas.blue : Atlas.white,
                    border: selected ? null : Border.all(color: Atlas.line, width: 2),
                  ),
                  child: selected ? const Icon(Icons.check_rounded, size: 16, color: Atlas.white) : null,
                ),
              ],
            ),
          ),
        ),
      ),
    ),
  );
}

class PracticeResultPage extends StatelessWidget {
  const PracticeResultPage({super.key, required this.result, required this.onAgain, required this.onDone});

  final PracticeResult result;
  final VoidCallback onAgain;
  final VoidCallback onDone;

  @override
  Widget build(BuildContext context) {
    final score = result.score, total = result.total;
    final remark = total > 0 && score == total
        ? 'Perfect round.'
        : total > 0 && score >= total * 0.6
            ? 'Nice work.'
            : 'Keep going.';
    return _PortraitPage(
      title: 'Practice',
      onBack: onDone,
      footer: [
        AtlasButton(label: 'Practice again', onPressed: onAgain),
        const SizedBox(height: 12),
        AtlasButton(label: 'Done', style: AtlasButtonStyle.secondary, onPressed: onDone),
        const SizedBox(height: 16),
        Text('A miss may be the model, not you. Scores are uncalibrated.',
            style: Atlas.caption(), textAlign: TextAlign.center),
      ],
      children: [
        const Align(alignment: Alignment.centerLeft, child: AtlasChip('Round complete')),
        const SizedBox(height: 16),
        Text('$score of $total', style: Atlas.display(score == total ? Atlas.green : Atlas.ink)),
        const SizedBox(height: 8),
        Text(remark, style: Atlas.heading()),
        if (result.missed.isNotEmpty) ...[
          const SizedBox(height: 24),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Atlas.cream, borderRadius: BorderRadius.circular(24)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('NOT RECOGNISED', style: Atlas.label()),
                const SizedBox(height: 12),
                Text(result.missed.take(8).join('  ·  '), style: Atlas.subheading()),
              ],
            ),
          ),
        ],
      ],
    );
  }
}

/// Header, scrolling body, and a footer pinned to the bottom: the layout of every portrait page.
class _PortraitPage extends StatelessWidget {
  const _PortraitPage({required this.title, required this.onBack, required this.children, required this.footer});

  final String title;
  final VoidCallback onBack;
  final List<Widget> children;
  final List<Widget> footer;

  @override
  Widget build(BuildContext context) => PortraitScaffold(
    header: NavHeader(title: title, onBack: onBack),
    body: ListView(padding: EdgeInsets.zero, children: [const SizedBox(height: 16), ...children]),
    footer: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: footer),
  );
}
