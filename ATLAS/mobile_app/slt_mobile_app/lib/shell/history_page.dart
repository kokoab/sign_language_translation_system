// HISTORY and SESSION DETAIL (Figma 09 / HISTORY, 10 / SESSION DETAIL): one row per Live session
// saved on this iPhone, and a detail page with the recognized signs, the sentence and a timeline.
import 'package:flutter/material.dart';

import 'style.dart';

const _months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

String _two(int value) => value.toString().padLeft(2, '0');

String _clock(num seconds) => '${seconds ~/ 60}:${_two(seconds.toInt() % 60)}';

/// A committed gloss as people read it: fingerspelled words lose their `fs-` marker.
String displayGloss(String gloss) => gloss.startsWith('fs-') ? gloss.substring(3).toUpperCase() : gloss;

/// (when, length, said) exactly as the desktop describe() words them.
(String, String, String) describeSession(Map<String, dynamic> row) {
  final started = DateTime.tryParse('${row['started_utc'] ?? ''}')?.toLocal();
  final when = started == null
      ? '${row['stamp'] ?? 'unknown'}'
      : '${_two(started.day)} ${_months[started.month - 1]} ${started.year} · ${_two(started.hour)}:${_two(started.minute)}';
  final length = _clock(((row['duration_seconds'] as num?) ?? 0).toInt());
  final sentences = [for (final s in row['sentences'] as List? ?? const []) if ('$s'.isNotEmpty) '$s'];
  final glosses = [for (final g in row['glosses'] as List? ?? const []) '$g'];
  final said = sentences.isNotEmpty
      ? sentences.last
      : glosses.isNotEmpty
          ? glosses.join(' ')
          : 'No recognized signs';
  return (when, length, said);
}

List<String> _glosses(Map<String, dynamic> row) =>
    [for (final g in row['glosses'] as List? ?? const []) displayGloss('$g')];

class HistoryPage extends StatelessWidget {
  const HistoryPage({super.key, required this.sessions, required this.onSelect, required this.onBack});

  final List<Map<String, dynamic>> sessions;
  final ValueChanged<Map<String, dynamic>> onSelect;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) => PortraitScaffold(
    header: NavHeader(title: 'History', onBack: onBack),
    body: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 16),
        Text('Your conversations.', style: Atlas.heading()),
        const SizedBox(height: 16),
        Text('Revisit the signs from each session.', style: Atlas.body()),
        const SizedBox(height: 16),
        Expanded(
          child: sessions.isEmpty
              ? const EmptyState(title: 'No sessions yet', hint: 'Sessions appear here after you use Live.')
              : ListView.separated(
                  padding: EdgeInsets.zero,
                  itemCount: sessions.length,
                  separatorBuilder: (_, _) => const SizedBox(height: 12),
                  itemBuilder: (context, index) => _SessionRow(row: sessions[index], onTap: onSelect),
                ),
        ),
      ],
    ),
    footer: const ShieldNote('Your sessions are saved on this device.', fill: Atlas.pale),
  );
}

/// Row / Session.
class _SessionRow extends StatelessWidget {
  const _SessionRow({required this.row, required this.onTap});

  final Map<String, dynamic> row;
  final ValueChanged<Map<String, dynamic>> onTap;

  @override
  Widget build(BuildContext context) {
    final (started, length, said) = describeSession(row);
    final glosses = _glosses(row);
    final sentences = row['sentences'] as List? ?? const [];
    final preview = sentences.isNotEmpty ? said : glosses.isNotEmpty ? glosses.join(' · ') : said;
    return Material(
      color: Atlas.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        side: const BorderSide(color: Atlas.line),
      ),
      child: InkWell(
        onTap: () => onTap(row),
        borderRadius: BorderRadius.circular(20),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(child: Text(started, style: Atlas.caption(Atlas.ink))),
                  Text(length, style: Atlas.caption()),
                ],
              ),
              const SizedBox(height: 8),
              Text(preview, maxLines: 1, overflow: TextOverflow.ellipsis, style: Atlas.caption()),
              if (row['complete'] == false) ...[
                const SizedBox(height: 8),
                Text('Unfinished', style: Atlas.label()),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class SessionDetailPage extends StatelessWidget {
  const SessionDetailPage({super.key, required this.row, required this.onBack});

  final Map<String, dynamic> row;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    final started = DateTime.tryParse('${row['started_utc'] ?? ''}')?.toLocal();
    final (stamp, length, _) = describeSession(row);
    final day = started == null ? stamp : '${_two(started.day)} ${_months[started.month - 1]} ${started.year}';
    final time = started == null ? length : '${_two(started.hour)}:${_two(started.minute)} · $length';
    final glosses = _glosses(row);
    final sentences = [for (final s in row['sentences'] as List? ?? const []) if ('$s'.isNotEmpty) '$s'];
    final timeline = [
      for (final e in row['timeline'] as List? ?? const [])
        if (e is Map) ((e['seconds'] as num?) ?? 0, displayGloss('${e['gloss'] ?? ''}')),
    ];
    final unfinished = row['complete'] == false;
    return PortraitScaffold(
      header: NavHeader(title: 'Session detail', onBack: onBack),
      body: ListView(
        padding: EdgeInsets.zero,
        children: [
          const SizedBox(height: 16),
          Text(day, style: Atlas.title()),
          const SizedBox(height: 16),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [AtlasChip(time), if (unfinished) const AtlasChip.warm('Unfinished')],
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Atlas.pale, borderRadius: BorderRadius.circular(24)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('RECOGNIZED SIGNS', style: Atlas.label()),
                const SizedBox(height: 12),
                Text(glosses.isEmpty ? 'No recognized signs' : glosses.join('  '), style: Atlas.heading()),
                if (sentences.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  for (final s in sentences) Text('“$s”', style: Atlas.body(Atlas.ink)),
                ],
                if (unfinished) ...[
                  const SizedBox(height: 12),
                  Text('This session ended before Finish.', style: Atlas.caption()),
                ],
              ],
            ),
          ),
          if (timeline.isNotEmpty) ...[
            const SizedBox(height: 16),
            Text('Sign timeline', style: Atlas.subheading()),
            const SizedBox(height: 16),
            for (final (seconds, gloss) in timeline)
              Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Row(
                  children: [
                    SizedBox(width: 40, child: Text(_clock(seconds), style: Atlas.caption())),
                    const SizedBox(width: 12),
                    Expanded(child: Text(gloss, style: Atlas.button())),
                  ],
                ),
              ),
          ],
        ],
      ),
      footer: AtlasButton(label: 'Back to history', style: AtlasButtonStyle.secondary, onPressed: onBack),
    );
  }
}
