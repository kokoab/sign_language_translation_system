// GLOSSES and GLOSS DETAIL (Figma 07 / GLOSSES, 08 / GLOSS DETAIL): a searchable two-column
// library of the 100 signs, and a detail page with the looping reference clip, playback speed
// and "Practice signs".
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:video_player/video_player.dart';

import 'style.dart';

const glossAssets = 'assets/gloss_examples';

class GlossExample {
  GlossExample(this.label, Map<String, dynamic> row)
      : category = (row['category'] as String? ?? '').replaceAll('_', ' '),
        seconds = (row['seconds'] as num?)?.toDouble() ?? 0,
        notes = [for (final note in row['notes'] as List? ?? const []) '$note'],
        poster = (row['assets'] as Map?)?['poster'] as String?,
        loop = (row['assets'] as Map?)?['loop'] as String?;

  final String label;
  final String category;
  final double seconds;
  final List<String> notes;
  final String? poster;
  final String? loop;
}

Future<List<GlossExample>> loadCatalog() async {
  try {
    final data = jsonDecode(await rootBundle.loadString('$glossAssets/catalog.json')) as Map<String, dynamic>;
    final examples = data['examples'] as Map<String, dynamic>;
    final labels = examples.keys.toList()..sort();
    return [for (final label in labels) GlossExample(label, examples[label] as Map<String, dynamic>)];
  } catch (_) {
    return const [];
  }
}

class GlossesPage extends StatefulWidget {
  const GlossesPage({super.key, required this.catalog, required this.onSelect, required this.onBack});

  final List<GlossExample> catalog;
  final ValueChanged<GlossExample> onSelect;
  final VoidCallback onBack;

  @override
  State<GlossesPage> createState() => _GlossesPageState();
}

class _GlossesPageState extends State<GlossesPage> {
  String _query = '';

  @override
  Widget build(BuildContext context) {
    final query = _query.trim().toUpperCase();
    final shown = query.isEmpty ? widget.catalog : [for (final g in widget.catalog) if (g.label.contains(query)) g];
    return PortraitScaffold(
      header: NavHeader(title: 'Glosses', onBack: widget.onBack),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const SizedBox(height: 16),
          Text('${widget.catalog.length} signs to explore.', style: Atlas.heading()),
          const SizedBox(height: 16),
          TextField(
            onChanged: (value) => setState(() => _query = value),
            textCapitalization: TextCapitalization.characters,
            style: Atlas.caption(Atlas.ink),
            decoration: InputDecoration(
              isDense: true,
              filled: true,
              fillColor: Atlas.canvas,
              hintText: 'Search signs',
              hintStyle: Atlas.caption(),
              prefixIcon: const Icon(Icons.search_rounded, size: 20, color: Atlas.muted),
              prefixIconConstraints: const BoxConstraints(minWidth: 44, minHeight: 44),
              contentPadding: const EdgeInsets.symmetric(vertical: 12),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: widget.catalog.isEmpty
                ? const EmptyState(title: 'No examples built yet', hint: 'Run scripts/build_gloss_examples_v17.py')
                : shown.isEmpty
                    ? EmptyState(title: 'No match', hint: 'No sign in the vocabulary contains "$query".')
                    : GridView.builder(
                        padding: EdgeInsets.zero,
                        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          mainAxisSpacing: 12,
                          crossAxisSpacing: 16,
                          childAspectRatio: 163 / 124,
                        ),
                        itemCount: shown.length,
                        itemBuilder: (context, index) => _GlossCard(item: shown[index], onTap: widget.onSelect),
                      ),
          ),
        ],
      ),
    );
  }
}

/// Card / Gloss: the poster photo with the gloss under it.
class _GlossCard extends StatelessWidget {
  const _GlossCard({required this.item, required this.onTap});

  final GlossExample item;
  final ValueChanged<GlossExample> onTap;

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: () => onTap(item),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: item.poster == null
                ? const ColoredBox(color: Atlas.pale)
                : Image.asset('$glossAssets/${item.poster}', fit: BoxFit.cover, cacheWidth: 360),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          item.label,
          textAlign: TextAlign.center,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: Atlas.label(Atlas.ink),
        ),
      ],
    ),
  );
}

class GlossDetailPage extends StatefulWidget {
  const GlossDetailPage({super.key, required this.item, required this.onPractice, required this.onBack});

  final GlossExample item;
  final VoidCallback onPractice;
  final VoidCallback onBack;

  @override
  State<GlossDetailPage> createState() => _GlossDetailPageState();
}

class _GlossDetailPageState extends State<GlossDetailPage> {
  static const _speeds = [0.5, 1.0, 1.5, 2.0];
  VideoPlayerController? _video;
  double _speed = 1.0;

  @override
  void initState() {
    super.initState();
    final loop = widget.item.loop;
    if (loop != null) {
      final video = VideoPlayerController.asset('$glossAssets/$loop');
      _video = video;
      video.setLooping(true);
      video.setVolume(0);
      video.addListener(_changed);
      video.initialize().then((_) {
        if (mounted) setState(video.play);
      });
    }
  }

  void _changed() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _video?.removeListener(_changed);
    _video?.dispose();
    super.dispose();
  }

  void _togglePlay() {
    final video = _video;
    if (video == null || !video.value.isInitialized) return;
    video.value.isPlaying ? video.pause() : video.play();
  }

  void _setSpeed(double speed) {
    setState(() => _speed = speed);
    _video?.setPlaybackSpeed(speed);
  }

  static String _speedLabel(double s) => '${s == s.roundToDouble() ? s.toInt() : s}×';

  @override
  Widget build(BuildContext context) {
    final item = widget.item;
    final video = _video;
    final playing = video != null && video.value.isInitialized && video.value.isPlaying;
    return PortraitScaffold(
      header: NavHeader(title: 'Gloss detail', onBack: widget.onBack),
      body: ListView(
        padding: EdgeInsets.zero,
        children: [
          const SizedBox(height: 16),
          GestureDetector(
            onTap: _togglePlay,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(24),
              child: AspectRatio(
                aspectRatio: 342 / 290,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    const ColoredBox(color: Atlas.pale),
                    if (video != null && video.value.isInitialized)
                      FittedBox(
                        fit: BoxFit.cover,
                        clipBehavior: Clip.hardEdge,
                        child: SizedBox(
                          width: video.value.size.width,
                          height: video.value.size.height,
                          child: VideoPlayer(video),
                        ),
                      )
                    else if (item.poster != null)
                      Image.asset('$glossAssets/${item.poster}', fit: BoxFit.cover)
                    else
                      Center(child: Text('No example clip', style: Atlas.caption())),
                    if (video != null && !playing)
                      Center(
                        child: Container(
                          width: 56,
                          height: 56,
                          decoration: const BoxDecoration(color: Atlas.white, shape: BoxShape.circle),
                          child: const Icon(Icons.play_arrow_rounded, size: 28, color: Atlas.blue),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Text(item.label, style: Atlas.title()),
          const SizedBox(height: 16),
          Text('Watch the example, then give it a try.', style: Atlas.body()),
          if (item.category.isNotEmpty || item.seconds > 0) ...[
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                if (item.category.isNotEmpty) AtlasChip(item.category),
                if (item.seconds > 0) AtlasChip('Clip ${item.seconds.toStringAsFixed(1)} s'),
              ],
            ),
          ],
          for (final note in item.notes)
            Padding(padding: const EdgeInsets.only(top: 8), child: Text(note, style: Atlas.caption())),
          if (video != null) ...[
            const SizedBox(height: 16),
            Text('PLAYBACK SPEED', style: Atlas.label()),
            const SizedBox(height: 12),
            Row(
              children: [
                for (final (index, speed) in _speeds.indexed) ...[
                  if (index > 0) const SizedBox(width: 8),
                  Expanded(
                    child: AtlasButton(
                      label: _speedLabel(speed),
                      height: 44,
                      style: speed == _speed ? AtlasButtonStyle.primary : AtlasButtonStyle.secondary,
                      onPressed: () => _setSpeed(speed),
                    ),
                  ),
                ],
              ],
            ),
          ],
        ],
      ),
      footer: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          AtlasButton(label: 'Practice signs', onPressed: widget.onPractice),
          const SizedBox(height: 16),
          Text('Part of the 100-sign vocabulary', style: Atlas.caption(), textAlign: TextAlign.center),
        ],
      ),
    );
  }
}
