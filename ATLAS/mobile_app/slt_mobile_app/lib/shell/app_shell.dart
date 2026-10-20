// The ATLAS mobile app shell (Figma "02 · Mobile Screens"). Portrait pages drawn here:
//   SPLASH → HOME → GLOSSES (→ gloss detail), PRACTICE (setup / result), HISTORY (→ session detail),
//   MANUAL (→ section → gloss clip)
// Live and the practice round are the native portrait/landscape camera screen (LiveReelViewController);
// it returns {"navigate": page, "section": manual section, practice results...} when it closes.
// The Reel models warm at launch; "Start translating" and practice wait until they are ready.
import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/slt_inference_service.dart';
import 'glosses_page.dart';
import 'history_page.dart';
import 'home_page.dart';
import 'manual_page.dart';
import 'practice_page.dart';
import 'style.dart';

const shellPages = ['HOME', 'GLOSSES', 'PRACTICE', 'HISTORY', 'MANUAL'];

class AppShell extends StatefulWidget {
  const AppShell({super.key, this.splash = const Duration(milliseconds: 1400)});

  /// How long the splash stays up at launch.
  final Duration splash;

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  final _native = SltInferenceService();
  final _random = math.Random();
  String _page = 'SPLASH';
  GlossExample? _selected;
  Map<String, dynamic>? _session;
  ManualSection? _manualSection;
  PracticeResult? _practiceResult;
  List<String> _lastRound = const [];
  List<GlossExample> _catalog = const [];
  List<Map<String, dynamic>> _sessions = const [];
  bool _ready = false;
  String _status = 'Warming up…';
  bool _nativeOpen = false;
  Timer? _poll;
  Timer? _splash;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations(const [DeviceOrientation.portraitUp]);
    loadCatalog().then((value) => mounted ? setState(() => _catalog = value) : null);
    _refreshSessions();
    _poll = Timer.periodic(const Duration(milliseconds: 400), (_) => _checkReady());
    _checkReady();
    _splash = Timer(widget.splash, () => _page == 'SPLASH' ? _go('HOME') : null);
  }

  @override
  void dispose() {
    _poll?.cancel();
    _splash?.cancel();
    super.dispose();
  }

  Future<void> _checkReady() async {
    try {
      final status = await _native.liveReelStatus();
      if (!mounted) return;
      final ready = status['ready'] == true;
      setState(() {
        _ready = ready;
        _status = ready
            ? ''
            : status['state'] == 'failed'
                ? "Couldn't load the sign models. Close ATLAS and open it again."
                : 'Warming up…';
      });
      if (ready || status['state'] == 'failed') _poll?.cancel();
    } catch (_) {
      _poll?.cancel();
      if (mounted) setState(() => _status = 'Live translation is not available on this device.');
    }
  }

  Future<void> _refreshSessions() async {
    try {
      final rows = await _native.listSessions();
      if (mounted) setState(() => _sessions = rows);
    } catch (_) {}
  }

  /// [section] opens MANUAL at that section instead of its hub.
  void _go(String page, {String? section}) {
    if (_nativeOpen) return;
    if (page == 'LIVE') {
      _openLive();
      return;
    }
    setState(() {
      _page = page;
      _selected = null;
      _session = null;
      _manualSection = page == 'MANUAL' ? manualSection(section) : null;
      if (page != 'PRACTICE') _practiceResult = null;
    });
    if (page == 'HISTORY') _refreshSessions();
  }

  /// Where the native screen asked to go; anything unknown lands on Home.
  void _arrive(Object? navigate, Object? section) => _go(
    navigate is String && shellPages.contains(navigate) ? navigate : 'HOME',
    section: section is String ? section : null,
  );

  /// A missed Practice sign: its clip in Glosses (back returns to the Glosses list).
  void _openGloss(String label) {
    final item = _catalog.where((g) => g.label == label).firstOrNull;
    if (item == null) return;
    setState(() {
      _page = 'GLOSSES';
      _practiceResult = null;
      _selected = item;
    });
  }

  Future<void> _speak(String text) async {
    try {
      await _native.speak(text);
    } catch (_) {}
  }

  Future<void> _deleteSession(Map<String, dynamic> row) async {
    try {
      await _native.deleteSession('${row['stamp'] ?? ''}');
    } catch (_) {}
    if (mounted) setState(() => _session = null);
    await _refreshSessions();
  }

  Future<void> _clearSessions() async {
    try {
      await _native.clearSessions();
    } catch (_) {}
    await _refreshSessions();
  }

  /// The manual's "Watch NAME in Glosses": the gloss clip, opened over the manual.
  void _watchGloss(String label) {
    final item = _catalog.where((g) => g.label == label).firstOrNull;
    if (item != null) setState(() => _selected = item);
  }

  Future<void> _openLive() async {
    if (!_ready) return;
    _nativeOpen = true;
    Map<String, Object?> result = const {};
    try {
      result = await _native.openLiveReel();
    } finally {
      _nativeOpen = false;
    }
    await _refreshSessions();
    _arrive(result['navigate'], result['section']);
  }

  Future<void> _startRound(List<String> glosses) async {
    if (!_ready || glosses.isEmpty) return;
    _lastRound = glosses;
    setState(() {
      _page = 'PRACTICE';
      _selected = null;
      _practiceResult = null;
    });
    _nativeOpen = true;
    Map<String, Object?> result = const {};
    try {
      result = await _native.openPractice(glosses);
    } finally {
      _nativeOpen = false;
    }
    final navigate = result['navigate'];
    if (navigate == 'PRACTICE') {
      // A finished round shows its score; leaving one early returns to the setup.
      setState(() => _practiceResult = result['completed'] == true
          ? PracticeResult(
              score: (result['score'] as int?) ?? 0,
              total: (result['total'] as int?) ?? glosses.length,
              missed: [for (final m in result['missed'] as List? ?? const []) '$m'],
            )
          : null);
    } else {
      _arrive(navigate, result['section']);
    }
  }

  void _startRandom(int size) {
    final pool = [for (final item in _catalog) item.label]..shuffle(_random);
    _startRound(pool.take(size).toList());
  }

  void _showSettings() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: Atlas.white,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(32))),
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 24, 24, 16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Settings', style: Atlas.heading()),
              const SizedBox(height: 16),
              _SettingRow('Recognition models', _ready ? 'Ready' : _status),
              _SettingRow('Vocabulary', '${_catalog.isEmpty ? 100 : _catalog.length} signs'),
              _SettingRow('Saved sessions', '${_sessions.length}'),
              if (Platform.isAndroid) _HandImageSetting(native: _native),
              if (Platform.isIOS || Platform.isAndroid) _LiveSettings(native: _native),
              const SizedBox(height: 16),
              const ShieldNote('All processing stays on your device.'),
            ],
          ),
        ),
      ),
    );
  }

  Widget _content() {
    switch (_page) {
      case 'SPLASH':
        return const SplashPage();
      case 'GLOSSES':
        final selected = _selected;
        if (selected != null) {
          return GlossDetailPage(
            key: ValueKey(selected.label),
            item: selected,
            onBack: () => setState(() => _selected = null),
            onPractice: () => _startRound([selected.label]),
          );
        }
        return GlossesPage(
          catalog: _catalog,
          onSelect: (item) => setState(() => _selected = item),
          onBack: () => _go('HOME'),
        );
      case 'MANUAL':
        final selected = _selected, section = _manualSection;
        if (selected != null) {
          return GlossDetailPage(
            key: ValueKey(selected.label),
            item: selected,
            onBack: () => setState(() => _selected = null),
            onPractice: () => _startRound([selected.label]),
          );
        }
        if (section != null) {
          return ManualSectionPage(
            section: section,
            onOpen: (id) => setState(() => _manualSection = manualSection(id)),
            onWatch: _watchGloss,
            onBack: () => setState(() => _manualSection = null),
          );
        }
        return ManualPage(
          onSelect: (id) => setState(() => _manualSection = manualSection(id)),
          onBack: () => _go('HOME'),
        );
      case 'HISTORY':
        final session = _session;
        if (session != null) {
          return SessionDetailPage(
            row: session,
            onBack: () => setState(() => _session = null),
            onSpeak: (Platform.isIOS || Platform.isAndroid) ? _speak : null,
            onDelete: (Platform.isIOS || Platform.isAndroid) ? () => _deleteSession(session) : null,
          );
        }
        return HistoryPage(
          sessions: _sessions,
          onSelect: (row) => setState(() => _session = row),
          onBack: () => _go('HOME'),
          onClearAll: (Platform.isIOS || Platform.isAndroid) ? _clearSessions : null,
        );
      case 'PRACTICE':
        final result = _practiceResult;
        if (result != null) {
          return PracticeResultPage(
            result: result,
            onAgain: () => _lastRound.length > 1 ? _startRandom(_lastRound.length) : _startRound(_lastRound),
            onDone: () => _go('HOME'),
            onWatch: _openGloss,
          );
        }
        return PracticeSetupPage(
          available: _catalog.length,
          ready: _ready,
          onStart: _startRandom,
          onBack: () => _go('HOME'),
        );
      default:
        return HomePage(
          ready: _ready,
          status: _status,
          sessionCount: _sessions.length,
          glossCount: _catalog.length,
          onAction: _go,
          onSettings: _showSettings,
        );
    }
  }

  @override
  Widget build(BuildContext context) => PopScope(
    canPop: _page == 'HOME',
    onPopInvokedWithResult: (didPop, _) {
      if (didPop) return;
      if (_selected != null) {
        setState(() => _selected = null);
      } else if (_session != null) {
        setState(() => _session = null);
      } else if (_manualSection != null) {
        setState(() => _manualSection = null);
      } else if (_page != 'SPLASH') {
        _go('HOME');
      }
    },
    child: Scaffold(
      backgroundColor: Atlas.white,
      body: AnimatedSwitcher(
        duration: const Duration(milliseconds: 180),
        child: KeyedSubtree(
          key: ValueKey('$_page/${_selected?.label}/${_session?['stamp']}/${_manualSection?.id}/${_practiceResult != null}'),
          child: _content(),
        ),
      ),
    ),
  );
}

class _SettingRow extends StatelessWidget {
  const _SettingRow(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(bottom: 12),
    child: Row(
      children: [
        Expanded(child: Text(label, style: Atlas.body(Atlas.ink))),
        Flexible(child: Text(value, style: Atlas.body(), textAlign: TextAlign.right)),
      ],
    ),
  );
}

/// Android: recognition with hand images (MobileCLIP crops; faster phones) or landmarks only.
/// Automatic follows a one-time speed test on this phone.
class _HandImageSetting extends StatefulWidget {
  const _HandImageSetting({required this.native});

  final SltInferenceService native;

  @override
  State<_HandImageSetting> createState() => _HandImageSettingState();
}

class _HandImageSettingState extends State<_HandImageSetting> {
  String _mode = 'AUTO';
  bool? _active;
  double? _cropMs;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final value = await widget.native.getHandImageMode();
      if (!mounted) return;
      setState(() {
        _mode = value['mode'] as String? ?? 'AUTO';
        _active = value['active'] as bool?;
        _cropMs = (value['crop_encode_ms'] as num?)?.toDouble();
      });
    } catch (_) {}
  }

  Future<void> _set(String mode) async {
    setState(() {
      _mode = mode;
      _busy = true;
    });
    try {
      final value = await widget.native.setHandImageMode(mode);
      if (mounted) setState(() => _active = value['active'] as bool?);
    } catch (_) {}
    await _load();
    if (mounted) setState(() => _busy = false);
  }

  @override
  Widget build(BuildContext context) {
    final state = _busy ? 'Loading…' : _active == null ? 'Loading…' : _active! ? 'Hand images' : 'Landmarks only';
    final speed = _cropMs == null ? '' : ' · ${_cropMs!.round()} ms per hand crop on this phone';
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _SettingRow('Recognition mode', state),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'AUTO', label: Text('Automatic')),
              ButtonSegment(value: 'ON', label: Text('Hand images')),
              ButtonSegment(value: 'OFF', label: Text('Landmarks')),
            ],
            selected: {_mode},
            showSelectedIcon: false,
            onSelectionChanged: _busy ? null : (value) => _set(value.first),
          ),
          const SizedBox(height: 6),
          Text('Hand images are more accurate but need a fast phone$speed.', style: Atlas.caption()),
        ],
      ),
    );
  }
}

/// iPhone: Live switches (both default off) and the first-Live tips.
class _LiveSettings extends StatefulWidget {
  const _LiveSettings({required this.native});

  final SltInferenceService native;

  @override
  State<_LiveSettings> createState() => _LiveSettingsState();
}

class _LiveSettingsState extends State<_LiveSettings> {
  bool _nameSpelling = false;
  bool _saveVideos = false;
  bool _tipsReset = false;

  @override
  void initState() {
    super.initState();
    widget.native.getLiveSettings().then((value) {
      if (!mounted) return;
      setState(() {
        _nameSpelling = value['nameSpelling'] == true;
        _saveVideos = value['saveVideos'] == true;
      });
    }).catchError((_) {});
  }

  Future<void> _set(String key, bool value) async {
    setState(() => key == 'nameSpelling' ? _nameSpelling = value : _saveVideos = value);
    try {
      await widget.native.setLiveSetting(key, value);
    } catch (_) {}
  }

  Widget _switch(String title, String detail, bool value, ValueChanged<bool> onChanged) => Padding(
    padding: const EdgeInsets.only(bottom: 12),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Atlas.body(Atlas.ink)),
              const SizedBox(height: 2),
              Text(detail, style: Atlas.caption()),
            ],
          ),
        ),
        const SizedBox(width: 12),
        Switch.adaptive(value: value, onChanged: onChanged),
      ],
    ),
  );

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      if (Platform.isIOS) _switch('Spell a name after NAME', 'After you sign NAME in Live, ATLAS reads letters for a few seconds.',
          _nameSpelling, (value) => _set('nameSpelling', value)),
      _switch('Save Live videos', Platform.isIOS ? 'Keep a video of each Live session in the Files app.' : 'Save a video of each Live take on this device.', _saveVideos,
          (value) => _set('saveVideos', value)),
      Align(
        alignment: Alignment.centerLeft,
        child: TextButton(
          onPressed: _tipsReset
              ? null
              : () async {
                  try {
                    await widget.native.resetLiveTips();
                  } catch (_) {}
                  if (mounted) setState(() => _tipsReset = true);
                },
          style: TextButton.styleFrom(padding: EdgeInsets.zero, minimumSize: const Size(44, 44)),
          child: Text(_tipsReset ? 'Live tips will show next time' : 'Show Live tips again', style: Atlas.button(Atlas.blue)),
        ),
      ),
    ],
  );
}
