import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/slt_inference_service.dart';
import 'messaging_screen.dart';

class TranslationScreen extends StatefulWidget {
  const TranslationScreen({super.key, this.cameras = const []});

  final List<CameraDescription> cameras;

  @override
  State<TranslationScreen> createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen>
    with WidgetsBindingObserver {
  final _inference = SltInferenceService();
  CameraController? _camera;
  SltInferenceResult? _result;
  bool _benchmarkEnabled = false;
  bool _isRecording = false;
  bool _isProcessing = false;
  bool _liveReelOpen = false;
  String _status = 'Preparing camera…';
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _load();
  }

  Future<void> _load() async {
    if (!Platform.isIOS) {
      setState(() => _status = 'iPhone inference is available on iOS only.');
      return;
    }
    try {
      _benchmarkEnabled = await _inference.getBenchmarkEnabled();
      await _initializeCamera();
    } catch (error) {
      if (mounted) {
        setState(() {
          _error = '$error';
          _status = 'Camera unavailable.';
        });
      }
    }
  }

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) {
      if (mounted) setState(() => _status = 'No camera was found.');
      return;
    }
    final selected = widget.cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );
    final controller = CameraController(
      selected,
      ResolutionPreset.high,
      enableAudio: false,
    );
    _camera = controller;
    await controller.initialize();
    if (mounted) setState(() => _status = 'Ready to record.');
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final camera = _camera;
    if (_liveReelOpen || camera == null || !camera.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      camera.dispose();
      _camera = null;
    } else if (state == AppLifecycleState.resumed && !_isProcessing) {
      _initializeCamera();
    }
  }

  Future<void> _toggleRecording() async {
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized || _isProcessing) return;
    try {
      if (!_isRecording) {
        await camera.startVideoRecording();
        setState(() {
          _isRecording = true;
          _result = null;
          _error = null;
          _status = 'Recording… sign naturally, then tap Stop.';
        });
        return;
      }
      final video = await camera.stopVideoRecording();
      setState(() {
        _isRecording = false;
        _isProcessing = true;
        _status = 'Reading movement, hands, face, and context…';
      });
      final result = await _inference.inferVideo(
        path: video.path,
        mirrored: camera.description.lensDirection == CameraLensDirection.front,
        benchmarkEnabled: _benchmarkEnabled,
      );
      if (!mounted) return;
      setState(() {
        _result = result;
        _isProcessing = false;
        _status = result.glosses.isEmpty
            ? 'No supported sign was recognized.'
            : 'Translation complete.';
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _isRecording = false;
        _isProcessing = false;
        _error = '$error';
        _status = 'Could not translate this recording.';
      });
    }
  }

  Future<void> _openLiveReel() async {
    if (_isRecording || _isProcessing || _liveReelOpen) return;
    _liveReelOpen = true;
    final camera = _camera;
    _camera = null;
    setState(() => _status = 'Live Reel is open.');
    // The native screen needs the front camera to itself.
    await camera?.dispose();
    try {
      await _inference.openLiveReel();
    } catch (error) {
      if (mounted) setState(() => _error = '$error');
    } finally {
      _liveReelOpen = false;
      if (mounted) await _initializeCamera();
    }
  }

  Future<void> _openSettings() async {
    var enabled = _benchmarkEnabled;
    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (context) => StatefulBuilder(
        builder: (context, setSheetState) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Settings',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 12),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Device benchmark'),
                  subtitle: const Text(
                    'Repeat inference and show timing, memory, thermal state, '
                    'and JSON export. Keep off for normal translation.',
                  ),
                  value: enabled,
                  onChanged: (value) async {
                    await _inference.setBenchmarkEnabled(value);
                    if (!mounted) return;
                    setState(() => _benchmarkEnabled = value);
                    setSheetState(() => enabled = value);
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _camera?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ASL Translator'),
        actions: [
          TextButton.icon(
            onPressed: Platform.isIOS ? _openLiveReel : null,
            icon: const Icon(Icons.bolt),
            label: const Text('Live Reel'),
          ),
          IconButton(
            onPressed: Platform.isIOS ? _openSettings : null,
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: SafeArea(
        child: OrientationBuilder(
          builder: (context, orientation) =>
              orientation == Orientation.landscape
              ? Row(
                  children: [
                    Expanded(flex: 3, child: _cameraPanel()),
                    SizedBox(width: 360, child: _controlsPanel()),
                  ],
                )
              : Column(
                  children: [
                    Expanded(child: _cameraPanel()),
                    Expanded(child: _controlsPanel()),
                  ],
                ),
        ),
      ),
    );
  }

  Widget _cameraPanel() {
    final camera = _camera;
    return Container(
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey.shade900,
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.16),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (camera != null && camera.value.isInitialized)
            _CoverCameraPreview(camera: camera)
          else
            const Center(
              child: Icon(
                Icons.videocam_off_outlined,
                color: Colors.white54,
                size: 56,
              ),
            ),
          const IgnorePointer(child: _ViewfinderOverlay()),
          if (_isProcessing)
            ColoredBox(
              color: Colors.black54,
              child: const Center(child: CircularProgressIndicator()),
            ),
          Positioned(
            top: 16,
            left: 16,
            child: Chip(
              backgroundColor: _isRecording ? Colors.red : Colors.black54,
              label: Text(
                _isRecording ? 'RECORDING' : '1–10 seconds',
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _controlsPanel() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(24, 4, 24, 20),
      child: Column(
        children: [
          Text(
            _status,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.titleMedium,
          ),
          if (_error != null)
            Text(
              _error!,
              textAlign: TextAlign.center,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton.filled(
                onPressed:
                    _isProcessing ||
                        _camera == null ||
                        !(_camera?.value.isInitialized ?? false)
                    ? null
                    : _toggleRecording,
                tooltip: _isRecording ? 'Stop recording' : 'Record sign',
                icon: Icon(
                  _isRecording ? Icons.stop : Icons.fiber_manual_record,
                ),
                style: IconButton.styleFrom(
                  backgroundColor: _isRecording ? Colors.red : Colors.indigo,
                  foregroundColor: Colors.white,
                  minimumSize: const Size.square(56),
                ),
              ),
              Expanded(
                child: Center(
                  child: _result != null || _error != null
                      ? TextButton.icon(
                          onPressed: () => setState(() {
                            _result = null;
                            _error = null;
                            _status = 'Ready to record.';
                          }),
                          icon: const Icon(Icons.refresh),
                          label: const Text('Try again'),
                        )
                      : const Text(
                          'Record · Stop · Translate',
                          textAlign: TextAlign.center,
                        ),
                ),
              ),
              IconButton.filled(
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (context) => const MessagingScreen(),
                  ),
                ),
                tooltip: 'Conversation',
                icon: const Icon(Icons.chat_bubble_outline),
              ),
            ],
          ),
          if (_result != null) ...[
            const SizedBox(height: 12),
            _resultCard(_result!),
          ],
        ],
      ),
    );
  }

  Widget _resultCard(SltInferenceResult result) {
    final benchmark = result.benchmark;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              result.glosses.isEmpty ? '—' : result.glosses.join(' · '),
              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                color: Colors.indigo,
                letterSpacing: 0.7,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              result.english.isEmpty ? 'No translation' : result.english,
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            if (result.safeFallbackUsed && result.literalEnglish.isNotEmpty)
              Text(
                'Literal fallback: ${result.literalEnglish}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            if (result.diagnosticPath != null) ...[
              const SizedBox(height: 8),
              const Text(
                'Recording and landmarks saved in Files › On My iPhone › '
                'ASL Translator › Diagnostics.',
              ),
            ],
            if (benchmark != null) ...[
              const Divider(height: 24),
              Text(
                'Benchmark · ${benchmark['executionEnvironment']}',
                style: Theme.of(context).textTheme.titleSmall,
              ),
              Text(
                'Extraction ${_ms(benchmark['extractionMilliseconds'])} · '
                'RGB ${_ms(benchmark['cropEncodingMilliseconds'])}\n'
                'Median ${_ms(benchmark['medianInferenceMilliseconds'])} · '
                'p90 ${_ms(benchmark['p90InferenceMilliseconds'])}\n'
                'Memory ${_mb(benchmark['residentMemoryBeforeBytes'])} → '
                '${_mb(benchmark['residentMemoryAfterBytes'])} · '
                'Thermal ${benchmark['thermalBefore']} → ${benchmark['thermalAfter']}',
              ),
              if (result.reportPath != null)
                Align(
                  alignment: Alignment.centerLeft,
                  child: TextButton.icon(
                    onPressed: () => _inference.shareReport(result.reportPath!),
                    icon: const Icon(Icons.ios_share),
                    label: const Text('Export benchmark JSON'),
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }

  String _ms(Object? value) =>
      value is num ? '${value.toStringAsFixed(1)} ms' : 'n/a';

  String _mb(Object? value) =>
      value is num ? '${(value / 1048576).toStringAsFixed(1)} MB' : 'n/a';
}

class _CoverCameraPreview extends StatelessWidget {
  const _CoverCameraPreview({required this.camera});

  final CameraController camera;

  @override
  Widget build(BuildContext context) => LayoutBuilder(
    builder: (context, constraints) {
      final value = camera.value;
      final orientation = value.isRecordingVideo
          ? value.recordingOrientation ?? value.deviceOrientation
          : value.previewPauseOrientation ??
                value.lockedCaptureOrientation ??
                value.deviceOrientation;
      final landscape =
          orientation == DeviceOrientation.landscapeLeft ||
          orientation == DeviceOrientation.landscapeRight;
      final aspect = landscape
          ? camera.value.aspectRatio
          : 1 / camera.value.aspectRatio;
      var width = constraints.maxWidth;
      var height = width / aspect;
      if (height < constraints.maxHeight) {
        height = constraints.maxHeight;
        width = height * aspect;
      }
      return Center(
        child: SizedBox(
          width: width,
          height: height,
          child: CameraPreview(camera),
        ),
      );
    },
  );
}

class _ViewfinderOverlay extends StatelessWidget {
  const _ViewfinderOverlay();

  @override
  Widget build(BuildContext context) =>
      CustomPaint(painter: _CornerPainter(), child: const SizedBox.expand());
}

class _CornerPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    const inset = 38.0;
    const length = 54.0;
    for (final values in [
      (Offset(inset, inset), 1.0, 1.0),
      (Offset(size.width - inset, inset), -1.0, 1.0),
      (Offset(inset, size.height - inset), 1.0, -1.0),
      (Offset(size.width - inset, size.height - inset), -1.0, -1.0),
    ]) {
      final (origin, horizontal, vertical) = values;
      canvas.drawPath(
        Path()
          ..moveTo(origin.dx + horizontal * length, origin.dy)
          ..lineTo(origin.dx, origin.dy)
          ..lineTo(origin.dx, origin.dy + vertical * length),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
