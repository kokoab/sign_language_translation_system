import 'package:flutter/services.dart';

class SltInferenceResult {
  const SltInferenceResult({
    required this.glosses,
    required this.english,
    required this.literalEnglish,
    required this.renderingMode,
    required this.safeFallbackUsed,
    this.benchmark,
    this.reportPath,
    this.diagnosticPath,
  });

  factory SltInferenceResult.fromMap(Map<Object?, Object?> value) {
    final benchmarkValue = value['benchmark'];
    return SltInferenceResult(
      glosses: List<String>.from(value['glosses'] as List? ?? const []),
      english: value['english'] as String? ?? '',
      literalEnglish: value['literalEnglish'] as String? ?? '',
      renderingMode: value['renderingMode'] as String? ?? 'unknown',
      safeFallbackUsed: value['safeFallbackUsed'] as bool? ?? true,
      benchmark: benchmarkValue is Map
          ? Map<String, Object?>.from(benchmarkValue)
          : null,
      reportPath: value['reportPath'] as String?,
      diagnosticPath: value['diagnosticPath'] as String?,
    );
  }

  final List<String> glosses;
  final String english;
  final String literalEnglish;
  final String renderingMode;
  final bool safeFallbackUsed;
  final Map<String, Object?>? benchmark;
  final String? reportPath;
  final String? diagnosticPath;
}

class SltInferenceService {
  static const _channel = MethodChannel('slt_mobile_app/inference_v17');

  Future<SltInferenceResult> inferVideo({
    required String path,
    required bool mirrored,
    required bool benchmarkEnabled,
  }) async {
    final value = await _channel.invokeMapMethod<Object?, Object?>(
      'inferVideo',
      {
        'path': path,
        'mirrored': mirrored,
        'benchmarkEnabled': benchmarkEnabled,
      },
    );
    if (value == null) {
      throw PlatformException(
        code: 'empty_result',
        message: 'The iOS inference pipeline returned no result.',
      );
    }
    return SltInferenceResult.fromMap(value);
  }

  Future<bool> getBenchmarkEnabled() async =>
      await _channel.invokeMethod<bool>('getBenchmarkEnabled') ?? false;

  Future<void> setBenchmarkEnabled(bool enabled) =>
      _channel.invokeMethod<void>('setBenchmarkEnabled', {'enabled': enabled});

  Future<void> shareReport(String path) =>
      _channel.invokeMethod<void>('shareReport', {'path': path});

  /// Native LIVE page (landscape camera, Start/Stop). Completes when it closes, with the page
  /// the user picked in its nav: {"navigate": "HOME" | "GLOSSES" | ...}.
  Future<Map<String, Object?>> openLiveReel() async =>
      Map<String, Object?>.from(await _channel.invokeMapMethod<String, Object?>('openLiveReel') ?? const {});

  /// Native practice round over [glosses]: {"navigate", "score", "attempts", "total", "missed", "completed"}.
  Future<Map<String, Object?>> openPractice(List<String> glosses) async => Map<String, Object?>.from(
        await _channel.invokeMapMethod<String, Object?>('openPractice', {'glosses': glosses}) ?? const {},
      );

  /// {"ready": bool, "state": "idle" | "loading" | "ready" | "failed", "error": String?}
  Future<Map<String, Object?>> liveReelStatus() async =>
      Map<String, Object?>.from(await _channel.invokeMapMethod<String, Object?>('liveReelStatus') ?? const {});

  /// Android only: hand-images mode {"mode": "AUTO" | "ON" | "OFF", "active": bool?, "crop_encode_ms": double?}.
  Future<Map<String, Object?>> getHandImageMode() async =>
      Map<String, Object?>.from(await _channel.invokeMapMethod<String, Object?>('getHandImageMode') ?? const {});

  /// Android only: set the mode; completes after the models are rebuilt with {"active": bool}.
  Future<Map<String, Object?>> setHandImageMode(String mode) async => Map<String, Object?>.from(
        await _channel.invokeMapMethod<String, Object?>('setHandImageMode', {'mode': mode}) ?? const {},
      );

  /// Live sessions saved on this device, newest first.
  Future<List<Map<String, dynamic>>> listSessions() async {
    final rows = await _channel.invokeListMethod<Object?>('listSessions') ?? const [];
    return [for (final row in rows) Map<String, dynamic>.from(row as Map)];
  }
}
